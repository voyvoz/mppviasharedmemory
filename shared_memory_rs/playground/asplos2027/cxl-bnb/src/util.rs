use log::info;
use mpi::collective::SystemOperation;
use mpi::environment::Universe;
use mpi::topology::Communicator;
use mpi::traits::CommunicatorCollectives;
use shm_cxl::core::invalidate;
use shm_cxl::core::load_and_store_fence;
use shm_cxl::core::prefetch_into_l0;
use shm_cxl::cxl_v2::CXLSharedMemSegment;
use shm_cxl::cxl_v2::SharedMemSegmentOptions;
use shm_cxl::cxl_v2::create_shared_memory_segment;
use shm_cxl::cxl_write_flush;
use shm_cxl::utils::timestamps::CxlMetricTimer;
use std::cell::RefCell;
use std::ops::Sub;

thread_local! {
    pub static FABRIC: RefCell<Option<TransportVariant>> = RefCell::new(Some(TransportVariant::NoTransport));
}
fn with_fabric<F, R>(f: F) -> R
where
    F: FnOnce(&mut TransportVariant) -> R,
{
    FABRIC.with(|cell| {
        let mut borrowed = cell.borrow_mut();
        if let Some(ref mut fabric) = *borrowed {
            f(fabric)
        } else {
            panic!("Fabric not initialized!");
        }
    })
}

pub fn on_fabric_id() -> u32 {
    with_fabric(|f| f.id())
}
pub fn is_root() -> bool {
    on_fabric_id() == 0
}

pub fn on_fabric_hosts() -> usize {
    with_fabric(|f| f.num_hosts())
}

pub fn all_reduce_max(local: u64) -> u64 {
    with_fabric(|f| f.all_reduce_max(local))
}

pub fn termination_check(can_terminate: bool) -> bool {
    with_fabric(|f| f.termination_check(can_terminate))
}

pub fn barrier() {
    with_fabric(|f| f.barrier());
}

pub fn reset() {
    with_fabric(|f| f.reset())
}

pub fn create_mpi_transport() {
    let universe = mpi::initialize().unwrap();

    info!(
        ">> MPI Variant  rank {} / world {}",
        universe.world().rank(),
        universe.world().size()
    );
    FABRIC.with(|f| {
        *f.borrow_mut() = Some(TransportVariant::Mpi(universe));
    });
}

pub fn create_cxl_transport(variant: u32) {
    let (dev, size) = shm_cxl::utils::open_test_cxl_dev();
    let options = SharedMemSegmentOptions::default_no_mem();
    let segment = create_shared_memory_segment(dev, size, options);
    info!(
        ">> CXL Variant({variant}) id {} / hosts {}",
        segment.id(),
        segment.num_hosts()
    );
    let cxl_fabric = match variant {
        0 => TransportVariant::CXL((segment, State::default())),
        _ => TransportVariant::CXLv2((segment, State::default())),
    };
    FABRIC.set(Some(cxl_fabric));
}

trait Fabric {
    fn id(&self) -> u32;
    fn num_hosts(&self) -> usize;

    fn all_reduce_max(&mut self, local_result: u64) -> u64;

    fn termination_check(&mut self, can_terminate: bool) -> bool;

    fn barrier(&mut self);

    fn reset(&mut self);
}

#[derive(Default)]
struct State {
    lh_state: u8,
}

enum TransportVariant {
    NoTransport,
    Mpi(Universe),
    CXL((CXLSharedMemSegment, State)),
    CXLv2((CXLSharedMemSegment, State)),
}

impl Fabric for TransportVariant {
    fn id(&self) -> u32 {
        match self {
            TransportVariant::NoTransport => 0,
            TransportVariant::Mpi(universe) => {
                let comm = universe.world();
                comm.rank() as u32
            }
            // MPI ranks are 0-based so convert as long as CXL machine_ids are 1 based
            TransportVariant::CXL((segment, _state)) => segment.id() as u32 - 1,
            TransportVariant::CXLv2((segment, _state)) => segment.id() as u32 - 1,
        }
    }
    fn num_hosts(&self) -> usize {
        match self {
            TransportVariant::NoTransport => 1,
            TransportVariant::Mpi(universe) => {
                let comm = universe.world();
                comm.size() as usize
            }
            TransportVariant::CXL((segment, _state)) => segment.num_hosts(),
            TransportVariant::CXLv2((segment, _state)) => segment.num_hosts(),
        }
    }

    fn all_reduce_max(&mut self, local_result: u64) -> u64 {
        match self {
            TransportVariant::NoTransport => local_result,
            TransportVariant::Mpi(universe) => {
                let comm = universe.world();
                let mut global_result = 0;
                comm.all_reduce_into(&local_result, &mut global_result, SystemOperation::max());
                global_result
            }
            TransportVariant::CXL((segment, _state)) => {
                // This variant is "stateless" outside of the function call so we can reduce anything in here

                let mut global_result = local_result;
                let sharing_cls = segment.next_free_cl();
                let id = segment.id();

                // Write
                cxl_write_flush!((sharing_cls.add(id.sub(1).into()), local_result));
                for i in 0..segment.config.num_hosts {
                    if i + 1 != id.into() {
                        invalidate(unsafe { sharing_cls.add(i) });
                    }
                }
                load_and_store_fence();

                // Barrier
                segment.com.barrier.wait_for_all();

                // Read others
                {
                    let _timer = CxlMetricTimer::load();
                    for i in 0..segment.config.num_hosts {
                        if i + 1 != id.into() {
                            let v = unsafe { *sharing_cls.add(i) };
                            if v.d[0] > global_result.into() {
                                global_result = v.d[0];
                            }
                        }
                    }
                }
                global_result
            }
            TransportVariant::CXLv2((segment, state)) => {
                // A latency hiding variant
                // this makes the call statefull so we need to keep track of possible states

                let mut global_result = local_result;
                // TODO maybe not hardcode this but should work as long as there are not 8 hosts
                let sharing_cls = segment.next_free_cl();
                let id = segment.id();

                // Write
                cxl_write_flush!((sharing_cls.add(id.sub(1).into()), local_result));

                match state.lh_state {
                    0 => {
                        // State 1 - invalidate
                        for i in 0..segment.config.num_hosts {
                            if i + 1 != id.into() {
                                invalidate(unsafe { sharing_cls.add(i) });
                            }
                        }
                        // Same ordering point as the blocking variant: the
                        // flush must complete before the next call's prefetch
                        // can pull the line back in, or the prefetch caches
                        // the pre-invalidate copy and the read returns it.
                        load_and_store_fence();
                        state.lh_state = 1;
                        return local_result;
                    }
                    1 => {
                        // State 2 - Prefetch
                        for i in 0..segment.config.num_hosts {
                            if i + 1 != id.into() {
                                prefetch_into_l0(unsafe { sharing_cls.add(i) });
                            }
                        }
                        state.lh_state = 2;
                        return local_result;
                    }
                    _ => {
                        // State 3 - Read
                        {
                            let _timer = CxlMetricTimer::load();
                            for i in 0..segment.config.num_hosts {
                                if i + 1 != id.into() {
                                    let v = unsafe { *sharing_cls.add(i) };
                                    if v.d[0] > global_result.into() {
                                        global_result = v.d[0];
                                    }
                                }
                            }
                        }
                        state.lh_state = 0;
                    }
                }
                global_result
            }
        }
    }

    fn termination_check(&mut self, can_terminate: bool) -> bool {
        let local_terminate: u64 = if can_terminate { 0 } else { 1 };
        match self {
            TransportVariant::NoTransport => can_terminate,
            TransportVariant::Mpi(universe) => {
                let comm = universe.world();
                let mut global_terminate = 0;
                comm.all_reduce_into(
                    &local_terminate,
                    &mut global_terminate,
                    SystemOperation::max(),
                );
                global_terminate == 0
            }
            TransportVariant::CXL((segment, _state)) => {
                // This variant is "stateless" outside of the function call so we can reduce anything in here

                let mut global_result = local_terminate;
                let sharing_cls = unsafe { segment.next_free_cl().add(8) };
                let id = segment.id();

                // Write
                cxl_write_flush!((sharing_cls.add(id.sub(1).into()), local_terminate));
                for i in 0..segment.config.num_hosts {
                    if i + 1 != id.into() {
                        invalidate(unsafe { sharing_cls.add(i) });
                    }
                }
                load_and_store_fence();

                // Barrier
                segment.com.barrier.wait_for_all();

                // Read others
                {
                    let _timer = CxlMetricTimer::load();
                    for i in 0..segment.config.num_hosts {
                        if i + 1 != id.into() {
                            let v = unsafe { *sharing_cls.add(i) };
                            if v.d[0] > global_result.into() {
                                global_result = v.d[0];
                            }
                        }
                    }
                }
                global_result == 0
            }
            TransportVariant::CXLv2((segment, _state)) => {
                // This variant is the same as CXL as we need the result of the check directly.
                // cant hide the latency with this kind of check.
                // Same 8-cacheline offset as CXL: the bound-sharing lines must
                // not be overwritten, or the latency-hiding drain reads a
                // termination flag where it expects a peer's best bound.

                let mut global_result = local_terminate;
                let sharing_cls = unsafe { segment.next_free_cl().add(8) };
                let id = segment.id();

                // Write
                cxl_write_flush!((sharing_cls.add(id.sub(1).into()), local_terminate));
                for i in 0..segment.config.num_hosts {
                    if i + 1 != id.into() {
                        invalidate(unsafe { sharing_cls.add(i) });
                    }
                }
                load_and_store_fence();

                // Barrier
                segment.com.barrier.wait_for_all();

                // Read others
                {
                    let _timer = CxlMetricTimer::load();
                    for i in 0..segment.config.num_hosts {
                        if i + 1 != id.into() {
                            let v = unsafe { *sharing_cls.add(i) };
                            if v.d[0] > global_result.into() {
                                global_result = v.d[0];
                            }
                        }
                    }
                }
                global_result == 0
            }
        }
    }

    fn barrier(&mut self) {
        match self {
            TransportVariant::NoTransport => {}
            TransportVariant::Mpi(universe) => {
                let comm = universe.world();
                comm.barrier();
            }
            TransportVariant::CXL((segment, _state)) => segment.com.barrier.wait_for_all(),
            TransportVariant::CXLv2((segment, _state)) => segment.com.barrier.wait_for_all(),
        }
    }

    fn reset(&mut self) {
        match self {
            TransportVariant::NoTransport => {}
            TransportVariant::Mpi(_universe) => {}
            TransportVariant::CXL((segment, _state)) => {
                // Reset bound sharing
                let sharing_cls = segment.next_free_cl();
                for i in 0..segment.num_hosts() {
                    cxl_write_flush!((sharing_cls.add(i), 0));
                }
                // Reset termination check
                let termination_cls = unsafe { segment.next_free_cl().add(8) };
                for i in 0..segment.num_hosts() {
                    cxl_write_flush!((termination_cls.add(i), 0));
                }
            }
            TransportVariant::CXLv2((segment, _state)) => {
                // Reset bound sharing
                let sharing_cls = segment.next_free_cl();
                for i in 0..segment.num_hosts() {
                    cxl_write_flush!((sharing_cls.add(i), 0));
                }
                // Reset termination check
                let termination_cls = unsafe { segment.next_free_cl().add(8) };
                for i in 0..segment.num_hosts() {
                    cxl_write_flush!((termination_cls.add(i), 0));
                }
            }
        }
    }
}
