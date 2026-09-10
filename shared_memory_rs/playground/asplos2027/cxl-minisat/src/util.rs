use crate::sat::LocalResult;
use crate::Lit;
use mpi::collective::Root;
use mpi::environment::Universe;
use mpi::point_to_point::Destination;
use mpi::point_to_point::Source;
use mpi::topology::Communicator;
use shm_cxl::core::CXLRBParams;
use shm_cxl::core::cxl_read_cacheline_sync;
use shm_cxl::core::cxl_write_cl_flush;
use shm_cxl::core::Cacheline;
use shm_cxl::core::invalidate;
use shm_cxl::core::prefetch_into_l0;
use shm_cxl::cxl_v2::create_shared_memory_segment;
use shm_cxl::cxl_v2::CXLSharedMemSegment;
use shm_cxl::cxl_v2::SharedMemSegmentOptions;
use shm_cxl::cxl_write_sync;
use std::cell::RefCell;
use std::cmp::max;

#[cfg(not(target_os = "linux"))]
pub fn mem_used_peak() -> i32 {
    -1
}

#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::Read;
#[cfg(target_os = "linux")]
use std::process;
use std::vec;
#[cfg(target_os = "linux")]
pub fn mem_used_peak() -> i32 {
    let mut buf = String::new();
    let mut stats = File::open(&format!("/proc/{}/status", process::id()))
        .expect("Couldn't open /proc/../status");
    stats
        .read_to_string(&mut buf)
        .expect("Couldn't read from /proc/../status");
    let line = buf
        .lines()
        .filter(|line| line.starts_with("VmPeak:"))
        .next()
        .expect("Couldn't find VmPeak in /proc/../status");
    let mem_kb: String = line.chars().filter(|c| c.is_digit(10)).collect();
    let mem_kb: usize = mem_kb.parse().unwrap();

    mem_kb as i32
}

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

pub fn share_clause(clause: &[Lit]) -> bool {
    with_fabric(|f| f.share_clause(clause))
}

pub fn probe_for_clauses() -> Option<Vec<Lit>> {
    with_fabric(|f| f.probe_for_clauses())
}

pub fn check_termination() -> bool {
    with_fabric(|f| f.check_termination())
}

pub fn broadcast_termination() {
    with_fabric(|f| f.broadcast_termination());
}

pub fn all_reduce(local_result: LocalResult) {
    with_fabric(|f| f.all_reduce(local_result));
}

pub fn cxl_fast_learn() -> Option<Vec<Lit>> {
    with_fabric(|f| {
        match f {
            TransportVariant::CXL((segment, state)) => {
                match segment.com.ipc.try_pop_all_from_any_with_state_machine().unwrap() {
                    None => None,
                    Some((_, mut res)) => {
                        let mut buf = vec![];
                        for i in 0..res.len(){
                            let u32_view: &mut [u32] = unsafe {
                                std::slice::from_raw_parts_mut(res[i].d.as_mut_ptr() as *mut u32, 16)
                            };
                            for i in 0..16 {
                                if u32_view[i] != 0 {
                                    buf.push(Lit(u32_view[i]));
                                }
                            }
                        }
                        Some(buf)
                    }
                }

            }
            _=>{
                // Not available on other transports
                None
            }
        }
    })
}

pub fn create_mpi_transport() {
    println!("MPI Variant");
    let universe = mpi::initialize().unwrap();
    FABRIC.with(|f| {
        *f.borrow_mut() = Some(TransportVariant::Mpi(universe));
    });
}

pub fn create_cxl_transport() {
    println!("CXL Variant");
    let (dev, size) = shm_cxl::utils::open_test_cxl_dev();
    let mut options = SharedMemSegmentOptions::default_no_mem();
    options.com.ipc_opts = options.com.ipc_opts.with_ringbuffer_options(CXLRBParams::default().with_capacity(100000));
    let segment = create_shared_memory_segment(dev, size, options);
    let signal_cl = segment.next_free_cl();
    cxl_write_sync!((signal_cl, 0));
    let cxl_fabric = TransportVariant::CXL((segment, State::default()));
    FABRIC.set(Some(cxl_fabric));
}

trait Fabric {
    fn id(&self) -> u32;
    // To share clauses
    /// Clauses are guaranteed to be smaller than 10 Lits for now
    fn share_clause(&mut self, clause: &[Lit]) -> bool;
    fn probe_for_clauses(&mut self) -> Option<Vec<Lit>>;
    // To tell others to stop once we found SAT
    fn broadcast_termination(&self);
    fn check_termination(&mut self) -> bool;
    // To send all result data to the lead process
    fn all_reduce(&self, local_result: LocalResult);
}

const MPI_TAG_CLAUSE: i32 = 1;
const MPI_TAG_TERMINATION: i32 = 2;

#[derive(Default)]
struct State {
    term_state: u8,
    term_last_val: bool,
}

enum TransportVariant {
    NoTransport,
    Mpi(Universe),
    CXL((CXLSharedMemSegment, State)),
}

impl Fabric for TransportVariant {
    fn id(&self) -> u32 {
        match self {
            TransportVariant::NoTransport => 0,
            TransportVariant::Mpi(universe) => {
                let comm = universe.world();
                comm.rank() as u32
            }
            TransportVariant::CXL((segment, state)) => segment.id() as u32,
        }
    }

    fn share_clause(&mut self, clause: &[Lit]) -> bool {
        match self {
            TransportVariant::NoTransport => {
                return false;
            }
            TransportVariant::Mpi(universe) => {
                let comm = universe.world();
                let len = clause.len() as i32;
                let data: Vec<i32> = clause.iter().map(|l| l.0 as i32).collect();

                let rank = comm.rank();
                let size = comm.size();
                for dest in 0..size {
                    if dest != rank {
                        let process = comm.process_at_rank(dest);
                        process.send_with_tag(&[len], MPI_TAG_CLAUSE);
                        process.send_with_tag(&data[..], MPI_TAG_CLAUSE);
                    }
                }

                return true;
            }
            TransportVariant::CXL((segment, state)) => {
                let mut data = Cacheline::empty();
                let u32_view: &mut [u32] = unsafe {
                    std::slice::from_raw_parts_mut(data.d.as_mut_ptr() as *mut u32, 16)
                };
                for i in 0..clause.len() {
                    u32_view[i] = clause[i].0;
                }
                // V1 - send via IPC?
                segment.com.ipc.try_broadcast(&data).unwrap();

                return true;
            }
        }
    }
    fn probe_for_clauses(&mut self) -> Option<Vec<Lit>> {
        match self {
            TransportVariant::NoTransport => {
                return None;
            }
            TransportVariant::Mpi(universe) => {
                let comm = universe.world();
                let rank = comm.rank();
                let size = comm.size();

                let mut rank_lit_buffer = vec![];

                for src in 0..size {
                    if src != rank {
                        let process = comm.process_at_rank(src);

                        while let Some(_) = process.immediate_probe_with_tag(MPI_TAG_CLAUSE) {
                            let mut len_buf = [0i32; 1];
                            process.receive_into_with_tag(&mut len_buf, MPI_TAG_CLAUSE);
                            let len = len_buf[0] as usize;

                            let mut lits_buf = vec![0i32; len];
                            process.receive_into_with_tag(&mut lits_buf[..], MPI_TAG_CLAUSE);

                            lits_buf
                                .iter()
                                .for_each(|&i| rank_lit_buffer.push(Lit(i as u32)));
                        }
                    }
                }
                Some(rank_lit_buffer)
            }
            TransportVariant::CXL((segment, state)) => {
                // V1 - recv from all peers?
                let mut buf = vec![];
                loop {
                    match segment.com.ipc.try_pop_any() {
                        Err(_e) => {
                            return None;
                        }
                        Ok(opt_res)=>{
                            match opt_res {
                                None => break,
                                Some((_, mut res)) => {
                                    let u32_view: &mut [u32] = unsafe {
                                        std::slice::from_raw_parts_mut(res.d.as_mut_ptr() as *mut u32, 16)
                                    };
                                    for i in 0..16 {
                                        if u32_view[i] != 0 {
                                            buf.push(Lit(u32_view[i]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Some(buf)
            }
        }
    }


    fn check_termination(&mut self) -> bool {
        let res = match self {
            TransportVariant::NoTransport => false,
            TransportVariant::Mpi(universe) => {
                let comm = universe.world();
                for src in 0..comm.size() {
                    if src != comm.rank() {
                        if let Some(_) = comm
                            .process_at_rank(src)
                            .immediate_probe_with_tag(MPI_TAG_TERMINATION)
                        {
                            let mut signal = [0i32; 1];
                            comm.process_at_rank(src)
                                .receive_into_with_tag(&mut signal, MPI_TAG_TERMINATION);
                            if signal[0] == 1 {
                                return true;
                            }
                        }
                    }
                }
                false
            }
            TransportVariant::CXL((segment, state)) => {
                let signal_cl = segment.next_free_cl();

                match state.term_state {
                    0 => {
                        state.term_state = 1;
                        state.term_last_val =
                        cxl_read_cacheline_sync(signal_cl).d[0] == 1;
                    }
                    1 => {
                        invalidate(signal_cl);
                        state.term_state = 2;
                    }
                    2 => {
                        prefetch_into_l0(signal_cl);
                        state.term_state = 3;
                    }
                    3=>{
                        state.term_last_val = unsafe {*signal_cl}.d[0] == 1;
                        state.term_state = 1;
                    }
                    _=>{}
                };
                state.term_last_val
            }
        };
        if res == true {
            println!("Someone else found SAT");
        }
        res
    }
    fn broadcast_termination(&self) {
        match self {
            TransportVariant::NoTransport => {
                return;
            }
            TransportVariant::Mpi(universe) => {
                let comm = universe.world();
                println!("Found SAT -> Broadcast stop");
                let signal = [1i32];
                for dest in 0..comm.size() {
                    if dest != comm.rank() {
                        comm.process_at_rank(dest)
                            .send_with_tag(&signal, MPI_TAG_TERMINATION);
                    }
                }
            }
            TransportVariant::CXL((segment, state)) => {
                let signal_cl = segment.next_free_cl();
                cxl_write_cl_flush(signal_cl, &Cacheline::from_b0(1));
            }
        }
    }


    fn all_reduce(&self, local_result: LocalResult) {
        match self {
            TransportVariant::NoTransport => {
                println!("{local_result:?}");
            }
            TransportVariant::Mpi(universe) => {
                let comm = universe.world();
                let root = comm.process_at_rank(0);
                if comm.rank() == 0 {
                    let mut results = vec![local_result; comm.size() as usize];

                    root.gather_into_root(&local_result, &mut results);

                    let mut overall_res = 0;
                    for (rank, result) in results.iter().enumerate() {
                        println!("[{rank}] result: {:?}", result);
                        overall_res = max(result.sat, overall_res);
                    }
                    let res = if overall_res == 1 {
                        "SAT"
                    } else if overall_res == 0 {
                        "UnSAT"
                    } else {
                        "Interrupted"
                    };
                    println!(">> {res} <<");
                } else {
                    root.gather_into(&local_result);
                }
            }
            TransportVariant::CXL((segment, state)) => {
                let id = segment.id();
                println!("[{id}] {local_result:?}");
                // TODO
            }
        }
    }
}
