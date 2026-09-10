use crate::com::barrier::{CXLBarrier, CXLBarrierOptions};
use crate::com::ipc::{CXLIpc, CXLIpcOptions};
use crate::com::pipe::{NamedPipeOptions, NamedPipes};
use crate::core::Cacheline;
use crate::cxl::CXLSharedState;
use crate::cxl::cxl_typed_from_cl_sync;
use crate::cxl::{cxl_new_sized_obj, cxl_typed_get_size_cl};
use crate::cxl_v2::RuntimeContext;
use crate::cxl_v2::{CXLSizedObject, SharedMemSegmentOptions};
use log::debug;

// ------------------------------------------------------
// ComOptions
// ------------------------------------------------------

#[derive(Debug, Copy, Clone)]
pub struct ComOptions {
    // Config of the barrier
    pub barrier_opts: CXLBarrierOptions,
    // Config of the IPC
    pub ipc_opts: CXLIpcOptions,
    //
    pub include_named_pipes: bool,
    pub named_pipe_options: NamedPipeOptions,
}
impl ComOptions {
    ///
    pub fn default() -> Self {
        let ipc_opts = CXLIpcOptions::default();
        ComOptions {
            barrier_opts: CXLBarrierOptions::from_hosts(ipc_opts.hosts),
            ipc_opts,
            include_named_pipes: false,
            named_pipe_options: NamedPipeOptions::default(),
        }
    }
    ///
    pub fn for_n_hosts(num_hosts: usize) -> Self {
        let barrier_opts = CXLBarrierOptions::from_hosts(num_hosts);
        let ipc_opts = CXLIpcOptions::default().with_hosts(num_hosts);
        ComOptions {
            barrier_opts,
            ipc_opts,
            include_named_pipes: false,
            named_pipe_options: NamedPipeOptions::default(),
        }
    }
    pub fn with_ipc_options(mut self, ipc_opts: CXLIpcOptions) -> Self {
        self.ipc_opts = ipc_opts;
        self
    }
    pub fn with_named_pipes(mut self) -> Self {
        self.include_named_pipes = true;
        self
    }
}

impl CXLSizedObject for ComOptions {
    fn required_cachelines(&self) -> usize {
        // IPC + Pipe
        self.ipc_opts.required_cachelines()
    }
}

// ------------------------------------------------------
// ComManager
// ------------------------------------------------------

pub struct ComManager {
    //
    pub base_cl: *mut Cacheline,
    //
    pub barrier: CXLBarrier,
    //
    pub ipc: CXLIpc,
    //
    pub named_pipes: Option<NamedPipes>,

    //
    pub options: SharedMemSegmentOptions,
}

impl ComManager {
    pub fn init(
        base_cl: *mut Cacheline,
        options: SharedMemSegmentOptions,
        context: &RuntimeContext,
    ) -> Self {
        debug!("ComManager::init(base_cl={base_cl:p},options={options:?},context={context:?})");

        // Skip Metadata
        let mut next_cl = unsafe{base_cl.add(1)};

        // Barrier
        let cl_barrier = next_cl;
        let barrier =
            CXLBarrier::init(cl_barrier, CXLBarrierOptions::from_hosts(options.num_hosts));
        next_cl = unsafe { next_cl.add(barrier.required_cxl_shared_state_cachelines()) };
        debug!(
            "ComManager::init()   Barrier: {} <-> {}",
            Cacheline::ptr_to_offset(cl_barrier),
            Cacheline::ptr_to_offset(next_cl)
        );

        // IPC state structure
        let cl_ipc = next_cl;
        let ipc = CXLIpc::init(cl_ipc, options.com.ipc_opts);
        next_cl = unsafe { next_cl.add(ipc.required_cxl_shared_state_cachelines()) };
        debug!(
            "ComManager::init()   IPC: {} <-> {}",
            Cacheline::ptr_to_offset(cl_ipc),
            Cacheline::ptr_to_offset(next_cl)
        );

        // Named pipes stuff
        let named_pipes = if options.com.include_named_pipes {
            let cl_named_pipes_base = next_cl;
            let named_pipes = NamedPipes::init(cl_named_pipes_base, options.com.named_pipe_options);
            next_cl = unsafe { next_cl.add(named_pipes.required_cxl_shared_state_cachelines()) };
            debug!(
                "ComManager::init()   NamedPipes: {} <-> {}",
                Cacheline::ptr_to_offset(cl_named_pipes_base),
                Cacheline::ptr_to_offset(next_cl)
            );
            Some(named_pipes)
        } else {
            None
        };

        let struct_size_cl = (next_cl as u64 - base_cl as u64).div_ceil(64);
        debug!("ComManager::init()   overall size: {struct_size_cl}");

        cxl_new_sized_obj::<ComManager>(base_cl, struct_size_cl);

        ComManager {
            base_cl,
            options,
            ipc,
            barrier,
            named_pipes,
        }
    }

    pub fn from_cl_sync(base_cl: *mut Cacheline, options: SharedMemSegmentOptions) -> Option<Self> {
        cxl_typed_from_cl_sync::<Self>(base_cl, |_meta| {
            // Skip Metadata
            let mut next_cl = unsafe{base_cl.add(1)};

            let cl_barrier = next_cl;
            let barrier = CXLBarrier::from_cl(cl_barrier, CXLBarrierOptions::from_hosts(options.num_hosts)).unwrap();
            next_cl = unsafe { next_cl.add(barrier.required_cxl_shared_state_cachelines()) };
            debug!(
                "ComManager::init()   Barrier: {} <-> {}",
                Cacheline::ptr_to_offset(cl_barrier),
                Cacheline::ptr_to_offset(next_cl)
            );

            // IPC state
            let cl_ipc = next_cl;
            let ipc = CXLIpc::from_cl_sync(cl_ipc, options.com.ipc_opts).unwrap();
            next_cl = unsafe { next_cl.add(ipc.required_cxl_shared_state_cachelines()) };
            debug!(
                "ComManager::init()   IPC: {} <-> {}",
                Cacheline::ptr_to_offset(cl_ipc),
                Cacheline::ptr_to_offset(next_cl)
            );

            // Named pipes stuff
            let named_pipes = if options.com.include_named_pipes {
                let cl_named_pipes_base = next_cl;
                let named_pipes =
                    NamedPipes::from_cl_sync(cl_named_pipes_base, options.com.named_pipe_options)
                        .unwrap();
                next_cl =
                    unsafe { next_cl.add(named_pipes.required_cxl_shared_state_cachelines()) };
                debug!(
                    "ComManager::init()   NamedPipes: {} <-> {}",
                    Cacheline::ptr_to_offset(cl_named_pipes_base),
                    Cacheline::ptr_to_offset(next_cl)
                );
                Some(named_pipes)
            } else {
                None
            };

            ComManager {
                base_cl,
                options,
                ipc,
                barrier,
                named_pipes,
            }
        })
    }
}

impl CXLSharedState for ComManager {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(_n: usize) -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        cxl_typed_get_size_cl::<ComManager>(self.base_cl) as usize
    }
    fn get_cxl_type_id() -> u64 {
        0x1981
    }
}
