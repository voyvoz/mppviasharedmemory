use crate::core::core::CACHELINE_SIZE;
use crate::core::host::MAX_HOSTS;
use crate::core::{CXLRBParams, CXLRingBuffer, Cacheline, RingError, cxl_read_cacheline_sync};
use crate::cxl::{CXLSharedState, cxl_typed_from_cl_sync, cxl_typed_new_sync};
use crate::cxl_v2::CXLSizedObject;
use crate::cxl_write_sync;
use crate::mutex::{CXLMutexGuarded, CXLMutexImpl, opts::MutexOptions};
use crate::utils::timestamps::SpinCounter;
use allocator_api2::alloc::Allocator;
use anyhow::Result;
use log::{debug, trace};
use std::alloc::Layout;

// ------------------------------------------------------
// PipeOptions
// ------------------------------------------------------

#[derive(Debug, Copy, Clone)]
pub struct PipeOptions {
    rb_opts: CXLRBParams,
}

impl PipeOptions {
    pub fn default() -> Self {
        PipeOptions {
            rb_opts: CXLRBParams::default(),
        }
    }
    pub fn with_rb_opts(mut self, rb_opts: CXLRBParams) -> Self {
        self.rb_opts = rb_opts;
        self
    }
}

impl CXLSizedObject for PipeOptions {
    fn required_cachelines(&self) -> usize {
        // ID + (2 * cacheline of a ringbuffer)
        1 + (2 * self.rb_opts.required_cachelines())
    }
}

// ------------------------------------------------------
// Pipe
// ------------------------------------------------------

const PIPE_ID_IDX: usize = 7;

#[derive(Debug)]
pub struct Pipe {
    base_cl: *mut Cacheline,
    rx: CXLRingBuffer,
    tx: CXLRingBuffer,

    pub options: PipeOptions,
}

impl Pipe {
    /// If id == 0 -> this Pipe is not used yet.
    #[inline]
    pub fn get_id(meta: &Cacheline) -> u64 {
        meta.d[PIPE_ID_IDX] as u64
    }
    #[inline]
    fn clear_id(&self) {
        cxl_write_sync!((self.base_cl => [(PIPE_ID_IDX, 0)]));
    }

    /// Creates a RingBuffer Pair at `base_cl`.
    pub fn init(base_cl: *mut Cacheline, options: PipeOptions) -> Self {
        debug!("Pipe::init(base_cl={base_cl:p})");

        cxl_typed_new_sync::<Self>(base_cl);

        let rx_cl = unsafe { base_cl.add(1) };
        let rx = CXLRingBuffer::init(rx_cl, options.rb_opts);

        let tx_cl = unsafe { rx_cl.add(rx.required_cxl_shared_state_cachelines()) };
        let tx = CXLRingBuffer::init(tx_cl, options.rb_opts);

        Pipe {
            base_cl,
            rx,
            tx,
            options,
        }
    }

    ///Treats `base_cl` as a pointer to Pipe.
    /// If `id` is given then directly set the
    pub fn from_cl_sync(
        base_cl: *mut Cacheline,
        id: Option<u64>,
        options: PipeOptions,
    ) -> Option<Self> {
        debug!("Pipe::from_cl_sync(base_cl={base_cl:p})");
        let rb_config = options.rb_opts;
        cxl_typed_from_cl_sync(base_cl, |meta| {
            let rx = CXLRingBuffer::from_cl_sync(unsafe { base_cl.add(1) }, rb_config).unwrap();
            let tx = CXLRingBuffer::from_cl_sync(
                unsafe { base_cl.add(1 + rx.required_cxl_shared_state_cachelines()) },
                rb_config,
            )
            .unwrap();
            let rmeote_id = Pipe::get_id(&meta);
            if rmeote_id > 0 {
                Pipe {
                    base_cl,
                    rx: tx,
                    tx: rx,
                    options,
                }
            } else {
                cxl_write_sync!((base_cl, PIPE_ID_IDX, id.unwrap()));
                Pipe {
                    base_cl,
                    rx: rx,
                    tx: tx,
                    options,
                }
            }
        })
    }

    #[inline]
    pub fn try_push(&mut self, data: &Cacheline) -> Result<(), RingError> {
        self.tx.try_push(data)
    }
    #[inline]
    pub fn try_pop(&mut self) -> Result<Cacheline, RingError> {
        self.rx.try_pop()
    }

    /// Loops until a message can be popped.
    pub fn await_pop(&mut self) -> Cacheline {
        let mut spins = SpinCounter::new();
        loop {
            let res = self.rx.try_pop();
            if res.is_ok() {
                return res.unwrap();
            }
            spins.wait();
        }
    }

    /// Resets all shared state of the buffers and removes the id.
    pub fn clear(&self) {
        self.tx.clear_sync();
        self.rx.clear_sync();
        self.clear_id();
    }
}

impl CXLSharedState for Pipe {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(_n: usize) -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        unimplemented!()
    }
    fn get_cxl_type_id() -> u64 {
        0x12344321
    }
}

// ------------------------------------------------------
// NamedPipesOptions
// ------------------------------------------------------

#[derive(Debug, Copy, Clone)]
pub struct NamedPipeOptions {
    // Amount of pipe available
    pipe_count: usize,
    // Options used of pipe creation
    pipe_opts: PipeOptions,
    // Lock options
    lock_opts: MutexOptions,
}

impl NamedPipeOptions {
    // TODO Add correct host count here?
    pub fn default() -> Self {
        NamedPipeOptions {
            pipe_count: MAX_PIPES,
            pipe_opts: PipeOptions::default(),
            lock_opts: MutexOptions::default(),
        }
    }
    pub fn with_pipe_opts(mut self, opts: PipeOptions) -> Self {
        self.pipe_opts = opts;
        self
    }
}

impl CXLSizedObject for NamedPipeOptions {
    fn required_cachelines(&self) -> usize {
        // Meta + Lock + (N * Pipe)
        1 + self.lock_opts.required_cachelines()
            + (self.pipe_opts.required_cachelines() * self.pipe_count)
    }
}

// ------------------------------------------------------
// NamedPipes
// ------------------------------------------------------

pub const MAX_PIPES: usize = 12; //256;

const NP_PIPES_IDX: usize = 6;
const NP_PIPE_STRIDE_IDX: usize = 7;

#[derive(Debug)]
pub struct NamedPipes {
    pipes: usize,
    stride: usize, // Cachelines between pipes

    lock: CXLMutexImpl,
    cl_pipes: *mut Cacheline,

    // Shared state
    base_cl: *mut Cacheline,

    options: NamedPipeOptions,
}

impl NamedPipes {
    /// Create a new NamedPipe structure using default sizes.
    pub fn new(alloc: std::sync::Arc<impl Allocator>, options: NamedPipeOptions) -> Self {
        let required_cls = NamedPipes::required_cxl_shared_state_cachelines_static();
        let mem_layout = Layout::from_size_align(required_cls * CACHELINE_SIZE, 64).unwrap();
        let mem = alloc.allocate(mem_layout).unwrap();

        NamedPipes::init(mem.as_ptr() as *mut Cacheline, options)
    }

    ///
    pub fn init(base_cl: *mut Cacheline, options: NamedPipeOptions) -> Self {
        debug!(
            "[{:#}] NamedPipes::init(base_cl={base_cl:p})",
            Cacheline::ptr_to_offset(base_cl)
        );

        // Parameter
        let pipes = options.pipe_count;
        let stride = options.pipe_opts.required_cachelines();

        // Init meta data
        cxl_typed_new_sync::<Self>(base_cl);
        cxl_write_sync!((base_cl => [ (NP_PIPES_IDX, pipes), (NP_PIPE_STRIDE_IDX, stride) ]));

        // Init data structures
        let cl_lock = unsafe { base_cl.add(1) };
        let tpl = CXLMutexImpl::init(cl_lock, MAX_HOSTS); // TODO makes this an option?
        let next_cl = unsafe { cl_lock.add(tpl.required_cxl_shared_state_cachelines()) };

        for n in 0..pipes {
            // Create pipes without assigning IDs!
            Pipe::init(unsafe { next_cl.add(n * stride) }, options.pipe_opts);
        }

        NamedPipes {
            pipes,
            stride,
            base_cl,
            lock: tpl,
            cl_pipes: next_cl,
            options,
        }
    }

    ///Treat `base_cl` as a pointer to a `NamedPipes` metadata and create an object from it.
    pub fn from_cl_sync(base_cl: *mut Cacheline, options: NamedPipeOptions) -> Option<Self> {
        debug!(
            "[{:#}] NamedPipes::from_cl(base_cl={base_cl:p})",
            Cacheline::ptr_to_offset(base_cl)
        );
        cxl_typed_from_cl_sync(base_cl, |meta| {
            let cl_lock = unsafe { base_cl.add(1) };
            let tpl = CXLMutexImpl::from_cl_sync(cl_lock).unwrap();
            let next_cl = unsafe { cl_lock.add(tpl.required_cxl_shared_state_cachelines()) };

            NamedPipes {
                pipes: meta.d[NP_PIPES_IDX] as usize,
                stride: meta.d[NP_PIPE_STRIDE_IDX] as usize,
                base_cl,
                lock: tpl,
                cl_pipes: next_cl,
                options,
            }
        })
    }

    /// Uses `id` to search for a pipe that already uses that id and is available.
    /// If none is found one is created with `id`.
    /// Locks the data structure during use.
    pub fn get_or_create_pipe(&self, id: u64) -> Option<Pipe> {
        debug!("NamedPipes::get_or_create_pipe(id={id})");
        let mut free_pipe_offset = usize::MAX as usize;
        let _g = self.lock.lock_guard();
        for n in 0..self.pipes {
            let offset = self.stride * n;
            // Check Pipe n metadata
            let target_pipe_ptr = unsafe { self.cl_pipes.add(offset) };
            let pipe_meta = cxl_read_cacheline_sync(target_pipe_ptr);
            let npid = Pipe::get_id(&pipe_meta);
            trace!(
                "[{:#}] looking @ {n}: npid:{npid}, free_pipe_offset:{free_pipe_offset}",
                Cacheline::ptr_to_offset(target_pipe_ptr)
            );
            // Use the first unused slot for a potential new pipe
            if (npid == 0) && (free_pipe_offset == usize::MAX) {
                free_pipe_offset = offset;
            }
            // If we encounter a pipe with the desired ID we use that instead
            if npid == id {
                trace!("found id {id} @ {n}");
                let p = Pipe::from_cl_sync(
                    unsafe { self.cl_pipes.add(offset) },
                    None,
                    self.options.pipe_opts,
                );
                return p;
            }
        }
        if free_pipe_offset == usize::MAX {
            trace!("no slot");
            return None;
        }

        trace!(
            "id={id} not found. Using free slot @ {}",
            free_pipe_offset / Pipe::required_cxl_shared_state_cachelines_static()
        );
        let p = Pipe::from_cl_sync(
            unsafe { self.cl_pipes.add(free_pipe_offset) },
            Some(id),
            self.options.pipe_opts,
        )
        .unwrap();
        Some(p)
    }
}

impl CXLSharedState for NamedPipes {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!();
    }
    fn required_cxl_shared_state_cachelines_n(_n: usize) -> usize {
        unimplemented!();
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        self.options.required_cachelines()
    }
    fn get_cxl_type_id() -> u64 {
        0x789456
    }
}

unsafe impl Send for NamedPipes {}
unsafe impl Sync for NamedPipes {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::allocator::os::OSAlloc;
    use crate::core::Cacheline;
    use crate::core::host::set_machine_id;
    use crate::core::set_global_offset;
    use crate::utils::open_test_cxl_dev;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn base() {
        let _ = env_logger::try_init();

        let os_alloc = OSAlloc::create();

        set_global_offset(0);
        set_machine_id(1);

        let target_id = 0xb00b;

        let config = NamedPipeOptions::default();

        let np = NamedPipes::new(Arc::new(os_alloc), config);

        let np_ptr = Cacheline::ptr_to_offset(np.base_cl);

        let join_handle = thread::spawn(move || {
            set_machine_id(2);

            let my_np =
                NamedPipes::from_cl_sync(Cacheline::ptr_from_offset(np_ptr), config).unwrap();

            let mut my_pipe = my_np.get_or_create_pipe(target_id).unwrap();

            let payload = Cacheline::from_b0(0x101);
            my_pipe.try_push(&payload).unwrap();
        });

        let mut pipe = np.get_or_create_pipe(target_id).unwrap();
        assert!(Pipe::get_id(&cxl_read_cacheline_sync(pipe.base_cl)) == target_id);

        thread::sleep(Duration::from_secs(1));
        let res = pipe.try_pop().unwrap();
        assert_eq!(res.d[0], 0x101);

        join_handle.join().unwrap();

        // TODO check ref counters?
    }

    #[test]
    fn pingpong_paired_channels() {
        let (base, _) = open_test_cxl_dev();
        let base = unsafe { base.add(16 * 1024) };

        let config = NamedPipeOptions::default();
        NamedPipes::init(base, config);

        let mem_ptr = base as usize;

        let channels: Vec<u64> = vec![10, 10, 20, 20];

        let mut threads = vec![];
        for c in channels {
            threads.push(thread::spawn(move || {
                let named_pipes =
                    NamedPipes::from_cl_sync(mem_ptr as *mut Cacheline, config).unwrap();

                let mut myc = named_pipes.get_or_create_pipe(c).unwrap();

                for n in 0..64 {
                    while myc.try_push(&Cacheline::from_b0(c + n)).is_err() {}
                    let r = loop {
                        if let Ok(cl) = myc.try_pop() {
                            break cl;
                        }
                    };
                    assert_eq!(c + n, r.d[0]);
                }
            }));
            thread::sleep(Duration::from_millis(100));
        }

        for t in threads {
            t.join().unwrap();
        }
    }
}
