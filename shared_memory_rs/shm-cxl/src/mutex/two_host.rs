use crate::core::host::PeerID;
use crate::core::{Cacheline, cxl_read_cacheline_sync};
use crate::cxl_write_sync;
use crate::mutex::common::CXLMutex;
use llfree::TreesLock;
use log::trace;

/*
 * TODO add ability to set bit index for a lock
 * -> allows multiple different locks within the same data structure!
 *
 */

/// Implementations
///
/// 2 Hosts Locks
///

/// At the moment we only support two participants for a CXL shared memory lock.
const MAX_ID: usize = 2;

/// Implementaiton of a classic Dekker mutex Algorithm.
#[derive(Debug)]
pub struct DekkersLock {
    /// local ID
    id: usize,
    /// CXL resident memory
    turn: *mut Cacheline,
    wants_to_enter: [*mut Cacheline; MAX_ID],
}

impl CXLMutex for DekkersLock {
    fn init(base_ptr: *mut Cacheline) -> Result<Self, Box<dyn std::error::Error>> {
        let id = PeerID::whoami();

        let l = DekkersLock {
            id: (id.machine_id % 2) as usize,
            turn: base_ptr,
            wants_to_enter: [unsafe { base_ptr.offset(1) }, unsafe { base_ptr.offset(2) }],
        };

        cxl_write_sync!((l.wants_to_enter[l.id], 0), (l.turn, 0));

        Ok(l)
    }
}
impl TreesLock for DekkersLock {
    fn lock(&self) {
        let other_id: u64 = self.id as u64 ^ 1;

        cxl_write_sync!((self.wants_to_enter[self.id], 1));

        while cxl_read_cacheline_sync(self.wants_to_enter[other_id as usize]).d[0] != 0 {
            if cxl_read_cacheline_sync(self.turn).d[0] == other_id {
                cxl_write_sync!((self.wants_to_enter[self.id], 0));

                while cxl_read_cacheline_sync(self.turn).d[0] == other_id {}

                cxl_write_sync!((self.wants_to_enter[self.id], 1));
            }
        }
    }
    fn unlock(&self) {
        cxl_write_sync!(
            (self.wants_to_enter[self.id], 0),
            (self.turn, self.id as u64 ^ 1)
        );
    }
}

/// All state is stored on CXL so its safe to send pointer around as long
/// as we dont remove the lock from the shared memory.
unsafe impl Send for DekkersLock {}
unsafe impl Sync for DekkersLock {}

/// Implementaiton of a Peterson mutex Algorithm.
/// TODO apparently not correct on non x86 ISA? -> https://github.com/hernanponcedeleon/Dat3M
#[derive(Copy, Clone, Debug)]
pub struct PetersonLock {
    /// local ID
    id: usize,
    /// CXL resident memory
    flags: [*mut Cacheline; MAX_ID],
    turn: *mut Cacheline,
}
impl CXLMutex for PetersonLock {
    fn init(base_ptr: *mut Cacheline) -> Result<Self, Box<dyn std::error::Error>> {
        let l = PetersonLock {
            id: (PeerID::whoami().machine_id % 2) as usize,
            turn: base_ptr,
            flags: [unsafe { base_ptr.offset(1) }, unsafe { base_ptr.offset(2) }],
        };

        cxl_write_sync![(l.flags[l.id], 0), (l.turn, 0)];

        Ok(l)
    }
}
impl TreesLock for PetersonLock {
    fn lock(&self) {
        trace!(">> lock({:?})", self.id);
        let other_id: u64 = self.id as u64 ^ 1;

        cxl_write_sync!((self.flags[self.id], 1), (self.turn, other_id));

        while cxl_read_cacheline_sync(self.flags[other_id as usize]).d[0] == 1
            && cxl_read_cacheline_sync(self.turn).d[0] == other_id
        {}
    }
    fn unlock(&self) {
        trace!(">> unlock({:?})", self.id);
        cxl_write_sync!((self.flags[self.id], 0));
    }
}

impl PetersonLock {
    pub fn print_state(&self) {
        Cacheline::print_hex(self.turn);
        Cacheline::print_hex(self.flags[0]);
        Cacheline::print_hex(self.flags[1]);
    }
}

/// All state is stored on CXL so its safe to send pointer around as long
/// as we dont remove the lock from the shared memory.
unsafe impl Send for PetersonLock {}
unsafe impl Sync for PetersonLock {}

//##########################
//  Tests
//#########################

#[cfg(test)]
mod test_two_hosts {
    use super::*;
    use crate::core::cl_set_sync;
    use crate::core::host::set_machine_id;
    use crate::utils::open_test_cxl_dev;
    use std::thread;

    #[test]
    fn peterson() {
        let (mem, _) = open_test_cxl_dev();

        // Set the lock cachelines to something wrong
        cl_set_sync(mem, 0x10101);
        cl_set_sync(unsafe { mem.add(1) }, 0x10101);
        cl_set_sync(unsafe { mem.add(2) }, 0x10101);
        let counter = unsafe { mem.add(4) };
        cxl_write_sync!((counter, 0));

        set_machine_id(1);

        //
        let lock = PetersonLock::init(mem).unwrap();
        lock.print_state();

        lock.lock();
        cxl_write_sync!((counter, cxl_read_cacheline_sync(counter).d[0] + 1));
        lock.unlock();

        lock.print_state();
        Cacheline::print_hex(counter);

        lock.lock();
        cxl_write_sync!((counter, cxl_read_cacheline_sync(counter).d[0] + 1));
        lock.unlock();

        lock.print_state();
        Cacheline::print_hex(counter);

        println!("<<+>>");

        // New thread acts as host B
        let mem_u = mem as usize;
        let counter_u = counter as usize;
        let h = thread::spawn(move || {
            let mem_ptr = mem_u as *mut Cacheline;
            let counter_ptr = counter_u as *mut Cacheline;

            let barrier_b = unsafe { mem_ptr.add(4) };

            set_machine_id(2);
            let lock_b = PetersonLock::init(mem_ptr).unwrap();

            lock_b.lock();
            cxl_write_sync!((counter_ptr, cxl_read_cacheline_sync(counter_ptr).d[0] + 1));
            lock_b.unlock();

            while cxl_read_cacheline_sync(barrier_b).d[0] == 3 {}

            println!("<<+B+>>");
        });

        // Await the barrier
        while cxl_read_cacheline_sync(counter).d[0] == 2 {}

        // We act as host A
        lock.lock();
        cxl_write_sync!((counter, cxl_read_cacheline_sync(counter).d[0] + 1));
        lock.unlock();

        println!("<<+A+>>");
        h.join().unwrap();

        Cacheline::print_hex(counter);
        assert!(cxl_read_cacheline_sync(counter).d[0] == 4);
    }
}
