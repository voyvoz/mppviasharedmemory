use crate::core::host::{MAX_HOSTS, PeerID, get_machine_id};
use crate::core::{Cacheline, cl_set_zero_sync, cxl_read_cacheline_sync, cxl_write_cl_sync};
use crate::cxl::{CXLSharedState, cxl_typed_from_cl_sync, cxl_typed_new_sync};
use crate::cxl_write_sync;
use crate::mutex::common::CXLMutex;
use llfree::TreesLock;
use log::{debug, trace};
use std::panic;

///
/// N Host Locks
///

const LOCKED: u64 = 0x1;
const UNLOCKED: u64 = 0x0;

///
#[derive(Debug)]
pub struct PetersonLightLock {
    prev_index: usize,
    state: *mut Cacheline,
}

impl PetersonLightLock {
    pub fn from_ptr(addr: *mut Cacheline, prev_index: usize) -> Self {
        PetersonLightLock {
            prev_index,
            state: addr,
        }
    }

    #[inline]
    fn turn(&self) -> *mut Cacheline {
        self.state
    }
    #[inline]
    fn flag_0(&self) -> *mut Cacheline {
        unsafe { self.state.add(1) }
    }
    #[inline]
    fn flag_1(&self) -> *mut Cacheline {
        unsafe { self.state.add(2) }
    }
    #[inline]
    fn my_idx(&self) -> usize {
        // On inner node locks we need to determine the idx based on the previous lock position
        // If we came here from the left side we need to lock the left side, same goes for the right side
        if self.prev_index != 0 {
            self.prev_index % 2
        } else {
            // machine_id is > 0
            // On leaf locks this is correct
            ((PeerID::whoami().machine_id - 1) % 2) as usize
        }
    }

    fn my_flag(&self) -> *mut Cacheline {
        match self.my_idx() {
            0 => self.flag_0(),
            _ => self.flag_1(),
        }
    }
    fn other_flag(&self) -> *mut Cacheline {
        match self.my_idx() {
            0 => self.flag_1(),
            _ => self.flag_0(),
        }
    }

    fn is_my_flag(&self) -> bool {
        let cl = cxl_read_cacheline_sync(PetersonLightLock::my_flag(&self));
        let i = PeerID::from_bytes([cl.d[6], cl.d[7]]);
        i.eq(&PeerID::whoami())
    }

    /// Sets the shared state to 0
    fn clear(&self) {
        cl_set_zero_sync(self.turn());
        cl_set_zero_sync(self.flag_0());
        cl_set_zero_sync(self.flag_1());
    }

    /// Sets the PeerID of the given Flag. Used to determine correctness when unlocking the lock again.
    pub fn set_peerid_for_flag(&self, flag: usize, id: PeerID) {
        let f = match flag {
            0 => self.flag_0(),
            _ => self.flag_1(),
        };
        let b = id.to_bytes();
        cxl_write_sync!((f, 6, b[0]), (f, 7, b[1]));
    }

    pub fn print_state(&self) {
        println!(
            "[{:#}] PetersonLightLock",
            Cacheline::ptr_to_offset(self.state)
        );
        Cacheline::print_hex(self.turn());
        Cacheline::print_hex(self.flag_0());
        Cacheline::print_hex(self.flag_1());
    }
}

impl CXLSharedState for PetersonLightLock {
    /// Always requires 3 Cachelines regardless of its config
    fn required_cxl_shared_state_cachelines_static() -> usize {
        3
    }
    fn required_cxl_shared_state_cachelines_n(n: usize) -> usize {
        PetersonLightLock::required_cxl_shared_state_cachelines_static()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        PetersonLightLock::required_cxl_shared_state_cachelines_static()
    }
    fn get_cxl_type_id() -> u64 {
        0x44578
    }
}

impl CXLMutex for PetersonLightLock {
    fn init(base_ptr: *mut Cacheline) -> Result<Self, Box<dyn std::error::Error>> {
        let l = PetersonLightLock {
            prev_index: 0,
            state: base_ptr,
        };
        l.clear();
        trace!("[{:}] init()", Cacheline::ptr_to_offset(l.state));
        Ok(l)
    }
}
impl TreesLock for PetersonLightLock {
    fn lock(&self) {
        trace!(
            "[{:}] [{:?}] >> lock(F{:?})",
            Cacheline::ptr_to_offset(self.state),
            PeerID::whoami(),
            self.my_idx()
        );

        let state = cxl_read_cacheline_sync(self.my_flag());
        if state.d[0] == LOCKED {
            panic!("lock already locked?");
        }

        // if we are on a leaf node then the my_idx is correct
        // If we are not on a leaf the flag depends on the previous node position

        let other_id: u64 = self.my_idx() as u64 ^ 1;

        cxl_write_sync!((self.my_flag(), 1), (self.turn(), other_id));

        let mut counter = 0;
        while cxl_read_cacheline_sync(self.other_flag()).d[0] == LOCKED
            && cxl_read_cacheline_sync(self.turn()).d[0] == other_id
        {
            counter += 1;
        }
        self.set_peerid_for_flag(self.my_idx(), PeerID::whoami());

        trace!(
            "[{:}] [{:?}] >!  got lock after {}",
            Cacheline::ptr_to_offset(self.state),
            PeerID::whoami(),
            counter
        );
    }
    fn unlock(&self) {
        trace!(
            "[{:}] [{:?}] !< unlock(F{:?})",
            Cacheline::ptr_to_offset(self.state),
            PeerID::whoami(),
            self.my_idx()
        );

        if !self.is_my_flag() {
            panic!("unlock on flag with wrong PeerID");
        }

        cxl_write_sync!((self.my_flag()=>[(0, UNLOCKED),(6,0),(7,0)]));
    }
}

/// All state is stored on CXL so its safe to send pointer around as long
/// as we dont remove the lock from the shared memory.
unsafe impl Send for PetersonLightLock {}
unsafe impl Sync for PetersonLightLock {}

/// ---------------------------------
/// The Actual N Host lock
/// ---------------------------------

const TPL_HOST_COUNT_IDX: usize = 7;

// https://metafunctor.com/post/2012-03-05-tournament-lock/tournament_lock/
// TODO maybe just make this a thin wrapper around *mut Cacheline?
#[derive(Copy, Clone, Debug)]
pub struct TournamentPetersonLock {
    /// number of participants in the tournament
    pub hosts: usize,
    /// Pointer to the first entry of the list of Peterson Locks
    /// that has "threads" entries.
    pub state: *mut Cacheline,
}

impl CXLSharedState for TournamentPetersonLock {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(n: usize) -> usize {
        // Header + number of nodes in binary tree based on number of hosts
        1 + ((n + (n / 2)) * PetersonLightLock::required_cxl_shared_state_cachelines_static())
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        TournamentPetersonLock::required_cxl_shared_state_cachelines_n(self.hosts)
    }
    fn get_cxl_type_id() -> u64 {
        0x3478
    }
}

impl TournamentPetersonLock {
    pub fn print_state(&self) {
        for i in 0..self.hosts {
            let p = PetersonLightLock::from_ptr(unsafe { self.state.add(i * 3) }, 0);
            p.print_state();
        }
    }

    /// `base` is the start of the CXL memory region where this lock will be placed.
    /// `hosts` indicates how many concurrent processes shall be supported.
    pub fn init(base_cl: *mut Cacheline, mut hosts: usize) -> Self {
        hosts = 4;
        debug!(
            "[{:}] TPL::init(base_cl={base_cl:p}, hosts={hosts})",
            Cacheline::ptr_to_offset(base_cl)
        );
        let t = TournamentPetersonLock {
            hosts,
            state: unsafe { base_cl.add(1) },
        };
        t.init_locks(1, hosts / 2, hosts / 2);
        cxl_typed_new_sync::<Self>(base_cl);
        cxl_write_sync!((base_cl, TPL_HOST_COUNT_IDX, hosts));
        t
    }

    pub fn from_cl_sync(base_cl: *mut Cacheline) -> Option<Self> {
        cxl_typed_from_cl_sync(base_cl, |state| TournamentPetersonLock {
            hosts: state.d[TPL_HOST_COUNT_IDX] as usize,
            state: unsafe { base_cl.add(1) },
        })
    }

    fn is_leaf_index(&self, index: usize) -> bool {
        index >= self.hosts / 2
    }

    fn init_locks(&self, index: usize, less_than: usize, mut size: usize) {
        if index < self.hosts {
            trace!(
                "[{:}] init_locks(index={index}, lt={less_than}, size={size}) -> is_leaf_index:{}",
                Cacheline::ptr_to_offset(self.state),
                self.is_leaf_index(index)
            );
            // init Peterson lock at specified node
            let l = PetersonLightLock::init(unsafe { self.state.add(3 * index) }).unwrap();
            // Set intended PeerIDs for flags of leaf nodes
            if self.is_leaf_index(index) {
                l.set_peerid_for_flag(0, PeerID::from_machine_id((less_than) as u16));
                l.set_peerid_for_flag(1, PeerID::from_machine_id((less_than + 1) as u16));
            }

            // init the left and right subtrees of this node
            size /= 2;
            self.init_locks(
                TournamentPetersonLock::get_child_left(index),
                less_than - size,
                size,
            );
            self.init_locks(
                TournamentPetersonLock::get_child_right(index),
                less_than + size,
                size,
            );
        }
    }

    fn lock_specific(&self, index: usize, prev_index: usize) {
        if index != 0 {
            let p = PetersonLightLock::from_ptr(unsafe { self.state.add(index * 3) }, prev_index);
            TreesLock::lock(&p);
        }
    }

    fn unlock_specific(&self, index: usize, prev_index: usize) {
        if index != 0 {
            self.unlock_specific(index / 2, index);

            let p = PetersonLightLock::from_ptr(unsafe { self.state.add(index * 3) }, prev_index);
            TreesLock::unlock(&p);
            // post-order: unlock after recursive call
            // to unlock from root down to leaf
        }
    }

    fn get_leaf_lock(&self) -> usize {
        return (self.hosts + (PeerID::whoami().machine_id - 1) as usize) / 2;
    }
    fn get_child_left(index: usize) -> usize {
        index * 2
    }
    fn get_child_right(index: usize) -> usize {
        index * 2 + 1
    }
}

impl TreesLock for TournamentPetersonLock {
    fn lock(&self) {
        // start at the leaf node lock for current thread
        let mut index = self.get_leaf_lock();

        // root is index 1, so exit the while loop when index 0 (index of
        // unitialized node lock) is reached.
        let mut prev_index = 0;
        while index != 0 {
            self.lock_specific(index, prev_index);
            prev_index = index;
            index /= 2;
        }
    }
    fn unlock(&self) {
        self.unlock_specific(self.get_leaf_lock(), 0);
    }
}

unsafe impl Send for TournamentPetersonLock {}
unsafe impl Sync for TournamentPetersonLock {}

/// ---------------------------------
/// Alternative implementation
/// ---------------------------------

#[derive(Debug)]
pub struct BakeryLock {
    myid: u16,
    entering: [*mut Cacheline; MAX_HOSTS],
    number: [*mut Cacheline; MAX_HOSTS],
    pub hosts: usize,
}

impl BakeryLock {
    pub fn init(base_cl: *mut Cacheline, hosts: usize) -> BakeryLock {
        cxl_typed_new_sync::<Self>(base_cl);
        let lock_start_cl = unsafe { base_cl.add(1) };
        let mut r = BakeryLock {
            myid: get_machine_id() - 1,
            entering: [0 as *mut Cacheline; MAX_HOSTS],
            number: [0 as *mut Cacheline; MAX_HOSTS],
            hosts: MAX_HOSTS,
        };
        for i in 0..MAX_HOSTS {
            r.entering[i] = unsafe { lock_start_cl.add(i) };
            r.number[i] = unsafe { lock_start_cl.add(i + MAX_HOSTS) };
            cxl_write_cl_sync(r.number[i], &Cacheline::from_b0(0));
            cxl_write_cl_sync(r.entering[i], &Cacheline::from_b0(0));
        }
        r
    }

    pub fn from_cl_sync(base_cl: *mut Cacheline) -> Option<BakeryLock> {
        cxl_typed_from_cl_sync(base_cl, |_meta| {
            let lock_start_cl = unsafe { base_cl.add(1) };
            let mut r = BakeryLock {
                myid: get_machine_id() - 1,
                entering: [0 as *mut Cacheline; MAX_HOSTS],
                number: [0 as *mut Cacheline; MAX_HOSTS],
                hosts: MAX_HOSTS,
            };
            for i in 0..MAX_HOSTS {
                r.entering[i] = unsafe { lock_start_cl.add(i) };
                r.number[i] = unsafe { lock_start_cl.add(i + MAX_HOSTS) };
            }
            r
        })
    }

    /// Blocking call to get the lock.
    pub fn lock(&self) {
        cxl_write_cl_sync(self.entering[self.myid as usize], &Cacheline::from_b0(1));
        let mut m = 0u64;
        for i in 0..MAX_HOSTS {
            let n = cxl_read_cacheline_sync(self.number[i]).d[0];
            if n > m {
                m = n;
            }
        }
        m = m + 1;
        cxl_write_cl_sync(
            self.number[self.myid as usize],
            &Cacheline::from_b0((m).into()),
        );
        //#[cfg(target_arch = "x86_64")]
        //std::arch::asm!("sfence", options(nostack, preserves_flags));
        //#[cfg(target_arch = "x86_64")]
        //std::arch::asm!("mfence", options(nostack, preserves_flags));
        //#[cfg(target_arch = "x86_64")]
        //std::arch::asm!("lfence", options(nostack, preserves_flags));
        //if m!=cxl_read_cacheline(self.number[self.myid as usize]).d[0] { //validate number
        //    println!("My number: {m}");
        //}
        cxl_write_cl_sync(self.entering[self.myid as usize], &Cacheline::from_b0(0));
        for i in 0..MAX_HOSTS {
            while cxl_read_cacheline_sync(self.entering[i]).d[0] != 0 {}
            let mut num = cxl_read_cacheline_sync(self.number[i]).d[0];
            while (num != 0) && ((num < m) || ((num == m) && (i < self.myid.into()))) {
                num = cxl_read_cacheline_sync(self.number[i]).d[0];
            }
        }
    }

    /// Unlock the lock.
    pub fn unlock(&self) {
        cxl_write_cl_sync(self.number[self.myid as usize], &Cacheline::from_b0(0));
    }
}

impl TreesLock for BakeryLock {
    fn lock(&self) {
        self.lock();
    }
    fn unlock(&self) {
        self.unlock();
    }
}

impl CXLSharedState for BakeryLock {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(n: usize) -> usize {
        // Header + (1 Turn + 1 Flag) per Host
        1 + (2 * MAX_HOSTS)
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        1 + (2 * MAX_HOSTS)
    }
    fn get_cxl_type_id() -> u64 {
        0x666
    }
}

//##########################
//  Tests
//#########################

#[cfg(test)]
mod test_nhost {
    use super::*;
    use crate::core::{Cacheline, cl_set_sync, cxl_read_cacheline_sync};
    use crate::{core::host, cxl_write_sync, utils::open_test_cxl_dev};
    use llfree::TreesLock;
    use std::thread;

    #[test]
    fn tournament_peterson() {
        let (mem, _) = open_test_cxl_dev();

        let barrier = mem as usize;
        cxl_write_sync!((barrier as *mut Cacheline, 0));

        let cl_locks_header = unsafe { mem.add(1) };
        let cl_locks_start = unsafe { mem.add(2) };
        // TODO only correct for hosts == 4
        for i in 0..10 {
            cl_set_sync(unsafe { cl_locks_start.add(3 * i) }, 0x10101);
            cl_set_sync(unsafe { cl_locks_start.add(3 * i + 1) }, 0x10101);
            cl_set_sync(unsafe { cl_locks_start.add(3 * i + 2) }, 0x10101);
        }

        let lock_slots = 8;
        let tpl = TournamentPetersonLock::init(cl_locks_header, lock_slots);
        assert!(cxl_read_cacheline_sync(cl_locks_header).d[0] == lock_slots as u64);
        println!("<<+-+>>");

        let mut threads = vec![];
        let machine_ids: Vec<usize> = (1..lock_slots + 1).collect();
        for i in machine_ids {
            threads.push(thread::spawn(move || {
                host::set_machine_id(i as u16);

                tpl.lock();

                let v = cxl_read_cacheline_sync(barrier as *mut Cacheline).d[0];
                cxl_write_sync!((barrier as *mut Cacheline, v + 1));

                tpl.print_state();

                tpl.unlock();

                tpl.lock();

                let v = cxl_read_cacheline_sync(barrier as *mut Cacheline).d[0];
                cxl_write_sync!((barrier as *mut Cacheline, v + 1));

                tpl.print_state();

                tpl.unlock();

                println!("<<+{i}+>>");
            }));
        }

        for t in threads {
            t.join().unwrap();
        }

        tpl.print_state();

        assert!(cxl_read_cacheline_sync(barrier as *mut Cacheline).d[0] == (lock_slots * 2) as u64);
    }
}
