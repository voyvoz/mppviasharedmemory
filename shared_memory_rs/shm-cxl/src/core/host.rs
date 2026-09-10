use gethostname::gethostname;
use std::cell::OnceCell;
use std::sync::OnceLock;

thread_local! {
    /// A cell that contains the computed ID
    static CELL_ID: OnceCell<u16> =  OnceCell::new();
    /// A cell that contains the number of connected nodes
    static CELL_CNT: OnceCell<u16> =  OnceCell::new();
}

/// A cell that contains the hostname
static CELL_HOSTNAME: OnceLock<String> = OnceLock::new();

pub fn get_hostname() -> String {
    let hn = CELL_HOSTNAME.get_or_init(|| {
        let hn = gethostname();
        hn.to_str().unwrap().trim().to_owned()
    });

    hn.to_string() //.as_ref()
}

// overwrite hostname
pub fn set_hostname_override(hn: String) {
    CELL_HOSTNAME
        .set(hn)
        .expect("Hostname has to be not set already when doing override.");
}

pub fn get_machine_cnt() -> u16 {
    let mut r = 0;
    CELL_CNT.with(|cell| {
        r = *cell.get().expect("machine_cnt has to be set before usage");
    });
    r
}

pub fn set_machine_cnt(cnt: u16) {
    CELL_CNT.with(|cell| cell.set(cnt).unwrap());
}

/// Used to retrieve the cluster node id.
///
/// Can be set via set_machine_id to any value.
pub fn get_machine_id() -> u16 {
    let mut r = 0;
    CELL_ID.with(|cell| {
        r = *cell
            .get()
            .expect("my_machine_id has to be set before usage");
    });
    r
}

/// Does nothing if machine_id has already been set
pub fn set_machine_id(my_id: u16) {
    CELL_ID.with(|cell| cell.set(my_id).unwrap());
}

pub const MAX_HOSTS: usize = 4;
pub const LEADER_ID: usize = 1;

pub type PeerMachineID = u16;

/// A PeerID is used to identify the machine and allocator that owns an allocation.
/// Once shared between allocators this is used to determine who is
/// responsible for freeing allocations.
///
/// It also contains free space to allow higher level APIs to set their
/// own identifiers within this ID srtucutre.
///
/// u128: | machine_id: u16 | alloc_id: u16 | Custom |
///
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PeerID {
    pub machine_id: PeerMachineID,
    pub alloc_id: u16,
    // Higher level APIs can use this however they want
    pub other: [u32; 3],
}

/// A 0 machine id is considered invalid
pub const MACHINE_ID_INVALID: PeerMachineID = 0;
/// A 0 alloc id means that we just dont care which one
pub const ALLOC_ID_UNSET: u16 = 0;

impl PeerID {
    /// Returns the PeerID of the calling process assuming no specific allocator
    pub fn whoami() -> Self {
        PeerID {
            machine_id: get_machine_id() as u16,
            alloc_id: ALLOC_ID_UNSET,
            other: [0, 0, 0], // TODO
        }
    }

    pub fn to_bytes(&self) -> [u64; 2] {
        let p1: u64 =
            (self.machine_id as u64) << 48 | (self.alloc_id as u64) << 32 | self.other[0] as u64;
        let p2: u64 = (self.other[1] as u64) << 32 | self.other[2] as u64;
        [p1, p2]
    }

    pub fn from_bytes(b: [u64; 2]) -> Self {
        PeerID {
            machine_id: (b[0] >> 48) as u16,
            alloc_id: (b[0] >> 32) as u16,
            other: [0, 0, 0], // TODO
        }
    }

    pub fn from_machine_id(machine_id: u16) -> Self {
        PeerID {
            machine_id,
            alloc_id: 0,
            other: [0, 0, 0],
        }
    }

    // Requires that machine_id and alloc_id match
    pub fn eq(&self, other: &PeerID) -> bool {
        self.machine_id == other.machine_id && self.alloc_id == other.alloc_id
    }
    pub fn neq(&self, other: &PeerID) -> bool {
        !self.eq(other)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn base() {
        //TODO
        // Two threads set their ids and check that they dont match!
    }
}
