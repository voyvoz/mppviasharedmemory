use crate::core::host::{
    MAX_HOSTS, get_hostname, get_machine_cnt, get_machine_id, set_machine_cnt, set_machine_id,
};
use crate::core::{Cacheline, cxl_read_cacheline_sync, cxl_write_cl_sync};
use crate::{ShmError, cxl_write_sync};
use libc::{major, minor};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::CString;
use std::fs::File;
use std::fs::OpenOptions;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::io;
use std::io::BufReader;
use std::io::prelude::*;
use std::os::fd::IntoRawFd;
use std::path::Path;
use std::process;
use std::time;

pub const DEFAULT_DEVDAX_PATH: &str = "/dev/dax0.0";
pub const DEFAULT_DEVDAX_MEM_SIZE: usize = 256 << 30;

pub fn get_daxdev_size(fname: &str) -> Result<usize, std::io::Error> {
    let target_path = CString::new(fname).unwrap();
    let mut stat_res: libc::stat = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::stat(target_path.as_ptr(), &mut stat_res) };
    if r < 0 {
        panic!(
            "Stat on {} failed with {}",
            fname,
            std::io::Error::last_os_error()
        );
    }
    let major = major(stat_res.st_rdev);
    let minor = minor(stat_res.st_rdev);
    let sp = format!("/sys/dev/char/{}:{}/size", major, minor);
    println!("looking at {sp}");

    let file = File::open(sp)?;

    let mut size_str = String::new();
    let r = BufReader::new(file).read_line(&mut size_str);
    if r.is_err() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Unable to read of daxdev",
        ));
    }
    //Ok(size)
    let size = size_str.trim().parse::<usize>();
    match size {
        Err(_) => Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Unable to parse size of daxdev",
        )),
        Ok(size) => Ok(size),
    }
}

pub fn get_host_signature() -> [u64; 8] {
    let mut sig = [0u64; 8];

    let hn = get_hostname();
    let mut s = DefaultHasher::new();
    hn.hash(&mut s);
    sig[0] = s.finish();
    ////let machine_id_str = mid::get("mySecretKey").unwrap();
    //println!("My machine_id_str: {machine_id_str}");
    //assert!(machine_id_str.chars().count()<=8*16);
    //for i in 0..machine_id_str.chars().count()/16 {
    //   sig[i]=u64::from_str_radix(&machine_id_str[i*16..i*16+16], 16).unwrap();
    //}
    if sig[0] == 0 {
        sig[0] = (sig[0] ^ sig[3]) | 1; // "make random but not 0"
    }
    //for i in 0..sig.len() {
    //    println!("sig[{i}] = {}", sig[i]);
    //}

    sig
}

pub fn get_process_signature() -> [u64; 8] {
    let mut sig = get_host_signature();
    sig.rotate_right(1);
    //sig[1..8]=sig[0..7];
    sig[0] = process::id() as u64;
    sig
}

pub fn append_alivetimestamp(mut sig: [u64; 8]) -> [u64; 8] {
    sig.rotate_right(1);
    sig[0] = alive_timestamp();
    sig
}

pub fn alive_timestamp() -> u64 {
    (time::SystemTime::now()
        .duration_since(time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis()
        / 100) as u64
}

pub fn parse_size(sizestring: &String) -> usize {
    let mut scale = 1usize;
    let mut trimmed = sizestring.trim();
    if sizestring.len() > 0 {
        scale = match trimmed.to_lowercase().chars().last().unwrap() {
            'k' => 1 << 10,
            'm' => 1 << 20,
            'g' => 1 << 30,
            't' => 1 << 40,
            '0'..='9' => 1,
            _ => panic!("Supported sizes are k,K,m,M,g,G,t,T"),
        };
        if scale != 1 {
            trimmed = &trimmed[0..trimmed.len() - 1]; //remove last
        }
    }
    let number;
    if trimmed.starts_with("0x") {
        trimmed = &trimmed[2..];
        number = usize::from_str_radix(trimmed, 16).expect("Has to be 0x[<0-9a-f]");
    } else {
        number = trimmed.parse::<usize>().expect("Sizes have to be numbers");
    }
    return number * scale;
}

/// Decides on a random offset within all available CXL shared memory.
/// NOTE we guarantee that there is at least 2GB of space left.
pub fn open_test_cxl_dev() -> (*mut Cacheline, usize) {
    let _ = env_logger::try_init();

    // open
    let fd = open_cxl_daxdev(DEFAULT_DEVDAX_PATH).unwrap();

    let mapsize = parse_size(
        &env::var("CXL_MAPSIZE").unwrap_or(
            get_daxdev_size(DEFAULT_DEVDAX_PATH)
                .unwrap_or(32 << 30)
                .to_string(),
        ),
    );
    let mapoffset = parse_size(&env::var("CXL_MAPOFFSET").unwrap_or("0x0".to_string()));

    // mmap
    let region = match mmap_cxl_dev(fd, mapsize, mapoffset) {
        Ok(v) => v,
        Err(e) => panic!("mmap_cxl_dev failed with {e}"),
    };

    region
}

pub fn open_cxl_daxdev(path: impl AsRef<Path>) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(path.as_ref())
}

///Open the given path and mmap it as MAP_SHARED.
/// Size must be multiple of 2MB in most circumstances.
pub fn mmap_cxl_dev(
    fd: File,
    len: usize,
    offset: usize,
) -> Result<(*mut Cacheline, usize), Box<dyn std::error::Error>> {
    println!("mmap_cxl_dev(fd={fd:?},len={len},offset={offset})");
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd.into_raw_fd(),
            offset as i64,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(ShmError::GenericError(format!(
            "libc::mmap: {}",
            std::io::Error::last_os_error().to_string()
        ))
        .into());
    }
    println!("mapped offset:{offset}+len:{len} into {ptr:?}");

    Ok((ptr as *mut Cacheline, len))
}

pub fn now_nsec() -> u128 {
    time::SystemTime::now()
        .duration_since(time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

pub fn now_sec() -> u64 {
    time::SystemTime::now()
        .duration_since(time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

///Uses CXL shared memory to coordinate IDs with another host using the same function
pub fn init_n_host_ids(base_addr: *mut Cacheline, wait_num: u16, timeout_ms: Option<u16>) -> u16 {
    const DEFAULT_SINGLE_HOST_TIMEOUT_MS: u16 = 1_000;
    const DEFAULT_MULTI_HOST_TIMEOUT_MS: u16 = 30_000;
    const DISCOVERY_ACK: u64 = 0xACCE_5510_ACCE_5510;
    const DISCOVERY_RELEASE: u64 = 0x0000_000A_CE0F_BA5E;
    const DISCOVERY_RELEASE_ACK: u64 = 0xACCE_5511_ACCE_5511;
    //let machine_id_str = mid::get("mySecretKey").unwrap();
    //println!("My machine_id_str: {machine_id_str}");
    //let mut myid = Cacheline::empty();
    //assert!(machine_id_str.chars().count()<8*16);
    //for i in 0..machine_id_str.chars().count()/16 {
    //   myid.d[i]=u64::from_str_radix(&machine_id_str[i*16..i*16+16], 16).unwrap();
    //}
    //if myid.d[0]==0
    //{
    //    myid.d[0]=(myid.d[7] ^ myid.d[6])|1; // "random but not 0"
    //}
    let mut myid = Cacheline {
        d: get_host_signature(),
    };
    //Cacheline::print_hex(&mut myid);
    let d = myid.d[0];
    //println!("id[0]: {d}");

    // We need three cachelines
    let mut field = [0usize as *mut Cacheline; MAX_HOSTS];
    for i in 0..field.len() {
        field[i] = unsafe { base_addr.offset(i as isize) };
        cxl_write_cl_sync(field[i], &Cacheline::from_b0(0));
    }

    let start_syncdata = unsafe { base_addr.offset(field.len() as isize) };
    for n in 0..MAX_HOSTS {
        cxl_write_cl_sync(
            unsafe { start_syncdata.offset(n as isize) },
            &Cacheline::empty(),
        );
    }

    let time_to_stabilize_ns = (timeout_ms.unwrap_or(match wait_num {
        0 => DEFAULT_SINGLE_HOST_TIMEOUT_MS,
        _ => DEFAULT_MULTI_HOST_TIMEOUT_MS,
    }) as u128)
        * 1000
        * 1000;

    //println!("ns to stabilize {time_to_stabilize_ns}");

    let mut sync_end = now_nsec() + time_to_stabilize_ns;

    let mut myhostid = 0u16;
    let mut hostcnt: u32;

    println!("Searching for other CXL nodes... (Expecting {} hosts)", wait_num);

    loop {
        cxl_write_cl_sync(field[(myhostid as usize) % field.len()], &myid);
        // "waste" some time so the likelyhood of others overwriting is greater
        // doing something usefull here as we need that hostcnt anyway
        hostcnt = 0u32;
        for f in field {
            if cxl_read_cacheline_sync(f).d[0] > 0 {
                hostcnt += 1;
            }
        }
        let t = cxl_read_cacheline_sync(field[(myhostid as usize) % field.len()]);
        if t.d[0] == 0 {
            //someone overwrote what i had written
            println!("New node detected");
            sync_end = now_nsec() + time_to_stabilize_ns;
            continue;
        }

        if t != myid {
            sync_end = now_nsec() + time_to_stabilize_ns;
            if t.d[0] < myid.d[0]
            // sort based on first byte
            {
                //println!("Evicted");
                myhostid = myhostid.wrapping_add(1);
                assert!((myhostid as usize) < field.len())
            } else {
                //println!("Prevailed"); // it can happen that both with same d[0] stay const. better would be to sort all d0 to d7 (TODO)
            }
        } else if wait_num > 0 && hostcnt == wait_num as u32 {
            break; // we got all we are waiting for
        } else if now_nsec() > sync_end {
            if wait_num > 0 && wait_num as u32 != hostcnt {
                sync_end = now_nsec() + time_to_stabilize_ns; //keep waiting for other hosts
            } else {
                break; // we are done here
            }
        }
    }

    println!(">> Done. {hostcnt} nodes found");

    //thread::sleep(time::Duration::from_millis(5 * (myhostid as u64)));

    //sanity check
    cxl_write_sync!((start_syncdata.offset(myhostid as isize), 0, hostcnt));

    assert!(
        hostcnt > 1,
        "it doesn't make sense to sync with only one host"
    );

    sync_end = now_nsec() + time_to_stabilize_ns;
    loop {
        let mut cnt = 0;
        for i in 0..hostcnt {
            if cxl_read_cacheline_sync(unsafe { start_syncdata.offset(i as isize) }).d[0]
                == (hostcnt as u64)
            {
                cnt += 1;
            }
        }
        if cnt == hostcnt {
            break;
        } else if now_nsec() > sync_end {
            panic!("unable to agree on host count during CXL discovery");
        }
        std::hint::spin_loop();
    }

    // Do not let host 0 replace syncdata[0] with the release word until every
    // peer has observed the complete syncdata array. Without this second
    // acknowledgement, a fast host 0 can publish the release word while a
    // slower peer is still waiting for every slot to equal `hostcnt`; that peer
    // then spins forever because slot 0 can never match again.
    cxl_write_sync!((field[myhostid as usize], 0, DISCOVERY_ACK));
    sync_end = now_nsec() + time_to_stabilize_ns;
    loop {
        let mut cnt = 0;
        for i in 0..hostcnt as usize {
            // A peer that already advanced past the release phase has
            // replaced its ACK with DISCOVERY_RELEASE_ACK -- a strictly
            // later protocol state, so it proves the ACK happened. Without
            // accepting it, the last host to ACK can spin here forever
            // once its peers move on (the ~1%-of-runs 3-host discovery
            // timeout).
            let word = cxl_read_cacheline_sync(field[i]).d[0];
            if word == DISCOVERY_ACK || word == DISCOVERY_RELEASE_ACK {
                cnt += 1;
            }
        }
        if cnt == hostcnt {
            break;
        }
        if now_nsec() > sync_end {
            panic!("timed out waiting for CXL discovery acknowledgements");
        }
        std::hint::spin_loop();
    }

    if myhostid == 0 {
        cxl_write_sync!((start_syncdata, 0, DISCOVERY_RELEASE));
    } else {
        sync_end = now_nsec() + time_to_stabilize_ns;
        while cxl_read_cacheline_sync(start_syncdata).d[0] != DISCOVERY_RELEASE {
            if now_nsec() > sync_end {
                panic!("timed out waiting for CXL discovery release");
            }
            std::hint::spin_loop();
        }
    }

    // The discovery words share the beginning of the mapped region with the
    // persistent CXL header.  Do not let host 0 reuse that storage until every
    // peer has actually observed the release word; otherwise a fast host 0 can
    // erase it during initialization and leave a slower peer spinning forever.
    // Only host 0 needs that guarantee, so only host 0 collects the acks: a
    // peer polling here can miss slots once host 0 starts reusing the region.
    cxl_write_sync!((field[myhostid as usize], 0, DISCOVERY_RELEASE_ACK));
    if myhostid == 0 {
        sync_end = now_nsec() + time_to_stabilize_ns;
        loop {
            let mut cnt = 0;
            for i in 0..hostcnt as usize {
                if cxl_read_cacheline_sync(field[i]).d[0] == DISCOVERY_RELEASE_ACK {
                    cnt += 1;
                }
            }
            if cnt == hostcnt {
                break;
            }
            if now_nsec() > sync_end {
                panic!("timed out waiting for CXL discovery release acknowledgements");
            }
            std::hint::spin_loop();
        }
    }

    set_machine_cnt(hostcnt as u16);
    set_machine_id(myhostid + 1);
    println!(
        ">> Sync complete. I'm {} of {} nodes",
        get_machine_id(),
        get_machine_cnt()
    );
    myhostid + 1
}

pub fn align_up<T>(ptr: *const T, align: usize) -> *const T {
    let align = align_of::<T>().max(align);
    (ptr as usize).next_multiple_of(align) as *const T
}
pub fn align_down<T>(ptr: *const T, align: usize) -> *const T {
    let align = align_of::<T>().max(align);
    ((ptr as usize / align) * align) as *const T
}
pub fn align_up_mut<T>(ptr: *mut T, align: usize) -> *mut T {
    let align = align_of::<T>().max(align);
    (ptr as usize).next_multiple_of(align) as *mut T
}
pub fn align_down_mut<T>(ptr: *mut T, align: usize) -> *mut T {
    let align = align_of::<T>().max(align);
    ((ptr as usize / align) * align) as *mut T
}

// Copied from: https://github.com/servo/rust-fnv/blob/main/lib.rs
const INITIAL_STATE: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0100_0000_01b3;
#[derive(Clone)]
pub struct FnvHasher(u64);
impl Default for FnvHasher {
    #[inline]
    fn default() -> FnvHasher {
        FnvHasher(INITIAL_STATE)
    }
}
impl FnvHasher {
    /// Create an FNV hasher starting with a state corresponding
    /// to the hash `key`.
    #[inline]
    #[must_use]
    pub fn with_key(key: u64) -> FnvHasher {
        FnvHasher(key)
    }
}
impl Hasher for FnvHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let FnvHasher(mut hash) = *self;

        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(PRIME);
        }

        *self = FnvHasher(hash);
    }
}
/// A builder for default FNV hashers.
pub type FnvBuildHasher = BuildHasherDefault<FnvHasher>;

/// A `HashMap` using a default FNV hasher.
pub type FnvHashMap<K, V> = HashMap<K, V, FnvBuildHasher>;

/// A `HashSet` using a default FNV hasher.
pub type FnvHashSet<T> = HashSet<T, FnvBuildHasher>;

#[cfg(test)]
mod tests {
    use super::parse_size;

    #[test]
    fn parse_size_uses_binary_unit_multipliers() {
        assert_eq!(parse_size(&"1K".to_string()), 1 << 10);
        assert_eq!(parse_size(&"1M".to_string()), 1 << 20);
        assert_eq!(parse_size(&"1G".to_string()), 1 << 30);
        assert_eq!(parse_size(&"1T".to_string()), 1 << 40);
        assert_eq!(parse_size(&"88G".to_string()), 88usize << 30);
    }
}
