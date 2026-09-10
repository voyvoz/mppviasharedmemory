use hashbrown::HashMap;
use shm_cxl::l1_api::memory_management::MemoryManager;
use std::env;
use std::panic;

// Host A
// shm Init
// Create a HashTable using HasBrown?
// Move it to CXL memor<
// Ping other process
// exit
fn proc_a() {
    // Setup
    let res = match MemoryManager::mm_init(vec![]) {
        Err(err) => panic!("{:?}", err),
        Ok(e) => e,
    };

    // Create a hash map using operating system memory
    let mut hash_map = HashMap::new_in(&*res.os_alloc);
    hash_map.insert("some_key", "some_value");

    //res.mm_move(source, destination_type)
    //res.mm_set_owner(obj, owner);
    //ipc_send(target, payload);
}

// Host B
// shm init
// read a table
// wait until host a pings us
// join tables using the hashmap
// exit
fn proc_b() {
    // Setup
    let res = match MemoryManager::mm_init(vec![]) {
        Err(err) => panic!("{:?}", err),
        Ok(e) => e,
    };
    // read input table
    // await IPC
    // do probing
    // put result into CXL
    // send IPC to Host A
}

// Host C
// shm init
// read results from B
// put back into CXL and signal Host
fn proc_c() {
    // Setup
    let res = match MemoryManager::mm_init(vec![]) {
        Err(err) => panic!("{:?}", err),
        Ok(e) => e,
    };

    // await IPC
    // use CXL memory as source and decompress strings
    // put back to CXL
    // send IPC to Host A
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() == 1 {
        let param = args[1].as_str();
        match param {
            "a" => proc_a(),
            "b" => proc_b(),
            "c" => proc_c(),
            _ => panic!("Invalid param"),
        }
    }
    panic!("Requires either 'a', 'b' or 'c'.");

    // DataFusion
    // TPC-H Q3 9 10 17 ?
}
