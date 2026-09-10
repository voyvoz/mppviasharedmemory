use clap::Parser;
use dimes2026::utils::{
    Cli, JoinKey, RowID, TableRow, get_build_table_simple, get_probe_table_simple,
};
use shm_cxl::allocator::CXLBitMapAlloc;
use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::com::barrier::barrier_wait_all;
use shm_cxl::core::{Cacheline, store_fence};
use shm_cxl::core::hash_map::CXLHashMap;
use shm_cxl::core::host::{
    PeerMachineID, get_machine_cnt, get_machine_id, set_machine_cnt, set_machine_id,
};
use shm_cxl::cxl::PersistantDataStructs;
use shm_cxl::utils::{FnvBuildHasher, FnvHashMap, init_n_host_ids, open_test_cxl_dev};
use std::alloc::Layout;
use std::fs::File;
use std::io::{BufReader, prelude::*};
use std::process::exit;
use std::time::{self, Duration};

fn run_baseline(
    htsize: usize,
    bsize: usize,
    build_table: &Vec<TableRow>,
    probe_table: &Vec<TableRow>,
) -> Vec<(RowID, RowID)> {
    let mut hm: FnvHashMap<JoinKey, Vec<(JoinKey, RowID)>> =
        FnvHashMap::with_capacity_and_hasher(htsize, FnvBuildHasher::new());

    let start = time::SystemTime::now();
    for i in build_table {
        hm.entry(i.join_key)
            .or_insert(Vec::with_capacity(bsize))
            .push((i.join_key, i.row));
    }
    let build_time = start.elapsed();
    println!("build_time: {build_time:?} for {} elems", build_table.len());

    let mut result = vec![];
    let probe_start = time::SystemTime::now();
    for i in probe_table {
        if let Some(get) = hm.get(&i.join_key) {
            for j in get {
                result.push((i.row, { *j }.1));
            }
        }
    }
    let probe_time = probe_start.elapsed();
    println!("probe_time: {probe_time:?} for {} elems", probe_table.len());
    result
}

fn run_cxl<A>(
    build_table: &Vec<TableRow>,
    probe_table: &Vec<TableRow>,
    hm: &mut CXLHashMap<A>,
) -> Vec<(RowID, RowID)>
where
    A: Allocator,
{
    let nhosts = get_machine_cnt();
    let myid = get_machine_id();

    println!("Starting build phase... elems:{}", build_table.len());
    let build_slice_size = build_table.len().div_ceil(nhosts as usize);
    let build_slice_start = build_slice_size * ((myid - 1) as usize);
    let mut build_slice_end = build_slice_start + build_slice_size;
    if build_slice_end > build_table.len() {
        build_slice_end = build_table.len();
    }
    let start = time::SystemTime::now();
    for i in build_slice_start..build_slice_end {
        let e: &TableRow = &build_table[i];
        hm.insert_sync(e.join_key.0, e.row.0);
    }
    let build_time = start.elapsed();
    println!("[cxl_build_time] local build: {build_time:?}");
    barrier_wait_all();
    hm.prep_for_probe();
    println!("[cxl_build_time] probe prep: {:?}", start.elapsed());

    println!("Starting probe phase... elems:{}", probe_table.len());
    let mut result = vec![];
    let probe_slice_size = probe_table.len().div_ceil(nhosts as usize);
    let probe_slice_start = probe_slice_size * ((myid - 1) as usize);
    let mut probe_slice_end = probe_slice_start + probe_slice_size;
    if probe_slice_end > probe_table.len() {
        probe_slice_end = probe_table.len();
    }
    let probe_start = time::SystemTime::now();
    for i in probe_slice_start..probe_slice_end {
        let e = &probe_table[i];
        let res = hm.get_values(e.join_key.0);
        for j in res {
            result.push((e.row, RowID::from_value(j.1)));
        }
    }
    let probe_time = probe_start.elapsed();
    println!("[cxl_probe_time] local probe: {probe_time:?}");
    barrier_wait_all();
    println!("[cxl_probe_time] barrier: {:?}", probe_start.elapsed());

    println!(
        "{myid} produced {} elems [ {build_slice_start} -> {build_slice_end} ]",
        result.len()
    );

    result
}

fn loadfile(fname: String, mut start_row_id: u64, max_rows: usize) -> Vec<TableRow> {
    println!("Loading {fname}");
    let file = File::open(fname).expect("filename has to be a valid file");
    let reader = BufReader::new(file);

    let mut v: Vec<TableRow> = Vec::new();
    let mut i = 0;
    for line in reader.lines() {
        v.push(TableRow {
            join_key: JoinKey(
                line.expect("kein Ahnung")
                    .trim()
                    .parse::<u64>()
                    .expect("Should be an integer"),
            ),
            row: RowID(start_row_id),
        });
        start_row_id += 1;
        i += 1;
        if i >= max_rows && max_rows != 0 {
            break;
        }
    }
    println!("Loaded {i} lines");
    v
}

fn main() {
    let args = Cli::parse();
    println!("Args: {args:?}");

    let build_table = if args.buildfile != "none" {
        loadfile(args.buildfile, 1000, 1000)
    } else {
        get_build_table_simple()
    };

    let probe_table = if args.probefile != "none" {
        loadfile(args.probefile, 10000000, 0)
    } else {
        get_probe_table_simple()
    };

    match args.num_hosts {
        0 => {
            let res = run_baseline(
                args.ht_buckets,
                args.ht_bucket_capacity,
                &build_table,
                &probe_table,
            );
            for i in res {
                println!("{i:?}");
            }
        }
        _ => {
            let (dev, size) = open_test_cxl_dev();
            if args.num_hosts > 1 {
                init_n_host_ids(dev, args.num_hosts as u16, Some(5000));
            } else {
                // set dummy
                set_machine_id(1);
                set_machine_cnt(1);
            };
            let myid = get_machine_id();
            if myid > 1 {
                //delay others to sync inits but not master
                std::thread::sleep(std::time::Duration::from_millis(50));
            }

            let mut hm: CXLHashMap<CXLBitMapAlloc>;
            let mut persist: PersistantDataStructs;
            if myid == 1 {
                persist = PersistantDataStructs::init(dev, size);
                hm = CXLHashMap::try_new(
                    args.ht_buckets,
                    args.ht_bucket_capacity,
                    args.num_hosts,
                    persist.alloc.clone(),
                )
                .unwrap();

                if args.num_hosts > 1 {
                    // Now tell the other hosts where they should expect stuff to be
                    for i in 2..=args.num_hosts {
                        // Send build side table location
                        //persist.ipc.try_send_to(i as PeerMachineID, &Cacheline::from_b0(probe_side_table_as_offset)).unwrap();
                        // Send probe side table location
                        //persist.ipc.try_send_to(i as PeerMachineID, payload);
                        // Send HastTable location
                        persist
                            .ipc
                            .try_send_to(
                                i as PeerMachineID,
                                &Cacheline::from_b0(Cacheline::ptr_to_offset(hm.get_base())),
                            )
                            .unwrap();
                    }
                }
            } else {
                persist = PersistantDataStructs::from_cl_sync(dev);
                // Get shared HashTable
                let (_, cl_hm) = persist
                    .ipc
                    .try_pop_from(1 as PeerMachineID, Some(Duration::from_millis(100)))
                    .unwrap();
                hm = CXLHashMap::from_cl_sync(
                    Cacheline::ptr_from_offset(cl_hm.d[0]),
                    persist.alloc.clone(),
                )
                .unwrap();
            }
            println!("-- init done --");

            barrier_wait_all();

            let local_res = run_cxl(&build_table, &probe_table, &mut hm);

            if myid != 1 {
                println!("-- Sending results ->>");
                // Allocate some memory
                let mem = persist
                    .alloc
                    .allocate(Layout::from_size_align(local_res.len() * 16, 64).unwrap())
                    .unwrap();
                // Write our results to that memory
                for i in 0..local_res.len() {
                    let elem = local_res[i];
                    let addr = unsafe { (mem.as_ptr() as *mut u64).add(i * 2) };
                    unsafe {
                        *addr = elem.0.0;
                        *(addr.add(1)) = elem.1.0;
                        #[cfg(target_arch = "x86_64")]
                        std::arch::asm!(
                            "clflushopt [{addr}]",
                            addr = in(reg) addr
                        );
                    }
                }
                store_fence();
                // Send message to main host to tell them about our result
                // Set ptr, capa, size
                let offset = (mem.as_ptr() as *mut Cacheline as u64 - dev as u64) / 64;
                let payload = Cacheline {
                    d: [
                        offset,
                        local_res.len() as u64,
                        mem.len() as u64,
                        3,
                        4,
                        5,
                        6,
                        7,
                    ],
                };
                persist
                    .ipc
                    .try_send_to(1 as PeerMachineID, &payload)
                    .unwrap();
            } else {
                println!("<<- Collecting results ->>");

                // assemble result from CXL Vecs
                let mut external_res: Vec<Vec<(RowID, RowID)>> = Vec::new();
                for i in 2..=args.num_hosts {
                    let msg = persist.ipc.try_pop_from(i as PeerMachineID, None).unwrap();
                    let cl_from_offset: *mut Cacheline = unsafe { dev.add(msg.1.d[0] as usize) };
                    let v: Vec<(RowID, RowID)> = unsafe {
                        Vec::from_raw_parts(
                            cl_from_offset as *mut (RowID, RowID),
                            msg.1.d[1] as usize,
                            msg.1.d[2] as usize,
                        )
                    };
                    let l = v.len();
                    external_res.push(v);
                    println!("{i} produced {l} elems");
                }

                let mut overall_res_size = local_res.len();
                external_res
                    .iter()
                    .for_each(|v| overall_res_size += v.len());

                // Validity check
                println!("-- Running local comparison --");
                let correct_res = run_baseline(
                    args.ht_buckets,
                    args.ht_bucket_capacity,
                    &build_table,
                    &probe_table,
                );
                println!("<<- Result ->>");
                println!(
                    "overall_res_size:{overall_res_size} should be correct_res_len:{}",
                    correct_res.len()
                );
                let mut hm2: FnvHashMap<u64, u64> = FnvHashMap::default();
                for i in correct_res {
                    hm2.insert(i.0.0, i.1.0);
                }
                for i in local_res {
                    hm2.remove(&i.0.0);
                }
                for i in 0..external_res.len() {
                    for i in &external_res[i] {
                        hm2.remove(&i.0.0);
                    }
                }
                if hm2.keys().len() == 0 {
                    exit(0);
                } else {
                    println!("! The following results are missing:");
                    for i in hm2.iter() {
                        println!("{i:?}");
                    }
                    exit(1);
                }
            }
        }
    };
}
