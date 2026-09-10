use clap::Parser;
use log::{debug, info};
use shm_cxl::com::ipc::{CXLIpc, CXLIpcOptions};
use shm_cxl::core::host::{get_machine_id, set_machine_id};
use shm_cxl::core::set_global_offset;
use shm_cxl::core::{CXLRBParams, Cacheline};
use shm_cxl::cxl_v2::CXLSizedObject;
use shm_cxl::utils::timestamps::{
    get_bench_metrics, record_wait_spins, reset_bench_metrics, set_bench_metrics_enabled,
};
use std::ffi::CString;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::io::FromRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;
use std::{mem, thread};

#[derive(Clone, Debug, Parser)]
pub struct Cli {
    /// How many CXL Hosts to expect
    #[arg(short, long, default_value_t = 3)]
    pub num_hosts: usize,

    #[arg(short, long, default_value_t = 128)]
    pub num_rb_capacity: usize,
    #[arg(short, long, default_value_t = 1024)]
    pub num_rb_slot_cize: usize,

    #[arg(short, long, default_value_t = 1)]
    pub num_rx_threads: usize, // NO
    #[arg(short, long, default_value_t = 1)]
    pub num_tx_threads: usize,

    #[arg(short, long, default_value_t = 1)]
    pub num_num_ipc_instances: usize,

    /// MTU assigned to the CXL-backed TUN interface.
    #[arg(long, default_value_t = 65000)]
    pub mtu: i32,
}

// https://www.kernel.org/doc/html/latest/networking/tuntap.html

const IFF_TUN: i16 = 0x0001;
const IFF_NO_PI: i16 = 0x1000;
const TUNSETIFF: u64 = 0x400454ca;
const IFF_BUF: usize = 65536;
#[repr(C)]
struct IfReq {
    ifr_name: [u8; 16],
    ifr_flags: i16,
    _pad: [u8; 22],
}

fn tun_alloc(name: &str) -> std::io::Result<File> {
    let path = CString::new("/dev/net/tun").unwrap();
    let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR) };
    if fd < 0 {
        debug!("libc::open failed");
        return Err(std::io::Error::last_os_error());
    }

    let mut ifr: IfReq = unsafe { mem::zeroed() };
    let name_bytes = name.as_bytes();
    ifr.ifr_name[..name_bytes.len()].copy_from_slice(name_bytes);
    ifr.ifr_flags = IFF_TUN | IFF_NO_PI;

    let res = unsafe { libc::ioctl(fd, TUNSETIFF as _, &ifr) };
    if res < 0 {
        debug!("libc::ioctl failed");
        unsafe { libc::close(fd) };
        return Err(std::io::Error::last_os_error());
    }

    Ok(unsafe { File::from_raw_fd(fd) })
}

const SIOCGIFFLAGS: libc::c_ulong = 0x8913;
const SIOCSIFFLAGS: libc::c_ulong = 0x8914;
const SIOCSIFADDR: libc::c_ulong = 0x8916;
const SIOCSIFNETMASK: libc::c_ulong = 0x891c;
const SIOCSIFMTU: libc::c_ulong = 0x8922;

fn named_ifreq(name: &str) -> IfReq {
    let mut ifr: IfReq = unsafe { mem::zeroed() };
    let bytes = name.as_bytes();
    assert!(
        bytes.len() < ifr.ifr_name.len(),
        "interface name is too long"
    );
    ifr.ifr_name[..bytes.len()].copy_from_slice(bytes);
    ifr
}

fn ipv4_sockaddr(octets: [u8; 4]) -> libc::sockaddr {
    let mut addr: libc::sockaddr = unsafe { mem::zeroed() };
    addr.sa_family = libc::AF_INET as libc::sa_family_t;
    // sockaddr.sa_data contains the two-byte port followed by the IPv4 bytes.
    for (index, octet) in octets.into_iter().enumerate() {
        addr.sa_data[index + 2] = octet as libc::c_char;
    }
    addr
}

fn ifreq_write<T: Copy>(ifr: &mut IfReq, value: T) {
    assert!(mem::size_of::<T>() <= ifr._pad.len() + mem::size_of::<i16>());
    unsafe {
        std::ptr::write_unaligned((&mut ifr.ifr_flags as *mut i16).cast::<T>(), value);
    }
}

fn ifreq_read<T: Copy>(ifr: &IfReq) -> T {
    unsafe { std::ptr::read_unaligned((&ifr.ifr_flags as *const i16).cast::<T>()) }
}

fn net_ioctl(fd: libc::c_int, request: libc::c_ulong, ifr: &mut IfReq) -> io::Result<()> {
    if unsafe { libc::ioctl(fd, request as _, ifr as *mut IfReq) } < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn configure_tun(name: &str, machine_id: u16, mtu: i32) -> io::Result<()> {
    let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }

    let result = (|| {
        let mut ifr = named_ifreq(name);
        ifreq_write(&mut ifr, ipv4_sockaddr([10, 0, 0, machine_id as u8]));
        net_ioctl(fd, SIOCSIFADDR, &mut ifr)?;

        let mut ifr = named_ifreq(name);
        ifreq_write(&mut ifr, ipv4_sockaddr([255, 255, 255, 0]));
        net_ioctl(fd, SIOCSIFNETMASK, &mut ifr)?;

        let mut ifr = named_ifreq(name);
        ifreq_write(&mut ifr, mtu);
        net_ioctl(fd, SIOCSIFMTU, &mut ifr)?;

        let mut ifr = named_ifreq(name);
        net_ioctl(fd, SIOCGIFFLAGS, &mut ifr)?;
        let flags: i16 = ifreq_read(&ifr);
        ifreq_write(&mut ifr, flags | libc::IFF_UP as i16);
        net_ioctl(fd, SIOCSIFFLAGS, &mut ifr)
    })();
    unsafe { libc::close(fd) };
    result
}

fn set_safe_mapping_defaults() {
    // sudo commonly strips caller-defined environment variables. Never let a
    // privileged no-argument launch fall back to mapping the complete DAX
    // device. This compiled fallback must match scripts/ip_cxl.sh because the
    // restricted sudo rule intentionally accepts no command-line arguments.
    if std::env::var_os("CXL_MAPOFFSET").is_none() {
        unsafe { std::env::set_var("CXL_MAPOFFSET", "0x1800000000") };
    }
    if std::env::var_os("CXL_MAPSIZE").is_none() {
        unsafe { std::env::set_var("CXL_MAPSIZE", "1G") };
    }
}

fn drop_sudo_privileges() -> io::Result<()> {
    if unsafe { libc::geteuid() } != 0 {
        return Ok(());
    }
    let Some(uid) = std::env::var("SUDO_UID")
        .ok()
        .and_then(|value| value.parse::<libc::uid_t>().ok())
    else {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "refusing to remain root: SUDO_UID is missing",
        ));
    };
    let gid = std::env::var("SUDO_GID")
        .ok()
        .and_then(|value| value.parse::<libc::gid_t>().ok())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "refusing to remain root: SUDO_GID is missing",
            )
        })?;
    if unsafe { libc::setgid(gid) } != 0 || unsafe { libc::setuid(uid) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn handle_metrics_control(mut stream: UnixStream) -> io::Result<()> {
    let mut command = String::new();
    BufReader::new(stream.try_clone()?).read_line(&mut command)?;
    match command.trim() {
        "START" => {
            set_bench_metrics_enabled(false);
            reset_bench_metrics();
            set_bench_metrics_enabled(true);
            writeln!(stream, "OK")?;
        }
        "STOP" => {
            set_bench_metrics_enabled(false);
            let metrics = get_bench_metrics();
            writeln!(
                stream,
                "{} {} {}",
                metrics.cxl_load_ns, metrics.cxl_store_ns, metrics.wait_spins
            )?;
        }
        other => {
            writeln!(stream, "ERROR unknown command {other}")?;
        }
    }
    Ok(())
}

fn main() {
    let _ = env_logger::try_init();

    let args = Cli::parse();
    println!("Args: {args:?}");

    // Alloc TUN
    let tun = tun_alloc("cxl").unwrap();
    println!("TUN created.");

    // Init CXL
    set_safe_mapping_defaults();
    let (dev, _size) = shm_cxl::utils::open_test_cxl_dev();
    set_global_offset(dev as u64);
    shm_cxl::utils::init_n_host_ids(dev, args.num_hosts as u16, Some(30_000));
    let machine_id = get_machine_id();
    let num_hosts = args.num_hosts as u16;
    configure_tun("cxl", machine_id, args.mtu).expect("failed to configure cxl TUN interface");
    drop_sudo_privileges().expect("failed to drop sudo privileges after configuring TUN");

    let metrics_socket =
        std::env::var("IP_CXL_METRICS_SOCKET").unwrap_or_else(|_| "tun.metrics.sock".to_string());
    let _ = std::fs::remove_file(&metrics_socket);
    let metrics_listener =
        UnixListener::bind(&metrics_socket).expect("failed to bind TUN metrics control socket");
    println!("machine_id={machine_id}");
    println!("TUN ready: 10.0.0.{machine_id}/24 mtu={}", args.mtu);

    // Initialization is outside every benchmark kernel. The local MPI rank
    // synchronously enables and reads these counters through metrics_listener.
    set_bench_metrics_enabled(false);
    reset_bench_metrics();
    let control = std::thread::spawn(move || {
        for connection in metrics_listener.incoming() {
            match connection {
                Ok(stream) => {
                    if let Err(error) = handle_metrics_control(stream) {
                        eprintln!("TUN metrics control error: {error}");
                    }
                }
                Err(error) => eprintln!("TUN metrics accept error: {error}"),
            }
        }
    });

    // A human-readable live snapshot is useful for diagnostics. Benchmark CSV
    // values come from the synchronous STOP response above, not this sampler.
    let metrics = std::thread::spawn(|| {
        loop {
            let current = get_bench_metrics();
            let row = format!(
                "{} {} {}\n",
                current.cxl_load_ns, current.cxl_store_ns, current.wait_spins
            );
            if std::fs::write("tun.metrics.tmp", row).is_ok() {
                let _ = std::fs::rename("tun.metrics.tmp", "tun.metrics");
            }
            thread::sleep(Duration::from_millis(50));
        }
    });

    let rb_config = CXLRBParams::default()
        .with_slot_size_cl(args.num_rb_slot_cize)
        .with_capacity(args.num_rb_capacity);
    let ipc_opts = CXLIpcOptions::default()
        .with_hosts(args.num_hosts)
        .with_ringbuffer_options(rb_config);

    // IPC
    let mut threads: Vec<JoinHandle<_>> = Vec::new();
    for instance_id in 0..args.num_num_ipc_instances {
        // Create IPC instance
        let base_cl_with_offset = unsafe { dev.add(ipc_opts.required_cachelines() * instance_id) };
        let ipc_base: u64 = if machine_id == 1 {
            CXLIpc::init(base_cl_with_offset, ipc_opts)
        } else {
            thread::sleep(Duration::from_secs(1));
            CXLIpc::from_cl_sync(base_cl_with_offset, ipc_opts).unwrap()
        }
        .get_base_cl() as u64;

        println!(
            "[{instance_id}] CXL IPC @ {} init done.",
            Cacheline::ptr_to_offset(base_cl_with_offset)
        );

        // -- Host to CXL
        let mut tun_reader = unsafe { File::from_raw_fd(libc::dup(tun.as_raw_fd())) };

        // Observing stats
        let tx_bytes_obs: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        let tx_packets_obs: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        let tx_spins_obs: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));

        let tx_bytes = tx_bytes_obs.clone();
        let tx_packets = tx_packets_obs.clone();
        let tx_spins = tx_spins_obs.clone();

        let tx = std::thread::spawn(move || {
            set_machine_id(machine_id);
            let mut ipc = CXLIpc::from_cl_sync(ipc_base as *mut Cacheline, ipc_opts).unwrap();
            let mut buf = [0u8; IFF_BUF];

            info!("[{instance_id}] Host to CXL ready.");

            loop {
                // Get a packet
                let packet_len = tun_reader.read(&mut buf).unwrap();

                // The transport routes IPv4 10.0.0.<machine_id>. Linux also
                // emits IPv6 link-local traffic when an interface comes up;
                // interpreting byte 19 of that traffic as an IPv4 host ID can
                // produce a permanent retry on an invalid peer.
                if packet_len < 20 || buf[0] >> 4 != 4 {
                    continue;
                }
                let target_host_id = buf[19] as u16;

                if target_host_id == machine_id || target_host_id == 0 || target_host_id > num_hosts
                {
                    // Drop loopback and packets outside the CXL TUN subnet.
                    continue;
                }

                let ip_slice = &buf[..packet_len];

                // Relay the packet
                let mut packet_spins = 0;
                loop {
                    match ipc.try_send_slice_to(target_host_id, ip_slice) {
                        Ok(_) => break,
                        Err(_err) => {
                            packet_spins += 1;
                            tx_spins.fetch_add(1, Ordering::Relaxed);
                            // Just retry
                        }
                    }
                }
                record_wait_spins(packet_spins);

                tx_bytes.fetch_add(packet_len as u64, Ordering::Relaxed);
                tx_packets.fetch_add(1, Ordering::Relaxed);
            }
        });
        threads.push(tx);

        // -- CXL to Host. Each source has its own SPSC ring, so receive from
        // those rings in parallel instead of making one worker scan all peers.
        // This avoids an O(hosts) polling path for every small MPI packet.
        let rx_bytes_obs: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        let rx_packets_obs: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));

        for source_id in 1..=num_hosts {
            if source_id == machine_id {
                continue;
            }
            let mut tun_writer = unsafe { File::from_raw_fd(libc::dup(tun.as_raw_fd())) };
            let rx_bytes = rx_bytes_obs.clone();
            let rx_packets = rx_packets_obs.clone();
            let rx = std::thread::spawn(move || {
                set_machine_id(machine_id);
                let mut ipc = CXLIpc::from_cl_sync(ipc_base as *mut Cacheline, ipc_opts).unwrap();

                info!("[{instance_id}] CXL to Host from peer {source_id} ready.");

                loop {
                    // Publish the polling spin counter at a bounded cadence. A
                    // never-ending await would retain all idle spins until the
                    // next packet and make a before/after benchmark snapshot
                    // inherit spins from an earlier run.
                    match ipc.await_slice_from(source_id, Some(Duration::from_millis(1))) {
                        Ok(packet) => {
                            let buf = packet.1.as_slice();

                            tun_writer.write_all(buf).unwrap();

                            rx_bytes.fetch_add(buf.len() as u64, Ordering::Relaxed);
                            rx_packets.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_timeout) => {}
                    }
                }
            });
            threads.push(rx);
        }

        let obs = thread::spawn(move || {
            let mut last_tx_bytes = 0;
            let mut last_tx_packets = 0;
            let mut last_tx_spins = 0;
            let mut last_rx_bytes = 0;
            let mut last_rx_packets = 0;
            loop {
                thread::sleep(Duration::from_secs(1));

                let tx_bytes = tx_bytes_obs.load(Ordering::Relaxed);
                let tx_packets = tx_packets_obs.load(Ordering::Relaxed);
                let tx_spins = tx_spins_obs.load(Ordering::Relaxed);

                let rx_bytes = rx_bytes_obs.load(Ordering::Relaxed);
                let rx_packets = rx_packets_obs.load(Ordering::Relaxed);

                if tx_packets - last_tx_packets != 0
                    || rx_bytes - last_rx_bytes != 0
                    || tx_spins - last_tx_spins != 0
                {
                    println!(
                        "[{instance_id}] | TX: {:.2} MB/s ({} pkt/s) spins/s: {} | RX: {:.2} MB/s ({} pkt/s)",
                        (tx_bytes - last_tx_bytes) as f64 / 1_000_000.0,
                        (tx_packets - last_tx_packets),
                        (tx_spins - last_tx_spins),
                        (rx_bytes - last_rx_bytes) as f64 / 1_000_000.0,
                        (rx_packets - last_rx_packets),
                    );
                }

                last_tx_bytes = tx_bytes;
                last_tx_packets = tx_packets;
                last_tx_spins = tx_spins;

                last_rx_bytes = rx_bytes;
                last_rx_packets = rx_packets;
            }
        });
        threads.push(obs);
    }

    for i in threads {
        i.join().unwrap();
    }
    control.join().unwrap();
    metrics.join().unwrap();
}
