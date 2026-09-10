use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::time::Duration;

// A minimal MPI-style message-passing layer over standard TCP sockets. Every
// rank holds a full-duplex connection to every other rank (established once),
// and communication happens in BSP rounds: each round every rank sends one
// length-framed message to every peer and receives one from every peer. This
// is the "standard network" counterpart to the CXL shared-memory engine.
pub struct NetComm {
    pub rank: usize,
    pub nranks: usize,
    read: Vec<Option<TcpStream>>,
    write: Vec<Option<TcpStream>>,
}

pub fn parse_peers(csv: &str) -> Vec<(String, u16)> {
    csv.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let (h, p) = s.rsplit_once(':').expect("peer must be host:port");
            (h.to_string(), p.parse::<u16>().expect("bad port"))
        })
        .collect()
}

fn short(h: &str) -> &str {
    h.split('.').next().unwrap_or(h)
}

pub fn infer_rank(peers: &[(String, u16)]) -> Option<usize> {
    let out = Command::new("hostname").output().ok()?;
    let hn = String::from_utf8_lossy(&out.stdout).trim().to_string();
    peers.iter().position(|(h, _)| short(h) == short(&hn))
}

impl NetComm {
    pub fn connect(peers: &[(String, u16)], rank: usize) -> Self {
        let n = peers.len();
        let listener = TcpListener::bind(("0.0.0.0", peers[rank].1))
            .unwrap_or_else(|e| panic!("bind {}: {e}", peers[rank].1));

        let mut read: Vec<Option<TcpStream>> = (0..n).map(|_| None).collect();
        let mut write: Vec<Option<TcpStream>> = (0..n).map(|_| None).collect();

        // Connect to all lower-ranked peers; they accept us.
        for j in 0..rank {
            let s = loop {
                match TcpStream::connect((peers[j].0.as_str(), peers[j].1)) {
                    Ok(s) => break s,
                    Err(_) => std::thread::sleep(Duration::from_millis(100)),
                }
            };
            s.set_nodelay(true).ok();
            let mut s = s;
            s.write_all(&(rank as u32).to_le_bytes()).unwrap();
            s.flush().unwrap();
            read[j] = Some(s.try_clone().unwrap());
            write[j] = Some(s);
        }

        // Accept from all higher-ranked peers.
        for _ in (rank + 1)..n {
            let (mut s, _) = listener.accept().unwrap();
            s.set_nodelay(true).ok();
            let mut b = [0u8; 4];
            s.read_exact(&mut b).unwrap();
            let peer = u32::from_le_bytes(b) as usize;
            read[peer] = Some(s.try_clone().unwrap());
            write[peer] = Some(s);
        }

        NetComm {
            rank,
            nranks: n,
            read,
            write,
        }
    }

    // One BSP round: send `out[j]` to peer j, return what each peer sent us.
    // A dedicated writer thread per peer avoids head-of-line blocking; the main
    // thread reads from every peer concurrently.
    pub fn exchange(&mut self, out: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let rank = self.rank;
        let mut inb: Vec<Vec<u8>> = (0..self.nranks).map(|_| Vec::new()).collect();
        let readers = &mut self.read;
        let writers = &mut self.write;

        std::thread::scope(|s| {
            for (j, w) in writers.iter_mut().enumerate() {
                if j == rank {
                    continue;
                }
                let data = &out[j];
                let w = w.as_mut().unwrap();
                s.spawn(move || {
                    w.write_all(&(data.len() as u64).to_le_bytes()).unwrap();
                    w.write_all(data).unwrap();
                    w.flush().unwrap();
                });
            }
            for (j, r) in readers.iter_mut().enumerate() {
                if j == rank {
                    continue;
                }
                let r = r.as_mut().unwrap();
                let mut lb = [0u8; 8];
                r.read_exact(&mut lb).unwrap();
                let len = u64::from_le_bytes(lb) as usize;
                let mut buf = vec![0u8; len];
                r.read_exact(&mut buf).unwrap();
                inb[j] = buf;
            }
        });
        inb
    }

    // All-gather one u64 per rank (used for global termination / reductions).
    pub fn allgather_u64(&mut self, val: u64) -> Vec<u64> {
        let payload = val.to_le_bytes().to_vec();
        let out = vec![payload; self.nranks];
        let inb = self.exchange(&out);
        (0..self.nranks)
            .map(|j| {
                if j == self.rank {
                    val
                } else {
                    u64::from_le_bytes(inb[j][..8].try_into().unwrap())
                }
            })
            .collect()
    }

    // Gather each rank's `owned` bytes onto rank 0. Returns per-rank buffers
    // (only meaningful on rank 0; rank r's own slot is filled locally).
    pub fn gather_to_root(&mut self, owned: &[u8]) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = (0..self.nranks).map(|_| Vec::new()).collect();
        if self.rank != 0 {
            out[0] = owned.to_vec();
        }
        let mut inb = self.exchange(&out);
        inb[self.rank] = owned.to_vec();
        inb
    }
}

pub fn ser_u64(v: &[u64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 8);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

pub fn deser_u64(b: &[u8]) -> Vec<u64> {
    b.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn ser_pairs(v: &[(u64, f64)]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 16);
    for (id, val) in v {
        b.extend_from_slice(&id.to_le_bytes());
        b.extend_from_slice(&val.to_bits().to_le_bytes());
    }
    b
}

pub fn for_each_pair(b: &[u8], mut f: impl FnMut(u64, f64)) {
    for c in b.chunks_exact(16) {
        let id = u64::from_le_bytes(c[0..8].try_into().unwrap());
        let val = f64::from_bits(u64::from_le_bytes(c[8..16].try_into().unwrap()));
        f(id, val);
    }
}
