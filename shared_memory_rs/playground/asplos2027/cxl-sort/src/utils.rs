use clap::Parser;

#[derive(Clone, Debug, Parser)]
pub struct Cli {
    /// Number of CXL hosts. Values >= 2 select distributed CXL radix sort.
    #[arg(short, long, default_value_t = 1)]
    pub num_hosts: usize,

    #[arg(short, long, default_value_t = 0)]
    pub threads: usize,

    #[arg(long, default_value_t = 4096)]
    pub nodes: usize,

    #[arg(long, default_value_t = 0xC0FFEE)]
    pub seed: u64,

    /// Select the rsmpi distributed radix-sort backend.
    #[arg(long)]
    pub mpi: bool,
}

pub fn resolve_threads(requested: usize) -> usize {
    if requested != 0 {
        requested
    } else {
        std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1)
    }
}

#[inline]
pub fn sort_key_at(index: u64, seed: u64) -> u64 {
    let mut value = index.wrapping_add(seed).wrapping_mul(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

pub fn gen_keys_range(seed: u64, start: usize, count: usize) -> Vec<u64> {
    (0..count)
        .map(|offset| sort_key_at((start + offset) as u64, seed))
        .collect()
}

/// Order-independent summary of a key multiset.
///
/// A result that is in non-decreasing order and has the same fingerprint as the
/// generated input is the sequentially sorted input: sortedness fixes the order
/// and the fingerprint fixes the multiset. Checking it this way costs one pass
/// and no reference copy, and every backend can apply the same check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fingerprint {
    pub count: u64,
    pub sum: u64,
    pub mixed: u64,
    pub xor: u64,
}

#[inline]
pub fn fingerprint_mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl Fingerprint {
    pub fn of(values: impl Iterator<Item = u64>) -> Self {
        let mut result = Self::default();
        for value in values {
            result.count = result.count.wrapping_add(1);
            result.sum = result.sum.wrapping_add(value);
            result.mixed = result.mixed.wrapping_add(fingerprint_mix(value));
            result.xor ^= value;
        }
        result
    }

    /// Combining is commutative and associative, so workers may split the
    /// input any way they like.
    pub fn merge(self, other: Self) -> Self {
        Self {
            count: self.count.wrapping_add(other.count),
            sum: self.sum.wrapping_add(other.sum),
            mixed: self.mixed.wrapping_add(other.mixed),
            xor: self.xor ^ other.xor,
        }
    }

    /// `[count, sum, mixed]` reduce with addition, `xor` with exclusive or.
    pub fn sums(&self) -> [u64; 3] {
        [self.count, self.sum, self.mixed]
    }

    pub fn from_parts(sums: [u64; 3], xor: u64) -> Self {
        Self {
            count: sums[0],
            sum: sums[1],
            mixed: sums[2],
            xor,
        }
    }
}

fn fold_parallel<F>(len: usize, threads: usize, worker: F) -> Fingerprint
where
    F: Fn(usize, usize) -> Fingerprint + Sync,
{
    let threads = threads.max(1).min(len.max(1));
    if threads == 1 {
        return worker(0, len);
    }
    let worker = &worker;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|index| {
                let begin = len * index / threads;
                let end = len * (index + 1) / threads;
                scope.spawn(move || worker(begin, end))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .fold(Fingerprint::default(), Fingerprint::merge)
    })
}

pub fn fingerprint_slice(values: &[u64], threads: usize) -> Fingerprint {
    fold_parallel(values.len(), threads, |begin, end| {
        Fingerprint::of(values[begin..end].iter().copied())
    })
}

/// Fingerprint the generated input without materialising it.
pub fn fingerprint_generated(n: usize, seed: u64, threads: usize) -> Fingerprint {
    fold_parallel(n, threads, |begin, end| {
        Fingerprint::of((begin..end).map(|index| sort_key_at(index as u64, seed)))
    })
}

pub fn is_sorted(values: &[u64], threads: usize) -> bool {
    if values.len() < 2 {
        return true;
    }
    // Each worker also compares across its left boundary, so the chunks
    // together cover every adjacent pair.
    let threads = threads.max(1).min(values.len() - 1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|index| {
                let begin = 1 + (values.len() - 1) * index / threads;
                let end = 1 + (values.len() - 1) * (index + 1) / threads;
                scope.spawn(move || values[begin - 1..end].windows(2).all(|w| w[0] <= w[1]))
            })
            .collect();
        handles.into_iter().all(|handle| handle.join().unwrap())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_and_materialised_fingerprints_agree() {
        for threads in [1usize, 3, 8] {
            let keys = gen_keys_range(0x1234, 0, 10_007);
            assert_eq!(
                fingerprint_slice(&keys, threads),
                fingerprint_generated(10_007, 0x1234, threads)
            );
        }
    }

    #[test]
    fn fingerprint_ignores_order_but_not_content() {
        let keys = gen_keys_range(0x99, 0, 5_003);
        let mut shuffled = keys.clone();
        shuffled.sort_unstable();
        assert_eq!(fingerprint_slice(&keys, 4), fingerprint_slice(&shuffled, 4));

        let mut altered = keys.clone();
        altered[17] = altered[17].wrapping_add(1);
        assert_ne!(fingerprint_slice(&keys, 4), fingerprint_slice(&altered, 4));

        // A swap that preserves the sum is still caught by the mixed hash.
        let mut swapped = keys.clone();
        swapped[3] = swapped[3].wrapping_add(64);
        swapped[4] = swapped[4].wrapping_sub(64);
        assert_ne!(fingerprint_slice(&keys, 4), fingerprint_slice(&swapped, 4));
    }

    #[test]
    fn is_sorted_finds_every_inversion() {
        for threads in [1usize, 2, 7, 64] {
            assert!(is_sorted(&[], threads));
            assert!(is_sorted(&[5], threads));
            assert!(is_sorted(&[1, 1, 2, 9], threads));
            assert!(!is_sorted(&[1, 2, 1], threads));
            let mut keys = gen_keys_range(0x77, 0, 1_009);
            keys.sort_unstable();
            assert!(is_sorted(&keys, threads));
            // An inversion at every position must be detected, including the
            // ones that fall on a worker boundary.
            for index in 1..keys.len() {
                let mut broken = keys.clone();
                broken.swap(index - 1, index);
                if broken[index - 1] != broken[index] {
                    assert!(!is_sorted(&broken, threads), "missed inversion at {index}");
                }
            }
        }
    }
}
