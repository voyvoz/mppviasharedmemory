# CXL benchmark

This directory holds exactly two files you run:

| file | purpose |
|---|---|
| `paper_bench.sh` | the benchmark. Fetches datasets, builds and ships binaries, runs the whole matrix, writes the CSV. Self-contained -- host list, SSH setup and the IP-over-CXL bridge are all inside it. |
| `aggregate_runs.py` | turns one or more run CSVs into a plotting CSV plus one `.dat` per algorithm. |

## Running the benchmark

The one-liner, from the repository root:

```sh
scripts/paper_bench.sh
```

That fetches, builds and runs everything, then prints a summary. To do it in
stages, or to see what a configuration would run without running it:

```sh
scripts/paper_bench.sh plan         # print the run matrix, run nothing
scripts/paper_bench.sh fetch        # only stage datasets (locally and on all hosts)
scripts/paper_bench.sh build        # only build and deploy the binaries
scripts/paper_bench.sh run          # only run (assumes fetch + build are done)
scripts/paper_bench.sh csv          # re-summarise results/bench.csv
```

Configuration is by environment variable, so a narrower run is a prefix. For
example, the full Sort matrix on two and three hosts including the MPI and
IP-over-CXL variants, three repetitions each:

```sh
ALGOS=sort PROFILE=extended NUM_DATASETS=4 REPEATS=3 HOST_COUNTS="2 3" \
    WITH_IP_CXL=1 MPI_CROSS_HOST=1 STOP_ON_DAX_FAILURE=1 \
    scripts/paper_bench.sh
```

Always run with `STOP_ON_DAX_FAILURE=1` on the shared testbed: it aborts the
matrix the moment a DAX device disappears instead of grinding through a few
hundred failing runs. Put the results somewhere of their own with
`RESULTS_DIR=...` when you do not want to overwrite `results/bench.csv`.

`MODE=auto` (the default) uses the cluster when the CXL hosts answer and falls
back to local single-host runs otherwise; force it with `MODE=cluster` or
`MODE=local`. In local mode the multi-host rows are still emitted, with
`status=skipped_no_cxl`, so the CSV shape does not change.

## Collecting the CSV

Each run writes `results/bench_<stamp>.csv` and copies it to
`results/bench.csv` (or under `RESULTS_DIR`, if set). One row per
`(algorithm, variant, dataset, hosts, repeat)`, so a bar plot is a group-by:

```python
df = pd.read_csv("results/bench.csv")
df = df[df.status == "ok"]
df.groupby(["algorithm", "variant", "dataset"]).wall_s.median().unstack().plot.bar()
```

Columns: `run_id, started_at, algorithm, variant, class, dataset, dataset_kind,
dataset_detail, hosts, threads, repeat, wall_s, kernel_s, load_s, cxl_load_s,
cxl_store_s, wait_spins, status, validation, exit_code, log, cmd`. `wall_s` is
end-to-end process time (the slowest host for a multi-host run), `kernel_s` the
algorithm's own reported time, and `load_s` the dataset ingest. `cxl_load_s` and
`cxl_store_s` are accumulated scoped access durations (including concurrent
worker-thread time), summed across participating processes; `wait_spins` is also
summed. `aggregate_runs.py` divides the CXL counters by `hosts` to report a
per-host mean. The access durations are work totals rather than a partition of
wall time, so plot them as grouped bars, not stacked under `wall_s`. All three
are numeric zero for variants that do not touch CXL, which keeps the CSV directly
plottable.

Aggregate a benchmark CSV into one plotting CSV plus one `.dat` CSV per
algorithm:

```bash
python3 scripts/aggregate_runs.py results/bench.csv -o results/aggregated
```

Pass multiple run CSVs to compare host counts in one output; successful rows are
grouped by algorithm, variant, hosts, class, and dataset:

```bash
python3 scripts/aggregate_runs.py results/bench_2hosts.csv results/bench_3hosts.csv \
  -o results/aggregated
```

Sort contains exactly three radix-sort backends: `local` (threaded, non-CXL),
`cxl` (parallel CXL hosts), and `mpi_npN` (N rsmpi ranks). In cluster builds,
`WITH_MPI=1` builds `cxl-sort-mpi` natively on the first host and deploys it to
the full host list. Local builds also build this feature-gated binary when
`WITH_MPI=1` and a local MPI development installation is available.

`NATIVE_CXL_BUILD=1` (the default in cluster mode) builds the direct-CXL
binaries and the TUN bridge natively on the first host as well, then deploys
them to the full host list. Cross-building them against musl instead makes the
memory-bound kernels about 3.8x slower than the identical code built against
glibc -- a purely allocator-driven gap that otherwise appears as a direct-CXL
versus MPI/IP-over-CXL difference, because the rsmpi binaries can only be built
natively. Set `NATIVE_CXL_BUILD=0` to return to the musl cross-build; the
harness also falls back to it automatically if the launcher cannot build.
All benchmark hosts run the same Ubuntu 24.04 / glibc 2.39 image.

The MPI radix backend range-partitions the `u64` key space with one Alltoallv,
then sorts each rank's contiguous key range in local DRAM with six 11-bit LSD
passes. Rank-order concatenation is globally sorted. This avoids the former
eight full-data network redistributions, one after every 8-bit radix pass.
The rank-local bucket is excluded from MPI and sorted concurrently with the
network transfer; after the remote bucket is sorted, the two runs are merged.
Validation is distributed as well: local and cross-rank ordering plus count,
sum, XOR, and mixed-hash fingerprints replace the former full result gather.

The direct-CXL radix path range-partitions the key space with one exchange and
keeps active partitions in host-local DRAM. Each host publishes only peer-owned
buckets to segmented CXL staging, sorts its local bucket concurrently with CXL
traffic, and sorts and merges the received buckets. Every bucket is cacheline
padded, so no two hosts write the same cacheline and CXL 2.0 coherence is never
assumed. Exchange traffic moves in parallel bulk transfers: writes use
non-temporal stores, which skip the read-for-ownership that an ordinary store to
write-back memory would put on the link, and reads invalidate a whole chunk
before copying it out. Like the MPI backend, the kernel ends with the result
distributed across the hosts; gathering it into one readable shared array is a
separate `publish` step reported as its own `[cxl_radix_phase] publish=` line
and excluded from `BENCH_KERNEL_NS`, matching the MPI backend's distributed
validation.

Both backends share the host-local sort and merge: six 11-bit LSD passes over
two alternating buffers, so a multi-hundred-megabyte partition is allocated and
first-touched once rather than once per pass, and a parallel merge that cuts the
left run into equal pieces and locates each cut in the right run by binary
search.

BFS `cxl_shared` keeps an unpartitioned frontier in CXL and replicates the
immutable CSR plus distance vector in host-local DRAM. Every host consumes the
same published frontier segments, so visited state remains identical without
assuming CXL 2.0 cache coherence or locking every discovered vertex. BFS
Pregel variants are `pregel_npN` and use rsmpi Allgather/Alltoallv rather than
the CXL IPC rings. With `WITH_MPI=1`, the feature-gated `cxl-bfs-mpi` binary is
built and deployed like `cxl-sort-mpi`. Packed BFS MPI baselines explicitly map
and bind one rank per core; this avoids Open MPI's rank-count-dependent default
binding policy. Cross-host BFS baselines map one rank per node.

With `WITH_IP_CXL=1`, the suite adds `ip_cxl_npN` variants for the MPI-capable
BFS, radix Sort, and MiniSat implementations. Open MPI's TCP and control planes
are pinned to a Linux `cxl` TUN interface whose packets are carried by the CXL
IPC rings. The bridge uses a dedicated 1 GiB mapping at `0x1800000000`
(96 GiB), outside the eight reusable algorithm windows, and runs only during
the final TUN phase.
Each MPI rank synchronously starts its local bridge counters at kernel entry and
stops/prints them at kernel exit, using the same `timestamp.rs` `BENCH_CXL_*`
points as direct-CXL kernels; launcher setup and idle polling are excluded.
The bridge uses one receive worker per peer ring. Three-rank cross-host BFS,
both ordinary TCP and IP-over-CXL, additionally selects Open MPI's ring
Allgather and basic-linear Alltoallv, avoiding its slower default small-message
collective choice on these transports.
Bring up and validate the bridge with:

```bash
scripts/paper_bench.sh ip-cxl up 3
scripts/paper_bench.sh ip-cxl check 3
scripts/paper_bench.sh ip-cxl down 3
```

`run` brings the bridge up and tears it down on its own, so this is only needed
when `sudo` wants a password (bring it up once with `IP_CXL_INTERACTIVE=1`), or
to inspect or clean up afterwards. The two small helpers the bridge needs on the
remote hosts are written out by the script; there is nothing else to deploy.

The restricted sudo rule executes only the no-argument `tun` binary. The helper
invokes plain `sudo` (without `-n`), uses privilege to create/configure the
interface, and immediately drops to the
invoking UID; benchmark and MPI executables remain unprivileged. The setup also
installs a marked, temporary MPI launch key and removes it during teardown.

Branch-and-bound knapsack (`ALGOS=bnb`) runs one binary (`cxl-bnb`) behind a
`--fabric` switch: `local` (single host, no transport), `mpi_npN` (rsmpi
all-reduce, one rank per host with `MPI_CROSS_HOST=1`), `cxl` (blocking
shared-memory reduction) and `cxlv2` (latency-hiding reduction: only every
third reduction call returns fresh data, so it reduces often and, because its
termination check is blocking, checks termination rarely). Cadence knobs:
`BNB_STEPS=1024`/`BNB_TERM=1` for local/MPI/CXL, `BNB_V2_STEPS=128`/
`BNB_V2_TERM=24` for CXLv2, and `BNB_DATASETS` overrides the four Pisinger
files. Because every instance is asserted against its known optimum, a run's
`validation=ok` really is a full-file correctness check. Run it with all four
scales via `NUM_DATASETS=4` (the default caps datasets at two).

### Datasets

| kernel | datasets (`PROFILE=small` / additions in `PROFILE=extended`) | source |
|---|---|---|
| bnb | `knapPI_2_100_1000`, `knapPI_14_50_1000`, `knapPI_16_50_1000`, `knapPI_15_50_1000` (all profiles) | Pisinger knapsack instances, shipped in `cxl-bnb/test_data` |
| bfs | `belgium_osm` + `as-Skitter` / `sx-stackoverflow` + `soc-LiveJournal1` | SuiteSparse (DIMACS10, SNAP) |
| kmeans | `uci-covtype` + `uci-census1990` / `uci-susy` + `uci-higgs` | UCI |
| minisat | `satlib-uf250` + `satlib-flat200` / `dimacs-ii16b1` + `dimacs-ssa7552` | SATLIB, DIMACS |
| sort | `keys-2p22/24` / `keys-2p26/28` | Sort Benchmark sizing, keys generated in-engine |
| matmul | `hpl-n1024/2048` / `hpl-n2304/2560` | HPL sizing, matrix generated |
| stencil | `jacobi-2k-50`, … | HPCG/NPB sizing, grid generated |

Every kernel pairs one high-diameter / low-degree input with one power-law or
larger input, following the GAP Benchmark Suite's rationale. Sort, matmul and
stencil have no file input in the crates — their standard inputs are generators,
not files — so they run two standardised instances each and are marked
`dataset_kind=generated` in the CSV.
MiniSat is invoked with `--core` for both ordinary MPI and IP-over-CXL variants.

### Knobs

`MODE=auto|local|cluster` (auto = cluster if the CXL hosts answer),
`PROFILE=small|extended|full`, `ALGOS`, `HOST_COUNTS="2 3"`, `REPEATS=3`, `THREADS`,
`RUN_TIMEOUT`, `MAX_POINTS` (cap k-means rows), `WITH_MPI=1` (adds the MPI
baselines and the MPI-only `minisat`), `WITH_IP_CXL=1` (implies `WITH_MPI=1`
and adds the TUN variants), `MPI_CROSS_HOST=1` (ordinary MPI uses one rank per
host; a marked temporary launch key is installed and removed automatically),
and `IP_CXL_ONLY=1` (rerun only those TUN variants).
Set `STOP_ON_DAX_FAILURE=1` to abort the matrix immediately if a DAX device
disappears or a run reports SIGBUS/DAX mapping failure. The host list, SSH
setup and the CXL window are configured at the top of `paper_bench.sh`; override
the host set for a run with `HOSTS_STR="dbcxla dbcxlc"` when a node is down.

The extended profile uses per-case DAX mappings: normal cases remain at 1 GiB,
`keys-2p26` uses 2 GiB, and `keys-2p28` uses 8 GiB. Direct-CXL windows are
spaced 8 GiB apart, and the IP-over-CXL window starts at 96 GiB. The runner
checks the complete range against every DAX device before starting and stops on
a missing device, SIGBUS/mmap failure, or CXL allocator exhaustion.
