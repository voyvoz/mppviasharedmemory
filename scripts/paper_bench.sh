#!/usr/bin/env bash
# ============================================================================
# paper_bench.sh -- one-shot benchmark driver for the ASPLOS'27 CXL kernels.
#
# Downloads two standard datasets per algorithm, builds and (optionally) ships
# the binaries, runs every algorithm x variant x dataset combination, and writes
# one tidy CSV of end-to-end runtimes:
#
#   results/bench_<stamp>.csv   (and results/bench.csv, a copy of the latest)
#
# CSV is one row per (algorithm, variant, dataset, hosts, repeat) so a bar plot
# is a straight group-by, e.g.
#
#   df = pd.read_csv("results/bench.csv")
#   df = df[df.status == "ok"]
#   df.groupby(["algorithm", "variant", "dataset"]).wall_s.median().unstack().plot.bar()
#
# ---------------------------------------------------------------------------
# Usage
#   scripts/paper_bench.sh                  # fetch + build + run everything
#   scripts/paper_bench.sh fetch            # only stage the datasets
#   scripts/paper_bench.sh build            # only build the binaries
#   scripts/paper_bench.sh run              # only run (assumes fetch+build done)
#   scripts/paper_bench.sh csv              # re-parse existing logs into a CSV
#   scripts/paper_bench.sh plan             # print what would run, run nothing
#   scripts/paper_bench.sh ip-cxl up        # bring the IP-over-CXL bridge up by hand
#
# Knobs (environment)
#   MODE=auto|local|cluster   auto: cluster if the CXL hosts answer, else local
#   PROFILE=small|extended|full
#                             extended: small inputs plus two larger datasets
#   ALGOS="bfs sssp ..."      subset of algorithms to run
#   HOST_COUNTS="2 3"         multi-host configurations to sweep
#   REPEATS=3                 repetitions per configuration
#   THREADS=0                 0 = all cores (crate default)
#   RUN_TIMEOUT=1800          per-run timeout in seconds
#   MAX_POINTS=0              cap on k-means input rows (0 = whole file)
#   NUM_DATASETS=2            datasets per algorithm (1 = first slot only)
#   WITH_MPI=0                also run the MPI baselines (needs mpirun + rsmpi)
#   MPI_CROSS_HOST=0          0: pack ordinary MPI ranks; 1: one rank per host
#   WITH_IP_CXL=0             run cross-host MPI over the CXL-backed TUN device
#   IP_CXL_ONLY=0             skip direct/local cases and run only the TUN phase
#   STOP_ON_DAX_FAILURE=0     abort immediately if DAX disappears or SIGBUSes
#   BFS_MPI_ONLY=0            for BFS, run only the MPI Pregel rank sweep
#   NATIVE_CXL_BUILD=1        build the direct-CXL binaries natively on the
#                             launcher instead of cross-building against musl,
#                             so they use the same libc as the MPI variants
#
# Cluster mode reuses the host list, SSH setup and CXL window from common.sh.
# ============================================================================
set -uo pipefail

STAGE="${1:-all}"

MODE=${MODE:-auto}
PROFILE=${PROFILE:-small}
REPEATS=${REPEATS:-3}
THREADS=${THREADS:-0}
HOST_COUNTS=${HOST_COUNTS-"2 3"} # empty string = single-host runs only
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
MAX_POINTS=${MAX_POINTS:-0}
WITH_MPI=${WITH_MPI:-0}
WITH_IP_CXL=${WITH_IP_CXL:-0}
[ "$WITH_IP_CXL" = 1 ] && WITH_MPI=1
IP_CXL_ONLY=${IP_CXL_ONLY:-0}
STOP_ON_DAX_FAILURE=${STOP_ON_DAX_FAILURE:-0}
BFS_MPI_ONLY=${BFS_MPI_ONLY:-0}
NATIVE_CXL_BUILD=${NATIVE_CXL_BUILD:-1}
NATIVE_DEPLOYED=""
MPI_RANKS=1 # set per MPI benchmark case
NUM_DATASETS=${NUM_DATASETS:-2} # datasets per algorithm (1 = first slot only)
SKIP_REASON="" # set by a bench_ function to force-skip the next run_case
ALGOS=${ALGOS:-"bfs pagerank triangle sort kmeans matmul stencil minisat"}

# Branch-and-bound knapsack reduction cadence. MPI and CXL block on every
# reduction, so they check termination after each one. CXLv2 hides the
# reduction latency (only every third call returns fresh data) but its
# termination check is blocking, so it reduces more often and checks rarely.
BNB_STEPS=${BNB_STEPS:-1024}
BNB_TERM=${BNB_TERM:-1}
BNB_V2_STEPS=${BNB_V2_STEPS:-128}
BNB_V2_TERM=${BNB_V2_TERM:-24}
# Four scales of Pisinger hard knapsack instances, calibrated on dbcxla so the
# largest stays inside a two-minute single-host run. Every file is solved
# whole (100 instances each); the families escalate in search effort.
BNB_DATASETS=${BNB_DATASETS:-"knapPI_2_100_1000 knapPI_14_50_1000 knapPI_16_50_1000 knapPI_15_50_1000"}

# Distinct name: common.sh defines its own `_self` and would clobber a shared one.
_paper_bench_self="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$_paper_bench_self")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "$_paper_bench_self")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RS_ROOT="$PROJECT_ROOT/shared_memory_rs"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
LOG_DIR="$RESULTS_DIR/logs"

# ---------------------------------------------------------------------------
# datasets
#
# Registry line: name|url|archive|find_pattern|canonical_name
# `archive` is tar.gz | zip | gz | plain. After unpacking, the first file
# matching `find_pattern` is normalised to data/<name>/<canonical_name>
# (transparently gunzipped if the match is a .gz).
#
# The graph, SAT and k-means kernels consume these files directly. Sort, matmul
# and stencil have no file input in the current crates -- their standard inputs
# are *generators* (Sort Benchmark gensort, HPL, HPCG), so they are driven by two
# standardised problem instances each and recorded with dataset_kind=generated.
# ---------------------------------------------------------------------------
SUITESPARSE=https://sparse.tamu.edu/MM
SATLIB=https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT
DIMACS=https://archive.dimacs.rutgers.edu/pub/challenge/sat/benchmarks/cnf
UCI=https://archive.ics.uci.edu/static/public

dataset_spec() {
    case "$1" in
    # -- graphs (GAP Benchmark Suite style: one high-diameter, one power-law) --
    road_usa)       echo "road_usa|$SUITESPARSE/DIMACS10/road_usa.tar.gz|tar.gz|road_usa.mtx|road_usa.mtx" ;;
    com-Orkut)      echo "com-Orkut|$SUITESPARSE/SNAP/com-Orkut.tar.gz|tar.gz|com-Orkut.mtx|com-Orkut.mtx" ;;
    soc-LiveJournal1) echo "soc-LiveJournal1|$SUITESPARSE/SNAP/soc-LiveJournal1.tar.gz|tar.gz|soc-LiveJournal1.mtx|soc-LiveJournal1.mtx" ;;
    sx-stackoverflow) echo "sx-stackoverflow|$SUITESPARSE/SNAP/sx-stackoverflow.tar.gz|tar.gz|sx-stackoverflow.mtx|sx-stackoverflow.mtx" ;;
    belgium_osm)    echo "belgium_osm|$SUITESPARSE/DIMACS10/belgium_osm.tar.gz|tar.gz|belgium_osm.mtx|belgium_osm.mtx" ;;
    as-Skitter)     echo "as-Skitter|$SUITESPARSE/SNAP/as-Skitter.tar.gz|tar.gz|as-Skitter.mtx|as-Skitter.mtx" ;;
    # -- SAT (SATLIB: one uniform-random, one structured) --
    satlib-uf250)   echo "satlib-uf250|$SATLIB/RND3SAT/uf250-1065.tar.gz|tar.gz|uf250-01.cnf|uf250-01.cnf" ;;
    satlib-flat200) echo "satlib-flat200|$SATLIB/GCP/flat200-479.tar.gz|tar.gz|flat200-1.cnf|flat200-1.cnf" ;;
    satlib-par16)   echo "satlib-par16|$SATLIB/DIMACS/PARITY/parity.tar.gz|tar.gz|par16-1.cnf|par16-1.cnf" ;;
    dimacs-ii16b1)  echo "dimacs-ii16b1|$DIMACS/ii16b1.cnf.Z|compress|ii16b1.cnf|ii16b1.cnf" ;;
    dimacs-ssa7552) echo "dimacs-ssa7552|$DIMACS/ssa7552-038.cnf.Z|compress|ssa7552-038.cnf|ssa7552-038.cnf" ;;
    # -- k-means (UCI) --
    uci-covtype)    echo "uci-covtype|$UCI/31/covertype.zip|zip|covtype.data.gz|covtype.data" ;;
    uci-census1990) echo "uci-census1990|$UCI/116/us+census+data+1990.zip|zip|USCensus1990.data.txt|census1990.data" ;;
    uci-higgs)      echo "uci-higgs|$UCI/280/higgs.zip|zip|HIGGS.csv.gz|HIGGS.csv" ;;
    uci-susy)       echo "uci-susy|$UCI/279/susy.zip|zip|SUSY.csv.gz|SUSY.csv" ;;
    *) return 1 ;;
    esac
}

# datasets_for <algo> -> the dataset keys for the active profile, capped at
# NUM_DATASETS (1 = just the first slot, which halves the run matrix)
datasets_for() {
    _datasets_for "$1" | tr ' ' '\n' | grep -v '^$' | head -"$NUM_DATASETS" | tr '\n' ' '
}

_datasets_for() {
    case "$1:$PROFILE" in
    bfs:small | sssp:small | pagerank:small | triangle:small) echo "belgium_osm as-Skitter" ;;
    bfs:extended) echo "belgium_osm as-Skitter sx-stackoverflow soc-LiveJournal1" ;;
    bfs:full | sssp:full | pagerank:full | triangle:full)     echo "road_usa com-Orkut" ;;
    # Pisinger knapsack instances, four escalating difficulty scales. The files
    # ship with the repository (cxl-bnb/test_data), nothing to download.
    bnb:small | bnb:extended | bnb:full) echo "$BNB_DATASETS" ;;
    minisat:small) echo "satlib-uf250 satlib-flat200" ;;
    minisat:extended) echo "satlib-uf250 satlib-flat200 dimacs-ii16b1 dimacs-ssa7552" ;;
    minisat:full)  echo "satlib-par16 satlib-flat200" ;;
    kmeans:small)  echo "uci-covtype uci-census1990" ;;
    kmeans:extended) echo "uci-covtype uci-census1990 uci-susy uci-higgs" ;;
    kmeans:full)   echo "uci-covtype uci-higgs" ;;
    # generated instances, nothing to download
    sort:small)    echo "keys-2p22 keys-2p24" ;;
    sort:extended) echo "keys-2p22 keys-2p24 keys-2p26 keys-2p28" ;;
    sort:full)     echo "keys-2p26 keys-2p28" ;;
    matmul:small)  echo "hpl-n1024 hpl-n2048" ;;
    matmul:extended) echo "hpl-n1024 hpl-n2048 hpl-n2304 hpl-n2560" ;;
    matmul:full)   echo "hpl-n4096 hpl-n8192" ;;
    # Full 2x2 stencil sweep.  This supports controlled contrasts where only
    # the grid size or only the iteration count changes between compared rows.
    stencil:small) echo "jacobi-2k-50 jacobi-2k-100 jacobi-4k-50 jacobi-4k-100" ;;
    stencil:extended) echo "jacobi-2k-50 jacobi-2k-100 jacobi-4k-50 jacobi-4k-100" ;;
    stencil:full)  echo "jacobi-8k-100 jacobi-16k-200" ;;
    *) echo "" ;;
    esac
}

# generated_params <dataset_key> -> the numeric instance behind a generated name
generated_params() {
    case "$1" in
    keys-2p22) echo "4194304" ;;
    keys-2p24) echo "16777216" ;;
    keys-2p26) echo "67108864" ;;
    keys-2p28) echo "268435456" ;;
    hpl-n1024) echo "1024" ;;
    hpl-n2048) echo "2048" ;;
    hpl-n2304) echo "2304" ;;
    hpl-n2560) echo "2560" ;;
    hpl-n4096) echo "4096" ;;
    hpl-n8192) echo "8192" ;;
    jacobi-2k-50)   echo "2048 2048 50" ;;
    jacobi-2k-100)  echo "2048 2048 100" ;;
    jacobi-4k-50)   echo "4096 4096 50" ;;
    jacobi-4k-100)  echo "4096 4096 100" ;;
    jacobi-8k-100)  echo "8192 8192 100" ;;
    jacobi-16k-200) echo "16384 16384 200" ;;
    esac
}

# kmeans_args <dataset_key> <path> -> loader + clustering flags for that dataset
kmeans_args() {
    local cap=""
    [ "$MAX_POINTS" != 0 ] && cap="--max-points $MAX_POINTS"
    case "$1" in
    uci-covtype)    echo "--input $2 --input-cols 54 -k 7 --max-iters 50 $cap" ;;
    uci-census1990) echo "--input $2 --skip-rows 1 --skip-cols 1 -k 10 --max-iters 30 $cap" ;;
    uci-higgs)      echo "--input $2 --skip-cols 1 --input-cols 28 -k 10 --max-iters 20 $cap" ;;
    uci-susy)       echo "--input $2 --skip-cols 1 --input-cols 18 -k 2 --max-iters 20 $cap" ;;
    esac
}

dataset_path() { echo "$DATA_ROOT/$1/$(dataset_spec "$1" | cut -d'|' -f5)"; }

# ---------------------------------------------------------------------------
# binaries
# ---------------------------------------------------------------------------
crate_for() {
    case "$1" in
    bfs) echo cxl-bfs ;; sssp) echo cxl-sssp ;; pagerank) echo cxl-pagerank ;;
    triangle) echo cxl-triangle ;; sort) echo cxl-sort ;; kmeans) echo cxl-kmeans ;;
    matmul) echo cxl-matmul ;; stencil) echo cxl-stencil ;; minisat) echo cxl-minisat ;;
    bnb) echo cxl-bnb ;;
    esac
}

# ---------------------------------------------------------------------------
# portability helpers (this script also runs on macOS for local dry runs)
# ---------------------------------------------------------------------------
case "$(date +%s%N 2>/dev/null)" in
'' | *N*) NOW_IMPL=python ;;
*) NOW_IMPL=date ;;
esac
now_ns() {
    if [ "$NOW_IMPL" = date ]; then date +%s%N
    else python3 -c 'import time; print(int(time.time()*1e9))'
    fi
}

TIMEOUT_CMD=()
if command -v timeout >/dev/null 2>&1; then TIMEOUT_CMD=(timeout "$RUN_TIMEOUT")
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_CMD=(gtimeout "$RUN_TIMEOUT")
fi

log() { printf '%s\n' "$*" >&2; }
hr() { log "-----------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# cluster configuration
#
# Edit the host list, user and CXL window here. Launch order is discovery
# order: the first N entries are used for an N-host run.
# ---------------------------------------------------------------------------
USER_NAME=${USER_NAME:-abaumstark}
DOMAIN=${DOMAIN:-prakinf.tu-ilmenau.de}
REMOTE_DIR=${REMOTE_DIR:-cxl_graphs} # relative to the remote $HOME
TARGET_TRIPLE=${TARGET_TRIPLE:-x86_64-unknown-linux-musl}

fqdn() { echo "$1.$DOMAIN"; }
rsh() {
    local h="$1"
    shift
    ssh "${SSH_OPTS[@]}" "$USER_NAME@$(fqdn "$h")" "$@"
}
rcp() { scp "${SSH_OPTS[@]}" "$1" "$USER_NAME@$(fqdn "$2"):$3"; }

cluster_config() {
    if [ -n "${HOSTS_STR:-}" ]; then
        read -ra HOSTS <<<"$HOSTS_STR"
    else
        HOSTS=(dbcxla dbcxlb dbcxlc)
    fi
    # Persistent, multiplexed SSH connections: one master per host reused for
    # every command, so a whole benchmark is a handful of handshakes rather
    # than hundreds, which trips the servers' connection rate limiting.
    mkdir -p "$HOME/.ssh/cm"
    SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=20
        -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm/%r@%h:%p" -o ControlPersist=900)
}

# ---------------------------------------------------------------------------
# IP over CXL
#
# The repository's TUN bridge carries IP traffic over the CXL IPC rings. It
# lives in its own small DAX window, well clear of the algorithm windows, and
# runs only for the IP-over-CXL benchmark phase. The two remote helpers below
# are written out on demand so this script stays the only file you need.
# ---------------------------------------------------------------------------
IP_CXL_MTU=${IP_CXL_MTU:-65000}
IP_CXL_INTERACTIVE=${IP_CXL_INTERACTIVE:-0}
IP_CXL_MARKER=ip-cxl-paper-bench

# Open MPI launch helper, run on the launcher host.
write_ip_cxl_ssh_helper() {
    cat >"$1" <<'HELPER'
#!/usr/bin/env bash
# Open MPI launch helper for temporary IP-over-CXL benchmark credentials.
set -euo pipefail

key=${IP_CXL_KEY:-"$HOME/cxl_graphs/.ip-cxl/id_ed25519"}
exec /usr/bin/ssh -i "$key" \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=20 \
    "$@"
HELPER
}

# Bring the bridge up on one host. The testbed sudo rule permits only the exact
# no-argument `tun` executable, which drops privilege once the interface is up.
# The bridge routes by the destination IP's last octet == the host's CXL
# machine_id, and machine_id comes from CXL discovery rather than host order,
# so the address is read back from the log instead of assumed.
write_ip_cxl_up_helper() {
    cat >"$1" <<'HELPER'
#!/bin/bash
set -e
n="$1"
[ "$n" = 3 ] || { echo "the sudo-safe no-argument TUN launch requires three hosts"; exit 2; }
cd "$(dirname "$0")"

pkill -x tun 2>/dev/null || true
sleep 1
export CXL_MAPOFFSET="${CXL_MAPOFFSET:-0x600000000}"
export CXL_MAPSIZE="${CXL_MAPSIZE:-1G}"
: > ip-cxl-tun.log
nohup sudo "$HOME/tun" > ip-cxl-tun.log 2>&1 &

mid=""
for _ in $(seq 1 60); do
    mid=$(grep -m1 -oE 'machine_id=[0-9]+' ip-cxl-tun.log | cut -d= -f2 || true)
    [ -n "$mid" ] && break
    sleep 0.5
done
[ -n "$mid" ] || { echo "ERROR: tun never reported machine_id"; tail ip-cxl-tun.log; exit 1; }

echo "MID=${mid}"
ip -brief addr show cxl
echo "hybrid transport up: 10.0.0.${mid}/24 on dev cxl"
HELPER
}

ip_cxl_mid_for() {
    rsh "$1" "grep -m1 -oE 'machine_id=[0-9]+' \$HOME/$REMOTE_DIR/ip-cxl-tun.log 2>/dev/null | cut -d= -f2"
}

ip_cxl_launcher() {
    local h mid
    for h in "${IP_CXL_SEL[@]}"; do
        mid=$(ip_cxl_mid_for "$h" || true)
        [ "$mid" = 1 ] && {
            echo "$h"
            return 0
        }
    done
    return 1
}

ip_cxl_check() {
    local h mid seen=" " launcher expected_offset expected_size
    expected_offset=$((IP_CXL_OFFSET))
    case "$IP_CXL_MAPSIZE" in
    *G | *g) expected_size=$((${IP_CXL_MAPSIZE%[Gg]} * 1024 * 1024 * 1024)) ;;
    *M | *m) expected_size=$((${IP_CXL_MAPSIZE%[Mm]} * 1024 * 1024)) ;;
    *) expected_size=$((IP_CXL_MAPSIZE)) ;;
    esac
    for h in "${IP_CXL_SEL[@]}"; do
        mid=$(ip_cxl_mid_for "$h" || true)
        [ -n "$mid" ] || {
            echo "$h: no TUN machine_id" >&2
            return 1
        }
        case "$seen" in *" $mid "*)
            echo "duplicate machine_id=$mid" >&2
            return 1
            ;;
        esac
        seen="$seen$mid "
        rsh "$h" "test -c /dev/dax0.0; pgrep -x tun >/dev/null; \
            test -S \$HOME/$REMOTE_DIR/tun.metrics.sock; \
            ip -4 -brief addr show cxl | grep -q '10.0.0.$mid/24'; \
            ip -o link show cxl | grep -q 'mtu $IP_CXL_MTU'; \
            grep -q 'len=$expected_size,offset=$expected_offset' \
              \$HOME/$REMOTE_DIR/ip-cxl-tun.log"
        echo "$h: machine_id=$mid ip=10.0.0.$mid"
    done
    launcher=$(ip_cxl_launcher) || return 1
    for mid in $(seq 1 "${#IP_CXL_SEL[@]}"); do
        rsh "$launcher" "ping -I cxl -c 1 -W 3 10.0.0.$mid >/dev/null"
    done
    echo "IP_CXL_LAUNCHER=$launcher"
}

ip_cxl_install_keys() {
    local launcher pub h
    launcher=$(ip_cxl_launcher) || {
        echo "no TUN host has machine_id=1" >&2
        return 1
    }
    rsh "$launcher" "mkdir -p \$HOME/$REMOTE_DIR/.ip-cxl && chmod 700 \$HOME/$REMOTE_DIR/.ip-cxl; \
        rm -f \$HOME/$REMOTE_DIR/.ip-cxl/id_ed25519 \$HOME/$REMOTE_DIR/.ip-cxl/id_ed25519.pub; \
        ssh-keygen -q -t ed25519 -N '' -C '$IP_CXL_MARKER' \
          -f \$HOME/$REMOTE_DIR/.ip-cxl/id_ed25519"
    pub=$(rsh "$launcher" "cat \$HOME/$REMOTE_DIR/.ip-cxl/id_ed25519.pub")
    for h in "${IP_CXL_SEL[@]}"; do
        rsh "$h" "mkdir -p \$HOME/$REMOTE_DIR/.ip-cxl; \
            touch \$HOME/.ssh/authorized_keys && chmod 600 \$HOME/.ssh/authorized_keys; \
            grep -v ' $IP_CXL_MARKER\$' \$HOME/.ssh/authorized_keys \
              > \$HOME/$REMOTE_DIR/.ip-cxl/authorized_keys.clean || true; \
            cp \$HOME/$REMOTE_DIR/.ip-cxl/authorized_keys.clean \$HOME/.ssh/authorized_keys; \
            printf '%s\\n' '$pub' >> \$HOME/.ssh/authorized_keys"
    done
    echo "IP_CXL_LAUNCHER=$launcher"
}

ip_cxl_remove_keys() {
    local h
    for h in "${IP_CXL_SEL[@]}"; do
        rsh "$h" "if [ -f \$HOME/.ssh/authorized_keys ]; then \
            grep -v ' $IP_CXL_MARKER\$' \$HOME/.ssh/authorized_keys \
              > \$HOME/$REMOTE_DIR/authorized_keys.ip-cxl-clean || true; \
            cp \$HOME/$REMOTE_DIR/authorized_keys.ip-cxl-clean \$HOME/.ssh/authorized_keys; \
            rm -f \$HOME/$REMOTE_DIR/authorized_keys.ip-cxl-clean; fi; \
            rm -rf \$HOME/$REMOTE_DIR/.ip-cxl" || true
    done
}

# ip_cxl_ctl <up|check|keys|down> [num-hosts]
ip_cxl_ctl() {
    local action="${1:-check}" n="${2:-${#HOSTS[@]}}" h helper rc=0
    [ "$n" -ge 2 ] && [ "$n" -le "${#HOSTS[@]}" ] || {
        echo "IP-over-CXL needs 2..${#HOSTS[@]} hosts (got $n)" >&2
        return 2
    }
    IP_CXL_SEL=("${HOSTS[@]:0:$n}")

    case "$action" in
    up)
        helper=$(mktemp -d)
        write_ip_cxl_up_helper "$helper/hybrid_up.sh"
        write_ip_cxl_ssh_helper "$helper/ip_cxl_ssh.sh"
        for h in "${IP_CXL_SEL[@]}"; do
            rsh "$h" "mkdir -p \$HOME/$REMOTE_DIR"
            rcp "$helper/hybrid_up.sh" "$h" "$REMOTE_DIR/hybrid_up.sh"
            rcp "$helper/ip_cxl_ssh.sh" "$h" "$REMOTE_DIR/ip_cxl_ssh.sh"
        done
        rm -rf "$helper"
        local pids=()
        for h in "${IP_CXL_SEL[@]}"; do
            (rsh "$h" "cd \$HOME/$REMOTE_DIR && \
                env CXL_MAPOFFSET=$IP_CXL_OFFSET CXL_MAPSIZE=$IP_CXL_MAPSIZE \
                CXL_MTU=$IP_CXL_MTU bash hybrid_up.sh $n") &
            pids+=("$!")
        done
        for p in "${pids[@]}"; do wait "$p" || rc=1; done
        if [ "$rc" != 0 ]; then
            echo "IP-over-CXL bridge startup failed; inspect $REMOTE_DIR/ip-cxl-tun.log on each host." >&2
            return 1
        fi
        ip_cxl_check
        ip_cxl_install_keys
        ;;
    check) ip_cxl_check ;;
    keys)
        ip_cxl_check
        ip_cxl_install_keys
        ;;
    down)
        ip_cxl_remove_keys
        for h in "${IP_CXL_SEL[@]}"; do
            rsh "$h" "pkill -x tun || true; rm -f \$HOME/$REMOTE_DIR/tun.metrics.sock" || true
        done
        ;;
    *)
        echo "usage: ip_cxl_ctl {up|check|keys|down} [num-hosts]" >&2
        return 2
        ;;
    esac
}

# ---------------------------------------------------------------------------
# execution mode
# ---------------------------------------------------------------------------
CLUSTER=0
if [ "$MODE" = cluster ] || [ "$MODE" = auto ]; then
    cluster_config
    if [ "$MODE" = cluster ]; then
        CLUSTER=1
    elif rsh "${HOSTS[0]}" true >/dev/null 2>&1; then
        CLUSTER=1
    fi
fi
[ "$MODE" = local ] && CLUSTER=0

if [ "$CLUSTER" = 1 ]; then
    # Remote runs `cd $REMOTE_DIR` first, so binaries and data are addressed
    # relative to that directory on every host.
    NHOSTS_AVAIL=${#HOSTS[@]}
    DATA_ROOT="data"
    BIN_ROOT="."
else
    NHOSTS_AVAIL=1
    DATA_ROOT="$DATA_DIR"
    BIN_ROOT="$RS_ROOT/target/release"
    TARGET_TRIPLE=""
fi
# Per-run log directory: labels repeat across sweeps, and parse_log globs every
# host log of a label, so a shared directory would let a previous sweep's leftover
# log leak into this run's numbers.
LOG_DIR="$RESULTS_DIR/logs/$(date +%Y%m%d_%H%M%S)"
CXL_BASE_OFFSET=${CXL_BASE_OFFSET:-0x600000000}
# The largest small-profile footprint is about 384 MiB (three radix-sort
# buffers at 2^24 keys). One GiB leaves more than 2.5x allocator headroom while
# keeping the mapped region small. Eight windows are deliberately reused; CXL
# initialization must reset persistent metadata correctly between runs.
# Sizing: the direct-CXL sort needs one shared array of about `keys * 8` bytes
# plus the runtime structures -- 32 MiB at 2^22, 128 MiB at 2^24, 512 MiB at
# 2^26 and 2 GiB at 2^28. Each of the maps below leaves ample spare and no more.
# Most cases retain the small 1 GiB map. The two larger radix datasets select
# 2/8 GiB per case below; slots are spaced by the maximum so live mappings can
# never overlap. Even the largest map covers only 1/32 of the 256 GiB device.
CXL_MAPSIZE=${CXL_MAPSIZE:-1G}
CXL_MAX_MAPSIZE_BYTES=${CXL_MAX_MAPSIZE_BYTES:-8589934592}
CXL_STEP=${CXL_STEP:-0x200000000}
CXL_SLOTS=${CXL_SLOTS:-8}
IP_CXL_OFFSET=${IP_CXL_OFFSET:-0x1800000000}
IP_CXL_MAPSIZE=${IP_CXL_MAPSIZE:-1G}

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$RESULTS_DIR/bench_${STAMP}.csv"

# ---------------------------------------------------------------------------
# stage: fetch
# ---------------------------------------------------------------------------
fetch_one() { # <dataset_key>, stages into $DATA_DIR (always local to this shell)
    local key="$1" spec name url archive pattern canonical dir found
    spec=$(dataset_spec "$key") || { log "unknown dataset $key"; return 1; }
    name=$(echo "$spec" | cut -d'|' -f1)
    url=$(echo "$spec" | cut -d'|' -f2)
    archive=$(echo "$spec" | cut -d'|' -f3)
    pattern=$(echo "$spec" | cut -d'|' -f4)
    canonical=$(echo "$spec" | cut -d'|' -f5)
    dir="$DATA_DIR/$name"

    if [ -s "$dir/$canonical" ]; then
        log "[data] $name: present ($(du -h "$dir/$canonical" | cut -f1))"
        return 0
    fi

    mkdir -p "$dir/raw"
    log "[data] $name: downloading $url"
    # Large UCI/SuiteSparse transfers occasionally close early. Curl retries
    # the complete object because some UCI endpoints do not support ranges.
    if ! curl -fL --retry 8 --retry-all-errors --retry-delay 5 \
        -o "$dir/raw/archive" "$url"; then
        log "[data] $name: DOWNLOAD FAILED"
        return 1
    fi

    log "[data] $name: unpacking"
    case "$archive" in
    tar.gz) tar xzf "$dir/raw/archive" -C "$dir/raw" ;;
    # Extract only the member we need -- the UCI archives bundle raw variants
    # several times the size of the payload. The CXL hosts have no unzip, so
    # fall back to python3's zipfile, which selects members just as well.
    zip)
        if command -v unzip >/dev/null 2>&1; then
            unzip -oq "$dir/raw/archive" "$pattern" -d "$dir/raw" 2>/dev/null ||
                unzip -oq "$dir/raw/archive" -d "$dir/raw"
        else
            python3 - "$dir/raw/archive" "$dir/raw" "$pattern" <<'PY'
import fnmatch, os, sys, zipfile

archive, dest, pattern = sys.argv[1:4]
with zipfile.ZipFile(archive) as z:
    names = [n for n in z.namelist() if fnmatch.fnmatch(os.path.basename(n), pattern)]
    for name in names or z.namelist():
        z.extract(name, dest)
PY
        fi
        ;;
    gz) mv "$dir/raw/archive" "$dir/raw/payload.gz" && gunzip -f "$dir/raw/payload.gz" ;;
    compress)
        mv "$dir/raw/archive" "$dir/raw/payload.Z" &&
            gzip -dc "$dir/raw/payload.Z" >"$dir/raw/$canonical"
        ;;
    plain) mv "$dir/raw/archive" "$dir/raw/$canonical" ;;
    esac

    found=$(find "$dir/raw" -type f -name "$pattern" | sort | head -1)
    if [ -z "$found" ]; then
        log "[data] $name: no file matching '$pattern' in the archive"
        return 1
    fi
    case "$found" in
    *.gz) gunzip -c "$found" >"$dir/$canonical" ;;
    *) cp "$found" "$dir/$canonical" ;;
    esac

    # SATLIB's random-3SAT instances end with a "%" / "0" marker that is not part
    # of DIMACS and that strict parsers reject. Truncate there.
    case "$canonical" in
    *.cnf)
        awk '/^%[[:space:]]*$/ { exit } { print }' "$dir/$canonical" >"$dir/$canonical.tmp" &&
            mv "$dir/$canonical.tmp" "$dir/$canonical"
        ;;
    esac

    rm -rf "$dir/raw"
    log "[data] $name: ready ($(du -h "$dir/$canonical" | cut -f1)) -> $dir/$canonical"
}

# bnb inputs are Pisinger knapsack files that ship with the repository, so
# staging copies them out of the checkout instead of downloading. knapPI_1x
# (two-digit family) files are the hard instances; the rest are smallcoeff.
bnb_src_for() {
    case "$1" in
    knapPI_1?_*) echo "$RS_ROOT/playground/asplos2027/cxl-bnb/test_data/hardinstances_pisinger/$1.csv" ;;
    *) echo "$RS_ROOT/playground/asplos2027/cxl-bnb/test_data/smallcoeff_pisinger/$1.csv" ;;
    esac
}

stage_fetch_bnb() {
    local rc=0 key src h
    for key in $(datasets_for bnb); do
        src=$(bnb_src_for "$key")
        if [ ! -s "$src" ]; then
            log "[data] $key: MISSING in the repository ($src)"
            rc=1
            continue
        fi
        mkdir -p "$DATA_DIR/$key"
        cp -f "$src" "$DATA_DIR/$key/$key.csv"
        log "[data] $key: staged from the repository ($(du -h "$src" | cut -f1))"
    done
    if [ "$CLUSTER" = 1 ]; then
        for h in "${HOSTS[@]}"; do
            for key in $(datasets_for bnb); do
                [ -s "$DATA_DIR/$key/$key.csv" ] || continue
                rsh "$h" "mkdir -p \$HOME/$REMOTE_DIR/data/$key" &&
                    rcp "$DATA_DIR/$key/$key.csv" "$h" "$REMOTE_DIR/data/$key/$key.csv" ||
                    rc=1
            done
            log "[$h] [data] bnb staged"
        done
    fi
    return $rc
}

needed_datasets() {
    local algo ds
    for algo in $ALGOS; do
        for ds in $(datasets_for "$algo"); do
            dataset_spec "$ds" >/dev/null 2>&1 && echo "$ds"
        done
    done | sort -u
}

stage_fetch() {
    local rc=0 ds
    hr; log "== fetch (profile=$PROFILE) =="
    for ds in $(needed_datasets); do
        fetch_one "$ds" || rc=1
    done
    if echo " $ALGOS " | grep -q ' bnb '; then
        stage_fetch_bnb || rc=1
    fi
    if [ "$CLUSTER" = 1 ]; then
        log "== staging datasets on ${HOSTS[*]} =="
        # Fetch once, then copy the canonical payloads with resumable rsync.
        # This avoids three fragile multi-GiB downloads from UCI. Completed
        # destinations are skipped by size/mtime on subsequent staging runs.
        local h pids=() ds
        for h in "${HOSTS[@]}"; do
            (
                rsh "$h" "mkdir -p \$HOME/$REMOTE_DIR/data" || exit
                for ds in $(needed_datasets); do
                    rsync -az --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
                        "$DATA_DIR/$ds/" \
                        "$USER_NAME@$(fqdn "$h"):$REMOTE_DIR/data/$ds/" || exit
                    rsh "$h" "rm -rf \$HOME/$REMOTE_DIR/data/$ds/raw"
                    echo "[$h] [data] $ds: staged"
                done
            ) &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p" || rc=1; done
    fi
    return $rc
}

# ---------------------------------------------------------------------------
# stage: build (+ deploy in cluster mode)
# ---------------------------------------------------------------------------
BUILT_OK=""
BUILD_FAILED=""
BUILD_SRC_SYNCED=0

# The benchmark hosts run identical Ubuntu 24.04 / glibc 2.39 images, so one
# native build on the launcher is valid on all of them.
ensure_build_src_synced() {
    [ "$BUILD_SRC_SYNCED" = 1 ] && return 0
    rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
        --exclude target --exclude .git --exclude .context \
        "$RS_ROOT/" "$USER_NAME@$(fqdn "${HOSTS[0]}"):$REMOTE_DIR/build-src/" \
        >"$LOG_DIR/build.src-sync.log" 2>&1 || return 1
    BUILD_SRC_SYNCED=1
}

# Build one crate natively on the launcher and deploy the result everywhere.
#
# The musl allocator makes the memory-bound kernels roughly 3.8x slower than the
# same code built against glibc, and the rsmpi binaries can only be built
# natively, so every distributed variant uses the same toolchain.
build_native_and_deploy() {
    local crate="$1" builder="${HOSTS[0]}" local_bin="$LOG_DIR/$crate.native" h
    ensure_build_src_synced || return 1
    rsh "$builder" "cd \$HOME/$REMOTE_DIR/build-src && cargo build -p $crate --release" \
        >"$LOG_DIR/build.$crate.native.log" 2>&1 || return 1
    scp "${SSH_OPTS[@]}" \
        "$USER_NAME@$(fqdn "$builder"):$REMOTE_DIR/build-src/target/release/$crate" \
        "$local_bin" >>"$LOG_DIR/build.$crate.native.log" 2>&1 || return 1
    for h in "${HOSTS[@]}"; do
        rcp "$local_bin" "$h" "$REMOTE_DIR/$crate" || return 1
    done
}

stage_build() {
    hr; log "== build (cluster=$CLUSTER) =="
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

    local algo crate cargo_cmd bin
    for algo in $ALGOS; do
        crate=$(crate_for "$algo")
        [ -z "$crate" ] && continue
        # MiniSat links rsmpi unconditionally. It cannot be cross-linked on the
        # macOS build host, so cluster builds compile it natively below.
        if [ "$CLUSTER" = 1 ] && [ "$algo" = minisat ]; then
            log "[build] $crate deferred to native MPI build"
            continue
        fi
        if [ "$CLUSTER" = 1 ] && [ "$NATIVE_CXL_BUILD" = 1 ]; then
            log "[build] $crate natively on ${HOSTS[0]}"
            if build_native_and_deploy "$crate"; then
                BUILT_OK="$BUILT_OK $algo"
                NATIVE_DEPLOYED="$NATIVE_DEPLOYED $algo"
                continue
            fi
            log "[build] $crate native build FAILED (see $LOG_DIR/build.$crate.native.log); falling back to the musl cross-build"
        fi
        if [ "$CLUSTER" = 1 ]; then
            cargo_cmd=(cargo zigbuild -p "$crate" --release --target "$TARGET_TRIPLE")
        else
            cargo_cmd=(cargo build -p "$crate" --release)
        fi
        log "[build] $crate"
        if (cd "$RS_ROOT" && RUSTFLAGS="-C target-feature=+crt-static" "${cargo_cmd[@]}" >"$LOG_DIR/build.$crate.log" 2>&1); then
            BUILT_OK="$BUILT_OK $algo"
        else
            BUILD_FAILED="$BUILD_FAILED $algo"
            log "[build] $crate FAILED (see $LOG_DIR/build.$crate.log)"
        fi
    done

    if [ "$CLUSTER" = 1 ]; then
        log "== deploying binaries to ${HOSTS[*]} =="
        local h
        for algo in $BUILT_OK; do
            case " $NATIVE_DEPLOYED " in *" $algo "*) continue ;; esac
            crate=$(crate_for "$algo")
            bin="$RS_ROOT/target/$TARGET_TRIPLE/release/$crate"
            [ -f "$bin" ] || continue
            for h in "${HOSTS[@]}"; do
                rcp "$bin" "$h" "$REMOTE_DIR/$crate" || log "[deploy] $crate -> $h FAILED"
            done
        done

        # IP-over-CXL uses a long-lived TUN bridge in a dedicated 1 GiB DAX
        # window. Build and deploy it separately from the algorithm list.
        if [ "$WITH_IP_CXL" = 1 ]; then
            # The bridge moves every IP-over-CXL byte, so it is built with the
            # same toolchain as the kernels it carries.
            local tun_bin="$RS_ROOT/target/$TARGET_TRIPLE/release/tun" tun_built=0
            if [ "$NATIVE_CXL_BUILD" = 1 ]; then
                log "[build] tun (IP over CXL) natively on ${HOSTS[0]}"
                if ensure_build_src_synced &&
                    rsh "${HOSTS[0]}" "cd \$HOME/$REMOTE_DIR/build-src && cargo build -p tun --release" \
                        >"$LOG_DIR/build.tun.native.log" 2>&1 &&
                    scp "${SSH_OPTS[@]}" \
                        "$USER_NAME@$(fqdn "${HOSTS[0]}"):$REMOTE_DIR/build-src/target/release/tun" \
                        "$LOG_DIR/tun.native" >>"$LOG_DIR/build.tun.native.log" 2>&1; then
                    tun_bin="$LOG_DIR/tun.native"
                    tun_built=1
                else
                    log "[build] tun native build FAILED (see $LOG_DIR/build.tun.native.log); falling back to the musl cross-build"
                fi
            fi
            if [ "$tun_built" != 1 ]; then
                log "[build] tun (IP over CXL)"
                (cd "$RS_ROOT" && RUSTFLAGS="-C target-feature=+crt-static" \
                    cargo zigbuild -p tun --release --target "$TARGET_TRIPLE" \
                    >"$LOG_DIR/build.tun.log" 2>&1) && tun_built=1
            fi
            if [ "$tun_built" = 1 ]; then
                for h in "${HOSTS[@]}"; do
                    rcp "$tun_bin" "$h" "$REMOTE_DIR/tun" ||
                        log "[deploy] tun -> $h FAILED"
                    rsh "$h" "test -e \$HOME/tun.pre-ip-cxl || cp \$HOME/tun \$HOME/tun.pre-ip-cxl" || true
                    rcp "$tun_bin" "$h" "tun" ||
                        log "[deploy] sudo-enabled tun -> $h FAILED"
                done
            else
                log "[build] tun FAILED (see $LOG_DIR/build.tun.log)"
                BUILD_FAILED="$BUILD_FAILED ip_cxl"
            fi
        fi

        # rsmpi discovers and links the target MPI installation at build time,
        # so it cannot be part of the macOS -> Linux zig cross-build. Build the
        # dedicated radix-sort MPI binary natively on the launcher host, fetch
        # it once, and deploy the same executable to every benchmark host.
        if [ "$WITH_MPI" = 1 ] && echo " $ALGOS " | grep -q ' sort '; then
            local builder="${HOSTS[0]}" mpi_local="$LOG_DIR/cxl-sort-mpi"
            log "[build] cxl-sort-mpi natively on $builder"
            if rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
                    --exclude target --exclude .git --exclude .context \
                    "$RS_ROOT/" "$USER_NAME@$(fqdn "$builder"):$REMOTE_DIR/build-src/" \
                    >"$LOG_DIR/build.cxl-sort-mpi.sync.log" 2>&1 &&
                rsh "$builder" "cd \$HOME/$REMOTE_DIR/build-src && \
                    cargo build -p cxl-sort --release --features mpi --bin cxl-sort-mpi" \
                    >"$LOG_DIR/build.cxl-sort-mpi.log" 2>&1 &&
                scp "${SSH_OPTS[@]}" \
                    "$USER_NAME@$(fqdn "$builder"):$REMOTE_DIR/build-src/target/release/cxl-sort-mpi" \
                    "$mpi_local"; then
                for h in "${HOSTS[@]}"; do
                    rcp "$mpi_local" "$h" "$REMOTE_DIR/cxl-sort-mpi" ||
                        log "[deploy] cxl-sort-mpi -> $h FAILED"
                done
            else
                log "[build] cxl-sort-mpi FAILED (see $LOG_DIR/build.cxl-sort-mpi.log)"
            fi
        fi
        if [ "$WITH_MPI" = 1 ] && echo " $ALGOS " | grep -q ' bfs '; then
            local builder="${HOSTS[0]}" mpi_local="$LOG_DIR/cxl-bfs-mpi"
            log "[build] cxl-bfs-mpi natively on $builder"
            if rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
                    --exclude target --exclude .git --exclude .context \
                    "$RS_ROOT/" "$USER_NAME@$(fqdn "$builder"):$REMOTE_DIR/build-src/" \
                    >"$LOG_DIR/build.cxl-bfs-mpi.sync.log" 2>&1 &&
                rsh "$builder" "cd \$HOME/$REMOTE_DIR/build-src && \
                    cargo build -p cxl-bfs --release --features mpi --bin cxl-bfs-mpi" \
                    >"$LOG_DIR/build.cxl-bfs-mpi.log" 2>&1 &&
                scp "${SSH_OPTS[@]}" \
                    "$USER_NAME@$(fqdn "$builder"):$REMOTE_DIR/build-src/target/release/cxl-bfs-mpi" \
                    "$mpi_local"; then
                for h in "${HOSTS[@]}"; do
                    rcp "$mpi_local" "$h" "$REMOTE_DIR/cxl-bfs-mpi" ||
                        log "[deploy] cxl-bfs-mpi -> $h FAILED"
                done
            else
                log "[build] cxl-bfs-mpi FAILED (see $LOG_DIR/build.cxl-bfs-mpi.log)"
            fi
        fi
        if [ "$WITH_MPI" = 1 ] && echo " $ALGOS " | grep -q ' minisat '; then
            local builder="${HOSTS[0]}" mpi_local="$LOG_DIR/cxl-minisat"
            log "[build] cxl-minisat natively on $builder"
            if rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
                    --exclude target --exclude .git --exclude .context \
                    "$RS_ROOT/" "$USER_NAME@$(fqdn "$builder"):$REMOTE_DIR/build-src/" \
                    >"$LOG_DIR/build.cxl-minisat.sync.log" 2>&1 &&
                rsh "$builder" "cd \$HOME/$REMOTE_DIR/build-src && \
                    cargo build -p cxl-minisat --release --bin cxl-minisat" \
                    >"$LOG_DIR/build.cxl-minisat.log" 2>&1 &&
                scp "${SSH_OPTS[@]}" \
                    "$USER_NAME@$(fqdn "$builder"):$REMOTE_DIR/build-src/target/release/cxl-minisat" \
                    "$mpi_local"; then
                BUILT_OK="$BUILT_OK minisat"
                for h in "${HOSTS[@]}"; do
                    rcp "$mpi_local" "$h" "$REMOTE_DIR/cxl-minisat" ||
                        log "[deploy] cxl-minisat -> $h FAILED"
                done
            else
                BUILD_FAILED="$BUILD_FAILED minisat"
                log "[build] cxl-minisat FAILED (see $LOG_DIR/build.cxl-minisat.log)"
            fi
        fi
    elif [ "$WITH_MPI" = 1 ] && echo " $ALGOS " | grep -q ' sort '; then
        # A local MPI run needs the feature-gated binary in addition to the
        # default non-MPI cxl-sort binary built above.
        log "[build] cxl-sort-mpi locally"
        if ! (cd "$RS_ROOT" && cargo build -p cxl-sort --release --features mpi \
                --bin cxl-sort-mpi >"$LOG_DIR/build.cxl-sort-mpi.log" 2>&1); then
            log "[build] cxl-sort-mpi FAILED (see $LOG_DIR/build.cxl-sort-mpi.log)"
        fi
    fi
    if [ "$CLUSTER" != 1 ] && [ "$WITH_MPI" = 1 ] && echo " $ALGOS " | grep -q ' bfs '; then
        log "[build] cxl-bfs-mpi locally"
        if ! (cd "$RS_ROOT" && cargo build -p cxl-bfs --release --features mpi \
                --bin cxl-bfs-mpi >"$LOG_DIR/build.cxl-bfs-mpi.log" 2>&1); then
            log "[build] cxl-bfs-mpi FAILED (see $LOG_DIR/build.cxl-bfs-mpi.log)"
        fi
    fi
    [ -n "$BUILD_FAILED" ] && log "[build] failed:$BUILD_FAILED (those algorithms are recorded as skipped)"
    return 0
}

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
csv_init() {
    printf '%s\n' \
        'run_id,started_at,algorithm,variant,class,dataset,dataset_kind,dataset_detail,hosts,threads,repeat,wall_s,kernel_s,load_s,cxl_load_s,cxl_store_s,wait_spins,status,validation,exit_code,log,cmd' \
        >"$CSV"
}

csv_row() {
    local out="" f
    for f in "$@"; do
        f=${f//\"/\"\"}
        out="$out\"$f\","
    done
    printf '%s\n' "${out%,}" >>"$CSV"
}

# parse_log <file...> -> "kernel_s|load_s|cxl_load_s|cxl_store_s|wait_spins|validation"
#
# Takes every host log of a run, not just the first: CXL machine IDs are handed
# out in discovery order, so the rank that prints the validation verdict is not
# necessarily HOSTS[0]. Kernel/load timings are the max across hosts, matching
# the slowest-host semantics of wall_s. CXL counters and spin counts are summed
# across processes so `aggregate_runs.py` can derive a per-host mean.
parse_log() {
    awk '
    FNR == 1 { locked = 0 }   # `locked` is per-file
    function keepmax(cur, new) { return (new != "" && (cur == "" || new + 0 > cur + 0)) ? new : cur }

    function dur(t,   v, u) {
        if (!match(t, /^[0-9]+(\.[0-9]+)?/)) return "";
        v = substr(t, RSTART, RLENGTH); u = substr(t, RSTART + RLENGTH);
        if (u ~ /^ns/)              return v / 1e9;
        if (u ~ /^(us|\xc2\xb5s)/)  return v / 1e6;
        if (u ~ /^ms/)              return v / 1e3;
        if (u ~ /^s/)               return v + 0;
        return "";
    }
    function grab(line, marker,   rest, tok) {
        rest = substr(line, index(line, marker) + length(marker));
        sub(/^[ \t]+/, "", rest);
        tok = rest; sub(/[ \t].*$/, "", tok);
        return dur(tok);
    }
    {
        # Authoritative: every crate emits this around the algorithm alone (no
        # load, no CXL attach, no collect, no validation, no printing). It wins
        # over every heuristic below.
        if ($0 ~ /^BENCH_KERNEL_NS=/) {
            split($0, kv, "=");
            exact = keepmax(exact, kv[2] / 1e9);
            next;
        }
        if ($0 ~ /^BENCH_LOAD_NS=/) {
            split($0, kv, "=");
            exact_load = keepmax(exact_load, kv[2] / 1e9);
            next;
        }
        if ($0 ~ /^BENCH_CXL_LOAD_NS=/) {
            split($0, kv, "=");
            cxl_load += kv[2] / 1e9;
            next;
        }
        if ($0 ~ /^BENCH_CXL_STORE_NS=/) {
            split($0, kv, "=");
            cxl_store += kv[2] / 1e9;
            next;
        }
        if ($0 ~ /^BENCH_WAIT_SPINS=/) {
            split($0, kv, "=");
            wait_spins += kv[2];
            next;
        }
        if (index($0, "Loaded ") && index($0, " in "))  load = keepmax(load, grab($0, " in "));
        # "compute:" is the kernel time of the distributed matmul/stencil runs
        # and beats the surrounding "elapsed:" total, so it wins if present.
        if (index($0, "compute:"))                     { kern = keepmax(kern, grab($0, "compute:")); locked = 1 }
        else if (!locked && index($0, "finished in "))   kern = keepmax(kern, grab($0, "finished in "));
        else if (!locked && index($0, " iters in "))     kern = keepmax(kern, grab($0, " iters in "));
        else if (!locked && index($0, " keys in "))      kern = keepmax(kern, grab($0, " keys in "));
        else if (!locked && index($0, "elapsed:"))       kern = keepmax(kern, grab($0, "elapsed:"));
        else if (!locked && $0 ~ /^\[[a-z_-]+\] [0-9]/)  kern = keepmax(kern, grab($0, "] "));
        # minisat logs "[<rank>] <elapsed> <event>" timestamps; the last one is
        # the total, so let it overwrite.
        if ($0 ~ /^\[[0-9]+\] [0-9].*finished[ \t]*$/) { kern = grab($0, "] "); locked = 1 }
        if ($0 ~ /FAIL/)              valid = "fail";
        else if (valid == "" && $0 ~ /matches sequential baseline/) valid = "ok";
        if ($0 ~ /^(SATISFIABLE|UNSATISFIABLE|s SATISFIABLE|s UNSATISFIABLE)/) valid = "ok";
    }
    END {
        k = (exact != "") ? exact : kern;
        printf "%s|%s|%.9f|%.9f|%.0f|%s", (k == "" ? "" : k), \
            (exact_load != "" ? exact_load : (load == "" ? "" : load)), \
            cxl_load + 0, cxl_store + 0, wait_spins + 0, valid
    }
    ' "$@" 2>/dev/null
}

# ---------------------------------------------------------------------------
# launching
# ---------------------------------------------------------------------------
RUN_ID=0
OFFSET_IDX=0
NEXT_OFFSET=""

# Rotate through a small pool of CXL windows.  The pool is intentionally reused
# once exhausted, so every algorithm must initialize or invalidate its own
# state instead of depending on zeroed DAX memory.
next_offset() {
    NEXT_OFFSET=$(printf "0x%x" $((CXL_BASE_OFFSET + (OFFSET_IDX % CXL_SLOTS) * CXL_STEP)))
    OFFSET_IDX=$((OFFSET_IDX + 1))
}

size_to_bytes() {
    case "$1" in
    *G | *g) echo $(( ${1%[Gg]} * 1024 * 1024 * 1024 )) ;;
    *M | *m) echo $(( ${1%[Mm]} * 1024 * 1024 )) ;;
    *K | *k) echo $(( ${1%[Kk]} * 1024 )) ;;
    *) echo $(( $1 )) ;;
    esac
}

# Return a conservative mapping for each benchmark case. Radix sort uses one
# segmented exchange array, but the larger inputs retain 2/8 GiB mappings to
# leave allocator-word and metadata headroom. Other extended cases stay below
# the normal 1 GiB map.
cxl_mapsize_for_case() {
    local algo="$1" dataset="$2" required=$((1024 * 1024 * 1024)) configured
    case "$algo:$dataset" in
    sort:keys-2p26) required=$((2 * 1024 * 1024 * 1024)) ;;
    sort:keys-2p28) required=$((8 * 1024 * 1024 * 1024)) ;;
    esac
    configured=$(size_to_bytes "$CXL_MAPSIZE")
    [ "$configured" -gt "$required" ] && required=$configured
    echo "$required"
}

# launch_local <log> <cmd...> -> echoes "<wall_ns> <rc>"
launch_local() {
    local logf="$1"; shift
    local t0 t1 rc
    t0=$(now_ns)
    # ${a[@]+...} keeps `set -u` happy when no timeout binary was found
    ${TIMEOUT_CMD[@]+"${TIMEOUT_CMD[@]}"} "$@" >"$logf" 2>&1
    rc=$?
    t1=$(now_ns)
    echo "$((t1 - t0)) $rc"
}

# launch_cluster <label> <nhosts> <offset> <mapsize> <cmdline>
# -> echoes "<wall_ns> <rc>"
# Every host runs the same command line against the same CXL window; the run's
# wall time is the slowest host, which is what an end-to-end number has to be.
launch_cluster() {
    local label="$1" n="$2" offset="$3" mapsize="$4" cmdline="$5"
    local sel=("${HOSTS[@]:0:$n}") h pids=() rc=0 wall=0 hw

    for h in "${sel[@]}"; do
        (
            rsh "$h" "cd $REMOTE_DIR && export CXL_MAPOFFSET=$offset CXL_MAPSIZE=$mapsize; \
                t0=\$(date +%s%N); timeout $RUN_TIMEOUT $cmdline; rc=\$?; t1=\$(date +%s%N); \
                echo \"BENCH_RC=\$rc\"; echo \"BENCH_WALL_NS=\$((t1-t0))\"" \
                >"$LOG_DIR/${label}.${h}.log" 2>&1
        ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done

    for h in "${sel[@]}"; do
        local f="$LOG_DIR/${label}.${h}.log"
        local hrc
        hrc=$(grep -m1 '^BENCH_RC=' "$f" 2>/dev/null | cut -d= -f2)
        hw=$(grep -m1 '^BENCH_WALL_NS=' "$f" 2>/dev/null | cut -d= -f2)
        [ -z "$hrc" ] && hrc=255
        [ -n "$hw" ] && [ "$hw" -gt "$wall" ] 2>/dev/null && wall=$hw
        [ "$hrc" != 0 ] && rc=$hrc
    done
    echo "$wall $rc"
}

# launch_ip_cxl <label> <ranks> <cmdline> -> echoes "<wall_ns> <rc>"
# Open MPI uses SSH only to start remote ranks; both its control sockets and
# payload transport are pinned to the CXL-backed TUN interface. Each rank also
# controls its local bridge's timestamp.rs counters exactly at kernel entry and
# exit; the bridge snapshot is printed by the rank as normal BENCH_CXL fields.
launch_ip_cxl() {
    local label="$1" ranks="$2" cmdline="$3"
    local launcher hostlist="localhost" i check_output logf t0 t1 rc collective_opts=""
    check_output=$(ip_cxl_ctl check "$NHOSTS_AVAIL" 2>/dev/null)
    launcher=$(echo "$check_output" | awk -F= '/^IP_CXL_LAUNCHER=/{print $2; exit}')
    [ -n "$launcher" ] || { log "FATAL: IP-over-CXL TUN is not ready"; echo "0 255"; return; }
    for i in $(seq 2 "$ranks"); do hostlist="$hostlist,10.0.0.$i"; done
    # Open MPI's default small-message collective selection regresses sharply
    # at three ranks on the high-diameter BFS input. The ring Allgather and
    # basic-linear Alltoallv keep the 3+-host TUN path stable and were neutral
    # or faster on the larger BFS datasets. Leave two-rank and non-BFS cases
    # unchanged so this is a scoped, measured transport choice.
    if [ "$ranks" -ge 3 ] && echo "$label" | grep -q '^bfs\.'; then
        collective_opts="--mca coll_tuned_use_dynamic_rules 1 \
          --mca coll_tuned_allgather_algorithm 4 \
          --mca coll_tuned_alltoallv_algorithm 1"
    fi
    logf="$LOG_DIR/${label}.${launcher}.log"
    t0=$(now_ns)
    rsh "$launcher" "cd \$HOME/$REMOTE_DIR && \
        export IP_CXL_KEY=\$HOME/$REMOTE_DIR/.ip-cxl/id_ed25519 \
               BENCH_IP_CXL=1 \
               IP_CXL_METRICS_SOCKET=\$HOME/$REMOTE_DIR/tun.metrics.sock; \
        timeout $RUN_TIMEOUT mpirun -np $ranks -H $hostlist --map-by ppr:1:node \
          --bind-to none --mca plm_rsh_agent \$HOME/$REMOTE_DIR/ip_cxl_ssh.sh \
          --mca oob_tcp_if_include cxl --mca pml ob1 \
          --mca btl self,tcp --mca btl_tcp_if_include cxl \
          $collective_opts \
          -x BENCH_IP_CXL -x IP_CXL_METRICS_SOCKET \
          --wdir \$HOME/$REMOTE_DIR $cmdline" >"$logf" 2>&1
    rc=$?
    t1=$(now_ns)
    echo "$((t1 - t0)) $rc"
}

MPI_SSH_MARKER="cxl-paper-bench-mpi"

setup_mpi_ssh() {
    local launcher="${HOSTS[0]}" pub h
    rsh "$launcher" "mkdir -p \$HOME/$REMOTE_DIR/.mpi-bench && chmod 700 \$HOME/$REMOTE_DIR/.mpi-bench; \
        rm -f \$HOME/$REMOTE_DIR/.mpi-bench/id_ed25519 \$HOME/$REMOTE_DIR/.mpi-bench/id_ed25519.pub; \
        ssh-keygen -q -t ed25519 -N '' -C '$MPI_SSH_MARKER' \
          -f \$HOME/$REMOTE_DIR/.mpi-bench/id_ed25519" || return 1
    pub=$(rsh "$launcher" "cat \$HOME/$REMOTE_DIR/.mpi-bench/id_ed25519.pub") || return 1
    for h in "${HOSTS[@]}"; do
        rsh "$h" "touch \$HOME/.ssh/authorized_keys && chmod 600 \$HOME/.ssh/authorized_keys; \
            grep -v ' $MPI_SSH_MARKER\$' \$HOME/.ssh/authorized_keys \
              > \$HOME/$REMOTE_DIR/authorized_keys.mpi-clean || true; \
            cp \$HOME/$REMOTE_DIR/authorized_keys.mpi-clean \$HOME/.ssh/authorized_keys; \
            rm -f \$HOME/$REMOTE_DIR/authorized_keys.mpi-clean; \
            printf '%s\\n' '$pub' >> \$HOME/.ssh/authorized_keys" || return 1
    done
    local helper_dir
    helper_dir=$(mktemp -d)
    write_ip_cxl_ssh_helper "$helper_dir/ip_cxl_ssh.sh"
    rcp "$helper_dir/ip_cxl_ssh.sh" "$launcher" "$REMOTE_DIR/ip_cxl_ssh.sh" || {
        rm -rf "$helper_dir"
        return 1
    }
    rm -rf "$helper_dir"
    rsh "$launcher" "chmod +x \$HOME/$REMOTE_DIR/ip_cxl_ssh.sh"
}

cleanup_mpi_ssh() {
    local h
    for h in "${HOSTS[@]}"; do
        rsh "$h" "if [ -f \$HOME/.ssh/authorized_keys ]; then \
            grep -v ' $MPI_SSH_MARKER\$' \$HOME/.ssh/authorized_keys \
              > \$HOME/$REMOTE_DIR/authorized_keys.mpi-clean || true; \
            cp \$HOME/$REMOTE_DIR/authorized_keys.mpi-clean \$HOME/.ssh/authorized_keys; \
            rm -f \$HOME/$REMOTE_DIR/authorized_keys.mpi-clean; fi" || true
    done
    rsh "${HOSTS[0]}" "rm -rf \$HOME/$REMOTE_DIR/.mpi-bench" || true
}

dax_devices_healthy() {
    local h
    for h in "${HOSTS[@]}"; do
        rsh "$h" 'test -c /dev/dax0.0 && size=$(cat /sys/dev/char/250:*/size 2>/dev/null | head -1) && test "${size:-0}" -gt 0' \
            >/dev/null 2>&1 || return 1
    done
}

dax_failure_in_logs() {
    local label="$1"
    grep -Eiq 'SIGBUS|Bus error|mmap[^:]*:.*(fail|error)|DAX[^:]*:.*(fail|error)' \
        "$LOG_DIR/${label}."*.log 2>/dev/null
}

cxl_allocation_failure_in_logs() {
    local label="$1"
    grep -Eiq 'AllocError|out of memory|memory allocation of .* failed|allocation[^:]*:.*(fail|error)|current version cant alloc' \
        "$LOG_DIR/${label}."*.log 2>/dev/null
}

# run_case <algo> <variant> <class> <dataset> <ds_kind> <ds_detail> <hosts> <bin> <args...>
run_case() {
    local algo="$1" variant="$2" class="$3" ds="$4" dskind="$5" dsdetail="$6" hosts="$7" bin="$8"
    shift 8
    local args="$*" case_mapsize
    case_mapsize=$(cxl_mapsize_for_case "$algo" "$ds")

    # skips -------------------------------------------------------------
    # `dsdetail` is the dataset path for file-backed datasets, so a missing
    # download is a plain existence check.
    local skip="${SKIP_REASON:-}"
    case " $BUILD_FAILED " in *" $algo "*) skip="skipped_build" ;; esac
    if [ -z "$skip" ] && [ "$hosts" -ge 2 ] && [ "$CLUSTER" != 1 ]; then skip="skipped_no_cxl"; fi
    if [ -z "$skip" ] && [ "$hosts" -gt "$NHOSTS_AVAIL" ]; then skip="skipped_hosts"; fi
    if [ -z "$skip" ] && [ "$dskind" = file ] && [ "$CLUSTER" != 1 ] && [ ! -s "$dsdetail" ]; then
        skip="skipped_no_data"
    fi

    local rep
    for rep in $(seq 0 $((REPEATS - 1))); do
        RUN_ID=$((RUN_ID + 1))
        local label="${algo}.${variant}.${ds}.h${hosts}.r${rep}"
        local started; started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        local cmd="$bin $args"

        if [ "$PLAN_ONLY" = 1 ]; then
            log "[plan] $label :: ${skip:+($skip) }$cmd"
            continue
        fi

        if [ -n "$skip" ]; then
            csv_row "$RUN_ID" "$started" "$algo" "$variant" "$class" "$ds" "$dskind" "$dsdetail" \
                "$hosts" "$THREADS" "$rep" "" "" "" "0" "0" "0" "$skip" "" "" "" "$cmd"
            continue
        fi

        log "[run ] $label"
        local res wall_ns rc logf kernel load cxl_load cxl_store wait_spins valid parsed
        if [ "$class" = ip_cxl ]; then
            logf="$LOG_DIR/${label}.${IP_CXL_LAUNCHER:-ip-cxl}.log"
            res=$(launch_ip_cxl "$label" "$MPI_RANKS" "$bin $args")
            # launch_ip_cxl resolves the actual machine-id-1 host; parse every
            # matching log rather than depending on launch order.
            logf=$(find "$LOG_DIR" -maxdepth 1 -name "${label}.*.log" -print -quit)
        elif [ "$class" = mpi ]; then
            # MPI is launched once, by the launcher, not once per host. `hosts`
            # is the machine count; MPI_RANKS is the rank count, which differs
            # when the ranks are packed onto a single machine.
            if [ "$CLUSTER" = 1 ]; then
                logf="$LOG_DIR/${label}.${HOSTS[0]}.log"
                # Open MPI's default binding policy depends on the rank count:
                # up to two ranks it binds each to a single core, above that to
                # a socket. A threaded kernel then sees one core and runs its
                # whole compute serially, which shows up as a transport
                # difference it is not. Every launch therefore states its
                # placement explicitly. Packed ranks share one machine and get
                # one core each; one rank per node gets that node's cores.
                local mpi_where="" mpi_agent="" mpi_collectives=""
                local mpi_mapping="--map-by core --bind-to core"
                if [ "$hosts" -ge 2 ]; then
                    # Open MPI shortens hostnames for its ssh launch, and the
                    # short names do not resolve here, so address peers by IP.
                    local hostlist="" h ip
                    for h in "${HOSTS[@]:0:$hosts}"; do
                        ip=$(getent hosts "$(fqdn "$h")" 2>/dev/null | awk '{print $1; exit}')
                        if [ -z "$ip" ] && command -v dig >/dev/null 2>&1; then
                            ip=$(dig +short "$(fqdn "$h")" 2>/dev/null | awk 'NF {print; exit}')
                        fi
                        hostlist="$hostlist,${ip:-$(fqdn "$h")}"
                    done
                    mpi_where="-H ${hostlist#,}"
                    mpi_agent="--mca plm_rsh_agent \$HOME/$REMOTE_DIR/ip_cxl_ssh.sh"
                    mpi_mapping="--map-by ppr:1:node --bind-to none"
                    if [ "$algo" = bfs ]; then
                        if [ "$MPI_RANKS" -ge 3 ]; then
                            mpi_collectives="--mca coll_tuned_use_dynamic_rules 1 \
                              --mca coll_tuned_allgather_algorithm 4 \
                              --mca coll_tuned_alltoallv_algorithm 1"
                        fi
                    fi
                fi
                NEXT_OFFSET=$(printf "0x%x" "$CXL_BASE_OFFSET")
                res=$(launch_cluster "$label" 1 "$NEXT_OFFSET" "$case_mapsize" \
                    "env IP_CXL_KEY=\$HOME/$REMOTE_DIR/.mpi-bench/id_ed25519 \
                     mpirun -np $MPI_RANKS $mpi_where $mpi_mapping $mpi_collectives $mpi_agent $bin $args")
            else
                logf="$LOG_DIR/${label}.local.log"
                # shellcheck disable=SC2086
                res=$(launch_local "$logf" mpirun -np "$MPI_RANKS" "$bin" $args)
            fi
        elif [ "$CLUSTER" = 1 ]; then
            # hosts 0/1 run on the first host only; hosts >= 2 run everywhere,
            # all mapping the same CXL window.
            local n=$hosts
            [ "$n" -lt 1 ] && n=1
            logf="$LOG_DIR/${label}.${HOSTS[0]}.log"
            if [ "$class" = cxl ]; then
                next_offset
            else
                NEXT_OFFSET=$(printf "0x%x" "$CXL_BASE_OFFSET")
            fi
            res=$(launch_cluster "$label" "$n" "$NEXT_OFFSET" "$case_mapsize" "$bin $args")
        else
            logf="$LOG_DIR/${label}.local.log"
            # shellcheck disable=SC2086
            res=$(launch_local "$logf" "$bin" $args)
        fi
        wall_ns=$(echo "$res" | cut -d' ' -f1)
        rc=$(echo "$res" | cut -d' ' -f2)

        parsed=$(parse_log "$LOG_DIR/${label}."*.log)
        kernel=$(echo "$parsed" | cut -d'|' -f1)
        load=$(echo "$parsed" | cut -d'|' -f2)
        cxl_load=$(echo "$parsed" | cut -d'|' -f3)
        cxl_store=$(echo "$parsed" | cut -d'|' -f4)
        wait_spins=$(echo "$parsed" | cut -d'|' -f5)
        valid=$(echo "$parsed" | cut -d'|' -f6)

        local wall_s status
        wall_s=$(awk -v n="$wall_ns" 'BEGIN { printf "%.6f", n / 1e9 }')
        if [ "$rc" = 0 ]; then status=ok
        elif [ "$rc" = 124 ]; then status=timeout
        else status=error; fi
        [ "$valid" = fail ] && status=mismatch

        csv_row "$RUN_ID" "$started" "$algo" "$variant" "$class" "$ds" "$dskind" "$dsdetail" \
            "$hosts" "$THREADS" "$rep" "$wall_s" "$kernel" "$load" "$cxl_load" "$cxl_store" \
            "$wait_spins" "$status" "$valid" "$rc" \
            "${logf#"$RESULTS_DIR"/}" "$cmd"
        log "       -> ${status} wall=${wall_s}s kernel=${kernel:-n/a}s"
        if [ "$STOP_ON_DAX_FAILURE" = 1 ] && [ "$CLUSTER" = 1 ]; then
            if ! dax_devices_healthy; then
                log "FATAL: a benchmark host lost its DAX device; stopping immediately for operator repair"
                exit 86
            fi
            if [ "$rc" != 0 ] && dax_failure_in_logs "$label"; then
                log "FATAL: $label reported a DAX/SIGBUS failure; stopping immediately for operator repair"
                exit 86
            fi
            if [ "$rc" != 0 ] && cxl_allocation_failure_in_logs "$label"; then
                log "FATAL: $label exhausted its configured CXL allocation; stopping before another case touches DAX"
                log "Check cxl_mapsize_for_case and allocator-safe segmentation for this dataset."
                exit 87
            fi
        fi
    done
}

thread_flag() { [ "$THREADS" != 0 ] && echo "--threads $THREADS"; }

# ---------------------------------------------------------------------------
# per-algorithm benchmark plans
#
# Variant naming (kept identical across kernels so plots line up):
#   local            host-local threaded radix sort
#   cxl              distributed radix sort over CXL shared memory
#   mpi_npN          distributed radix sort over rsmpi with N ranks
#   seq/threads      baselines used by the non-sort kernels
#   cxl_shared       shared data structure in CXL memory, N hosts
#   cxl_partitioned  host-local topology, CXL only for the exchange, N hosts
#   pregel           message passing over the CXL IPC rings, N hosts
#   cxl_ipc          k-means' IPC engine (its equivalent of pregel)
#   cxl_exchange     stencil's halo exchange over the IPC rings (grid stays
#                    host-local) -- its equivalent of pregel
#   ip_cxl_npN       rsmpi/TCP with one rank per host over CXL-backed TUN
#   mpi / net        off-CXL baselines (rsmpi, TCP)
# ---------------------------------------------------------------------------
bench_graph() { # <algo> <extra args...>
    local algo="$1"; shift
    local extra="$*"
    local bin="$BIN_ROOT/$(crate_for "$algo")"
    local mpi_bin="$BIN_ROOT/cxl-bfs-mpi"
    local ds path n ranks machines
    for ds in $(datasets_for "$algo"); do
        path=$(dataset_path "$ds")
        local base="--mtx $path $extra $(thread_flag)"
        if [ "$algo" != bfs ] || [ "$BFS_MPI_ONLY" != 1 ]; then
            run_case "$algo" seq single "$ds" file "$path" 0 "$bin" --num-hosts 0 $base
            run_case "$algo" threads single "$ds" file "$path" 1 "$bin" --num-hosts 1 $base
            for n in $HOST_COUNTS; do
                run_case "$algo" cxl_shared cxl "$ds" file "$path" "$n" "$bin" --num-hosts "$n" $base
                if [ "$algo" != bfs ]; then
                    run_case "$algo" pregel cxl "$ds" file "$path" "$n" "$bin" --num-hosts "$n" --pregel $base
                fi
                if [ "$algo" = bfs ]; then
                    run_case "$algo" cxl_partitioned cxl "$ds" file "$path" "$n" "$bin" \
                        --num-hosts "$n" --partitioned $base
                fi
            done
        fi
        if [ "$algo" = bfs ] && [ "$WITH_MPI" = 1 ]; then
            for ranks in 1 $HOST_COUNTS; do
                MPI_RANKS=$ranks
                if [ "${MPI_CROSS_HOST:-0}" = 1 ] && [ "$ranks" -le "$NHOSTS_AVAIL" ]; then
                    machines=$ranks
                else
                    machines=1
                fi
                run_case bfs "pregel_np$ranks" mpi "$ds" file "$path" "$machines" \
                    "$mpi_bin" --mpi $base
            done
        fi
    done
}

bench_sort() {
    local bin="$BIN_ROOT/$(crate_for sort)"
    local mpi_bin="$BIN_ROOT/cxl-sort-mpi"
    local ds keys n ranks machines
    for ds in $(datasets_for sort); do
        keys=$(generated_params "$ds")
        local base="--nodes $keys $(thread_flag)"
        run_case sort local single "$ds" generated "keys=$keys" 1 "$bin" --num-hosts 1 $base
        for n in $HOST_COUNTS; do
            run_case sort cxl cxl "$ds" generated "keys=$keys" "$n" "$bin" \
                --num-hosts "$n" $base
        done
        if [ "$WITH_MPI" = 1 ]; then
            for ranks in $HOST_COUNTS; do
                MPI_RANKS=$ranks
                if [ "${MPI_CROSS_HOST:-0}" = 1 ] && [ "$ranks" -le "$NHOSTS_AVAIL" ]; then
                    machines=$ranks
                else
                    machines=1
                fi
                run_case sort "mpi_np$ranks" mpi "$ds" generated "keys=$keys ranks=$ranks" \
                    "$machines" "$mpi_bin" --mpi $base
            done
        fi
    done
}

bench_kmeans() {
    local bin="$BIN_ROOT/$(crate_for kmeans)"
    local ds path args n ranks machines
    for ds in $(datasets_for kmeans); do
        path=$(dataset_path "$ds")
        args=$(kmeans_args "$ds" "$path")
        run_case kmeans single single "$ds" file "$path" 1 "$bin" $args
        for n in $HOST_COUNTS; do
            run_case kmeans cxl_shared cxl "$ds" file "$path" "$n" "$bin" \
                $args --cxl-hosts "$n" --cxl-mode shared
            run_case kmeans cxl_ipc cxl "$ds" file "$path" "$n" "$bin" \
                $args --cxl-hosts "$n" --cxl-mode ipc
        done
        if [ "$WITH_MPI" = 1 ]; then
            for ranks in $HOST_COUNTS; do
                MPI_RANKS=$ranks
                if [ "${MPI_CROSS_HOST:-0}" = 1 ] && [ "$ranks" -le "$NHOSTS_AVAIL" ]; then
                    machines=$ranks
                else
                    machines=1
                fi
                run_case kmeans "mpi_np$ranks" mpi "$ds" file "$path" "$machines" "$bin" \
                    $args --mpi --expect-ranks "$ranks"
            done
        fi
    done
}

bench_matmul() {
    local bin="$BIN_ROOT/$(crate_for matmul)"
    local ds nn n tflag ranks machines
    tflag=$([ "$THREADS" != 0 ] && echo "$THREADS")
    for ds in $(datasets_for matmul); do
        nn=$(generated_params "$ds")
        local dims="$nn $nn $nn"
        run_case matmul seq single "$ds" generated "n=$nn" 0 "$bin" serial $dims
        run_case matmul threads single "$ds" generated "n=$nn" 1 "$bin" threads $dims $tflag
        for n in $HOST_COUNTS; do
            run_case matmul cxl_shared cxl "$ds" generated "n=$nn" "$n" "$bin" dist $dims "$n"
            run_case matmul cxl_opt_dist cxl "$ds" generated "n=$nn" "$n" "$bin" \
                dist-opt $dims "$n"
            run_case matmul cxl_ipc cxl "$ds" generated "n=$nn" "$n" "$bin" \
                ipc-dist $dims "$n"
        done
        if [ "$WITH_MPI" = 1 ]; then
            for ranks in $HOST_COUNTS; do
                MPI_RANKS=$ranks
                if [ "${MPI_CROSS_HOST:-0}" = 1 ] && [ "$ranks" -le "$NHOSTS_AVAIL" ]; then
                    machines=$ranks
                else
                    machines=1
                fi
                run_case matmul "mpi_np$ranks" mpi "$ds" generated "n=$nn ranks=$ranks" \
                    "$machines" "$bin" mpi-dist $dims "$ranks"
            done
        fi
    done
}

bench_stencil() {
    local bin="$BIN_ROOT/$(crate_for stencil)"
    local ds dims n tflag ranks machines
    tflag=$([ "$THREADS" != 0 ] && echo "$THREADS")
    for ds in $(datasets_for stencil); do
        dims=$(generated_params "$ds")
        run_case stencil seq single "$ds" generated "grid=$dims" 0 "$bin" serial $dims
        run_case stencil threads single "$ds" generated "grid=$dims" 1 "$bin" threads $dims $tflag
        for n in $HOST_COUNTS; do
            run_case stencil cxl_shared cxl "$ds" generated "grid=$dims" "$n" "$bin" dist $dims "$n"
            run_case stencil cxl_opt_dist cxl "$ds" generated "grid=$dims" "$n" "$bin" \
                dist-opt $dims "$n"
            run_case stencil cxl_opt2_dist cxl "$ds" generated "grid=$dims" "$n" "$bin" \
                dist-opt-opt $dims "$n"
            run_case stencil cxl_exchange cxl "$ds" generated "grid=$dims" "$n" "$bin" \
                dist-exchange $dims "$n"
        done
        if [ "$WITH_MPI" = 1 ]; then
            for ranks in $HOST_COUNTS; do
                MPI_RANKS=$ranks
                if [ "${MPI_CROSS_HOST:-0}" = 1 ] && [ "$ranks" -le "$NHOSTS_AVAIL" ]; then
                    machines=$ranks
                else
                    machines=1
                fi
                run_case stencil "mpi_np$ranks" mpi "$ds" generated "grid=$dims ranks=$ranks" \
                    "$machines" "$bin" mpi-exchange $dims "$ranks"
            done
        fi
    done
}

bench_bnb() {
    # Branch-and-bound knapsack. One binary carries every fabric behind
    # --fabric: local (no transport), MPI (rsmpi collectives, one rank per
    # host), CXL (blocking shared-memory reduction) and CXLv2 (latency-hiding
    # reduction). The CXL variants read the host count from CXL_NUM_HOSTS.
    local bin="$BIN_ROOT/$(crate_for bnb)"
    local ds path n ranks machines
    local base_params="--steps-until-global-reduction $BNB_STEPS --reductions-until-termination-check $BNB_TERM"
    local v2_params="--steps-until-global-reduction $BNB_V2_STEPS --reductions-until-termination-check $BNB_V2_TERM"
    for ds in $(datasets_for bnb); do
        path="$DATA_ROOT/$ds/$ds.csv"
        run_case bnb local single "$ds" file "$path" 1 "$bin" \
            --input-file "$path" $base_params
        for n in $HOST_COUNTS; do
            run_case bnb cxl cxl "$ds" file "$path" "$n" \
                "env CXL_NUM_HOSTS=$n $bin" --fabric CXL --input-file "$path" $base_params
            run_case bnb cxlv2 cxl "$ds" file "$path" "$n" \
                "env CXL_NUM_HOSTS=$n $bin" --fabric CXLv2 --input-file "$path" $v2_params
        done
        if [ "$WITH_MPI" = 1 ]; then
            for ranks in $HOST_COUNTS; do
                MPI_RANKS=$ranks
                if [ "${MPI_CROSS_HOST:-0}" = 1 ] && [ "$ranks" -le "$NHOSTS_AVAIL" ]; then
                    machines=$ranks
                else
                    machines=1
                fi
                run_case bnb "mpi_np$ranks" mpi "$ds" file "$path" "$machines" "$bin" \
                    --fabric MPI --input-file "$path" $base_params
            done
        fi
    done
}

bench_minisat() {
    # cxl-minisat is MPI-only today (cube-and-conquer over rank-indexed
    # assumptions), so its variants are rank counts under mpirun.
    local bin="$BIN_ROOT/$(crate_for minisat)"
    local ds path n
    if ! command -v mpirun >/dev/null 2>&1 && [ "$CLUSTER" != 1 ]; then
        log "[skip] minisat: no mpirun on this machine"
        BUILD_FAILED="$BUILD_FAILED minisat"
    fi
    # MPI_CROSS_HOST=0 packs all ranks onto one machine. With
    # MPI_CROSS_HOST=1, stage_run installs and later removes a marked ephemeral
    # key so Open MPI can start one rank on each selected benchmark host.
    for ds in $(datasets_for minisat); do
        path=$(dataset_path "$ds")
        local ranks machines
        for ranks in 1 $HOST_COUNTS; do
            MPI_RANKS=$ranks
            if [ "${MPI_CROSS_HOST:-0}" = 1 ] && [ "$ranks" -le "$NHOSTS_AVAIL" ]; then
                machines=$ranks
            else
                machines=1
            fi
            run_case minisat "mpi_np$ranks" mpi "$ds" file "$path" "$machines" "$bin" \
                --core "$path"
        done
    done
}

# Run network-capable MPI implementations through Linux TUN interfaces backed
# by the CXL IPC rings. These are kept in a final phase because the bridge has
# busy receive workers and would otherwise perturb the direct-CXL measurements.
bench_ip_cxl() {
    local algo ds path keys n bin base
    for algo in $ALGOS; do
        case "$algo" in
        bfs)
            bin="$BIN_ROOT/cxl-bfs-mpi"
            for ds in $(datasets_for bfs); do
                path=$(dataset_path "$ds")
                base="--mpi --mtx $path --source 0 $(thread_flag)"
                for n in $HOST_COUNTS; do
                    [ "$n" -lt 2 ] && continue
                    MPI_RANKS=$n
                    run_case bfs "ip_cxl_np$n" ip_cxl "$ds" file "$path" "$n" "$bin" $base
                done
            done
            ;;
        sort)
            bin="$BIN_ROOT/cxl-sort-mpi"
            for ds in $(datasets_for sort); do
                keys=$(generated_params "$ds")
                base="--mpi --nodes $keys $(thread_flag)"
                for n in $HOST_COUNTS; do
                    [ "$n" -lt 2 ] && continue
                    MPI_RANKS=$n
                    run_case sort "ip_cxl_np$n" ip_cxl "$ds" generated \
                        "keys=$keys ranks=$n transport=tun" "$n" "$bin" $base
                done
            done
            ;;
        minisat)
            bin="$BIN_ROOT/cxl-minisat"
            for ds in $(datasets_for minisat); do
                path=$(dataset_path "$ds")
                for n in $HOST_COUNTS; do
                    [ "$n" -lt 2 ] && continue
                    MPI_RANKS=$n
                    run_case minisat "ip_cxl_np$n" ip_cxl "$ds" file "$path" "$n" "$bin" \
                        --core "$path"
                done
            done
            ;;
        kmeans)
            bin="$BIN_ROOT/$(crate_for kmeans)"
            for ds in $(datasets_for kmeans); do
                path=$(dataset_path "$ds")
                base=$(kmeans_args "$ds" "$path")
                for n in $HOST_COUNTS; do
                    [ "$n" -lt 2 ] && continue
                    MPI_RANKS=$n
                    run_case kmeans "ip_cxl_np$n" ip_cxl "$ds" file "$path" "$n" "$bin" \
                        $base --mpi --expect-ranks "$n"
                done
            done
            ;;
        stencil)
            bin="$BIN_ROOT/$(crate_for stencil)"
            for ds in $(datasets_for stencil); do
                dims=$(generated_params "$ds")
                for n in $HOST_COUNTS; do
                    [ "$n" -lt 2 ] && continue
                    MPI_RANKS=$n
                    run_case stencil "ip_cxl_np$n" ip_cxl "$ds" generated \
                        "grid=$dims ranks=$n transport=tun" "$n" "$bin" mpi-exchange $dims "$n"
                done
            done
            ;;
        matmul)
            bin="$BIN_ROOT/$(crate_for matmul)"
            for ds in $(datasets_for matmul); do
                nn=$(generated_params "$ds")
                for n in $HOST_COUNTS; do
                    [ "$n" -lt 2 ] && continue
                    MPI_RANKS=$n
                    run_case matmul "ip_cxl_np$n" ip_cxl "$ds" generated \
                        "n=$nn ranks=$n transport=tun" "$n" "$bin" mpi-dist $nn $nn $nn "$n"
                done
            done
            ;;
        bnb)
            bin="$BIN_ROOT/$(crate_for bnb)"
            for ds in $(datasets_for bnb); do
                path="$DATA_ROOT/$ds/$ds.csv"
                for n in $HOST_COUNTS; do
                    [ "$n" -lt 2 ] && continue
                    MPI_RANKS=$n
                    run_case bnb "ip_cxl_np$n" ip_cxl "$ds" file "$path" "$n" "$bin" \
                        --fabric MPI --input-file "$path" \
                        --steps-until-global-reduction "$BNB_STEPS" \
                        --reductions-until-termination-check "$BNB_TERM"
                done
            done
            ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# stage: run
# ---------------------------------------------------------------------------
stage_run() {
    hr
    log "== run =="
    log "   mode=$([ "$CLUSTER" = 1 ] && echo cluster || echo local) profile=$PROFILE repeats=$REPEATS"
    log "   hosts=$([ "$CLUSTER" = 1 ] && echo "${HOSTS[*]}" || echo localhost) host_counts=$HOST_COUNTS"
    log "   csv=$CSV"
    # The CXL window must fit the device: mapping past the end only shows up as
    # a SIGBUS once the run touches the tail, which looks like an algorithm bug.
    if [ "$CLUSTER" = 1 ] && [ -n "$HOST_COUNTS" ]; then
        local devsz top max_hosts=0 n h max_map step_bytes ip_size
        for n in $HOST_COUNTS; do
            [ "$n" -gt "$max_hosts" ] && max_hosts=$n
        done
        max_map=$CXL_MAX_MAPSIZE_BYTES
        n=$(size_to_bytes "$CXL_MAPSIZE")
        [ "$n" -gt "$max_map" ] && max_map=$n
        step_bytes=$((CXL_STEP))
        if [ "$step_bytes" -lt "$max_map" ]; then
            log "FATAL: CXL_STEP ($step_bytes bytes) is smaller than the largest per-case mapping ($max_map bytes)."
            log "Increase CXL_STEP so benchmark windows cannot overlap."
            exit 1
        fi
        top=$((CXL_BASE_OFFSET + (CXL_SLOTS - 1) * step_bytes + max_map))
        if [ "$WITH_IP_CXL" = 1 ]; then
            ip_size=$(size_to_bytes "$IP_CXL_MAPSIZE")
            local ip_top=$((IP_CXL_OFFSET + ip_size))
            [ "$ip_top" -gt "$top" ] && top=$ip_top
        fi
        for h in "${HOSTS[@]:0:$max_hosts}"; do
            devsz=$(rsh "$h" \
                'test -c /dev/dax0.0 && cat /sys/dev/char/*/size 2>/dev/null | head -1' \
                2>/dev/null)
            if [ -z "$devsz" ]; then
                log "FATAL: /dev/dax0.0 is unavailable on $h; restore the DAX device before running CXL variants."
                exit 1
            fi
            if [ "$top" -gt "$devsz" ]; then
                log "FATAL: CXL window runs past the device on $h."
                log "  base $(printf 0x%x "$CXL_BASE_OFFSET") + $CXL_SLOTS slots spaced by $(printf 0x%x "$CXL_STEP"), max map $max_map bytes"
                log "  = $((top / 1024 / 1024 / 1024)) GiB, device is $((devsz / 1024 / 1024 / 1024)) GiB."
                log "  Lower CXL_BASE_OFFSET or CXL_SLOTS."
                exit 1
            fi
        done
    fi

    csv_init

    local algo
    local mpi_ssh_ready=0
    if [ "$PLAN_ONLY" != 1 ] && [ "$CLUSTER" = 1 ] && [ "$WITH_MPI" = 1 ] && \
            [ "${MPI_CROSS_HOST:-0}" = 1 ] && [ "$IP_CXL_ONLY" != 1 ]; then
        log "== configuring ephemeral cross-host MPI credentials =="
        if ! setup_mpi_ssh; then
            cleanup_mpi_ssh
            log "FATAL: could not configure cross-host MPI SSH"
            exit 1
        fi
        mpi_ssh_ready=1
        trap 'cleanup_mpi_ssh' EXIT INT TERM
    fi
    if [ "$IP_CXL_ONLY" != 1 ]; then
        for algo in $ALGOS; do
            hr; log "== $algo =="
            case "$algo" in
            bfs) bench_graph bfs --source 0 ;;
            sssp) bench_graph sssp --source 0 ;;
            pagerank) bench_graph pagerank --iters "${PR_ITERS:-20}" ;;
            triangle) bench_graph triangle ;;
            sort) bench_sort ;;
            kmeans) bench_kmeans ;;
            matmul) bench_matmul ;;
            stencil) bench_stencil ;;
            minisat) [ "$WITH_MPI" = 1 ] && bench_minisat || log "[skip] minisat (set WITH_MPI=1)" ;;
            bnb) bench_bnb ;;
            *) log "unknown algorithm: $algo" ;;
            esac
        done
    fi
    if [ "$mpi_ssh_ready" = 1 ]; then
        cleanup_mpi_ssh
        trap - EXIT INT TERM
    fi

    if [ "$WITH_IP_CXL" = 1 ]; then
        hr; log "== IP over CXL (TUN + MPI/TCP) =="
        if [ "$PLAN_ONLY" = 1 ]; then
            bench_ip_cxl
        elif [ "$CLUSTER" != 1 ]; then
            log "FATAL: IP-over-CXL requires cluster mode"
            exit 1
        else
            if ! IP_CXL_OFFSET="$IP_CXL_OFFSET" IP_CXL_MAPSIZE="$IP_CXL_MAPSIZE" \
                    ip_cxl_ctl up "$NHOSTS_AVAIL"; then
                log "FATAL: could not configure IP-over-CXL TUN interfaces"
                log "Run IP_CXL_INTERACTIVE=1 scripts/paper_bench.sh ip-cxl up once, then rerun."
                exit 1
            fi
            trap 'ip_cxl_ctl down "$NHOSTS_AVAIL" >/dev/null 2>&1 || true' \
                EXIT INT TERM
            bench_ip_cxl
            ip_cxl_ctl down "$NHOSTS_AVAIL" >/dev/null 2>&1 || true
            trap - EXIT INT TERM
        fi
    fi

    [ "$PLAN_ONLY" = 1 ] && return 0
    cp "$CSV" "$RESULTS_DIR/bench.csv"
    hr
    log "== done: $CSV (copied to $RESULTS_DIR/bench.csv) =="
    summary
}

summary() {
    awk -F'","' 'NR > 1 {
        gsub(/^"|"$/, "", $0);
        key = $3 "/" $4 "/" $6 "/h" $9;
        if ($18 == "ok") { sum[key] += $12; cnt[key]++ } else { bad[key] = bad[key] " " $18 }
    }
    END {
        printf "%-34s %8s %8s %s\n", "algorithm/variant/dataset/hosts", "runs", "mean_s", "status";
        for (k in cnt) printf "%-34s %8d %8.3f %s\n", k, cnt[k], sum[k] / cnt[k], "ok";
        for (k in bad) if (!(k in cnt)) printf "%-34s %8d %8s %s\n", k, 0, "-", bad[k];
    }' "$CSV" | sort >&2
}

# ---------------------------------------------------------------------------
PLAN_ONLY=0
case "$STAGE" in
fetch) stage_fetch ;;
build) stage_build ;;
run) stage_run ;;
csv) CSV="$RESULTS_DIR/bench.csv"; summary ;;
plan) PLAN_ONLY=1; stage_run ;;
all) stage_fetch; stage_build; stage_run ;;
ip-cxl)
    # Manual control of the IP-over-CXL bridge. `run` brings it up and tears it
    # down on its own; this is for bringing it up once interactively when the
    # sudo rule needs a password, and for inspecting or cleaning up afterwards.
    cluster_config
    ip_cxl_ctl "${2:-check}" "${3:-${#HOSTS[@]}}"
    ;;
*)
    log "usage: paper_bench.sh [all|fetch|build|run|plan|csv|ip-cxl {up|check|keys|down} [hosts]]"
    exit 1
    ;;
esac
