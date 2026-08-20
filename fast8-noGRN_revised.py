import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from multiprocessing import Pool, freeze_support, cpu_count

import time
import os

from numba import njit


# ============================================================
# Globals
# ============================================================

_GLOBAL_INDPTR = None
_GLOBAL_INDICES = None
_GLOBAL_INDPTR_T = None
_GLOBAL_INDICES_T = None
_NUM_NODES = 0

# Reusable Stage-2 workspace (one copy per worker process).
# This avoids allocating/initializing an O(n) visited array for every node.
_GLOBAL_RS_VISITED = None
_GLOBAL_RS_QUEUE = None
_GLOBAL_RS_TOKEN = 0

# RCC bounds are exact integer quantities.  Use an integer sentinel for +infinity
# rather than float32 so that pruning can never be affected by rounding.
RCC_INF = np.iinfo(np.int64).max // 4


def init_worker(
    indptr,
    indices,
    indptr_t,
    indices_t,
    num_nodes
):
    global _GLOBAL_INDPTR
    global _GLOBAL_INDICES
    global _GLOBAL_INDPTR_T
    global _GLOBAL_INDICES_T
    global _NUM_NODES
    global _GLOBAL_RS_VISITED
    global _GLOBAL_RS_QUEUE
    global _GLOBAL_RS_TOKEN

    _GLOBAL_INDPTR = indptr
    _GLOBAL_INDICES = indices

    _GLOBAL_INDPTR_T = indptr_t
    _GLOBAL_INDICES_T = indices_t

    _NUM_NODES = num_nodes

    # Stage-2 arrays are allocated lazily, only in workers that actually
    # execute local-expansion tasks.
    _GLOBAL_RS_VISITED = None
    _GLOBAL_RS_QUEUE = None
    _GLOBAL_RS_TOKEN = 0


# ============================================================
# Fast directed BFS
# ============================================================

@njit(fastmath=True)
def numba_fast_bfs(
    indptr,
    indices,
    num_nodes,
    source_node
):
    """
    Exact directed BFS.

    dists[v] = shortest directed distance source -> v
    dists[v] = -1 if v is unreachable.

    Returns
    -------
    dists
    sum_dists
        Sum of all finite distances excluding the source.
    reachable_count
        Number of reachable nodes excluding source.
    max_ecc
        Reachable eccentricity of source.
    """

    dists = np.full(
        num_nodes,
        -1,
        dtype=np.int32
    )

    queue = np.empty(
        num_nodes,
        dtype=np.int32
    )

    head = 0
    tail = 0

    queue[tail] = source_node
    tail += 1

    dists[source_node] = 0

    max_ecc = 0
    reachable_count = 0
    sum_dists = np.int64(0)

    while head < tail:

        curr = queue[head]
        head += 1

        curr_d = dists[curr]

        start_idx = indptr[curr]
        end_idx = indptr[curr + 1]

        for p in range(
            start_idx,
            end_idx
        ):

            nb = indices[p]

            if dists[nb] == -1:

                d = curr_d + 1

                dists[nb] = d

                queue[tail] = nb
                tail += 1

                sum_dists += np.int64(d)
                reachable_count += 1

                if d > max_ecc:
                    max_ecc = d

    return (
        dists,
        sum_dists,
        reachable_count,
        max_ecc
    )


# ============================================================
# RS_k
# ============================================================

@njit(fastmath=True)
def numba_calc_rs_k(
    indptr,
    indices,
    node_idx,
    k,
    visited_marker,
    queue,
    visit_token
):
    """
    RS_k(u): number of nodes reached within k directed BFS layers.

    The caller supplies a reusable visited-marker array and queue.  A unique
    integer visit_token identifies the current BFS, so the function touches
    only nodes/edges actually explored and does not initialize an O(n) array
    for every source node.

    still_growing=False means the BFS frontier was exhausted before depth k,
    hence epsilon(u) < k.
    """

    head = 0
    tail = 0

    queue[tail] = node_idx
    tail += 1
    visited_marker[node_idx] = visit_token

    layer_end = tail
    current_step = 0
    still_growing = True

    while (
        head < tail
        and
        current_step < k
    ):

        curr = queue[head]
        head += 1

        start_idx = indptr[curr]
        end_idx = indptr[curr + 1]

        for p in range(
            start_idx,
            end_idx
        ):

            nb = indices[p]

            if visited_marker[nb] != visit_token:

                visited_marker[nb] = visit_token

                queue[tail] = nb
                tail += 1

        if head == layer_end:

            if tail == layer_end:

                still_growing = False
                break

            layer_end = tail
            current_step += 1

    return tail, still_growing


# ============================================================
# SCC-DAG forward closure Q_s
# ============================================================

@njit
def scc_forward_closure(
    dag_indptr,
    dag_indices,
    num_scc,
    source_scc
):
    """
    Q_s = SCCs reachable from sentinel SCC,
    including the sentinel SCC itself.
    """

    reachable = np.zeros(
        num_scc,
        dtype=np.uint8
    )

    queue = np.empty(
        num_scc,
        dtype=np.int32
    )

    head = 0
    tail = 0

    reachable[source_scc] = 1

    queue[tail] = source_scc
    tail += 1

    while head < tail:

        c = queue[head]
        head += 1

        for p in range(
            dag_indptr[c],
            dag_indptr[c + 1]
        ):

            d = dag_indices[p]

            if reachable[d] == 0:

                reachable[d] = 1

                queue[tail] = d
                tail += 1

    return reachable


# ============================================================
# Rigorous SCC escape bound
# ============================================================

@njit
def compute_escape_bound(
    dag_indptr,
    dag_indices,
    topo_order,
    scc_sizes,
    in_Q
):
    """
    For SCC C outside Q_s:

        E_s(C)
        =
        (|C|-1)
        +
        max(
            0,
            max_{C->D, D outside Q_s}
            [1 + E_s(D)]
        )

    This is a rigorous upper bound on distances to targets
    reachable from C but lying outside the sentinel's
    reachable SCC closure Q_s.
    """

    num_scc = len(
        scc_sizes
    )

    E = np.zeros(
        num_scc,
        dtype=np.int64
    )

    for pos in range(
        len(topo_order) - 1,
        -1,
        -1
    ):

        c = topo_order[pos]

        if in_Q[c] == 1:

            E[c] = 0
            continue

        best = np.int64(0)

        for p in range(
            dag_indptr[c],
            dag_indptr[c + 1]
        ):

            d = dag_indices[p]

            # Once the path enters Q_s,
            # the sentinel triangle bound takes over.
            if in_Q[d] == 1:
                continue

            val = (
                np.int64(1)
                +
                E[d]
            )

            if val > best:
                best = val

        E[c] = (
            np.int64(
                scc_sizes[c] - 1
            )
            +
            best
        )

    return E


# ============================================================
# Workers
# ============================================================

def worker_sssp_directed(
    node_idx
):
    """
    Every sampled node is now a sentinel candidate.

    No GRN > 20% criterion is used.
    """

    global _GLOBAL_INDPTR
    global _GLOBAL_INDICES
    global _GLOBAL_INDPTR_T
    global _GLOBAL_INDICES_T
    global _NUM_NODES

    # --------------------------------------------------------
    # Forward exact BFS
    # --------------------------------------------------------

    (
        _,
        ssspl,
        ssspn,
        ecc_out_v
    ) = numba_fast_bfs(
        _GLOBAL_INDPTR,
        _GLOBAL_INDICES,
        _NUM_NODES,
        node_idx
    )

    # --------------------------------------------------------
    # Reverse BFS
    #
    # On transpose:
    # d_to_s[u] = d_G(u, sentinel)
    # --------------------------------------------------------

    (
        d_to_s,
        _,
        _,
        _
    ) = numba_fast_bfs(
        _GLOBAL_INDPTR_T,
        _GLOBAL_INDICES_T,
        _NUM_NODES,
        node_idx
    )

    # Keep exact integer distances.  -1 denotes unreachable.
    # Do not convert these certificate inputs to float32.
    return (
        node_idx,
        ssspl,
        ssspn,
        ecc_out_v,
        d_to_s
    )


def worker_calc_rs(
    args
):
    node_idx, k = args

    global _GLOBAL_INDPTR
    global _GLOBAL_INDICES
    global _NUM_NODES
    global _GLOBAL_RS_VISITED
    global _GLOBAL_RS_QUEUE
    global _GLOBAL_RS_TOKEN

    # Allocate O(n) workspace only once per Stage-2 worker process.
    if _GLOBAL_RS_VISITED is None:
        _GLOBAL_RS_VISITED = np.zeros(
            _NUM_NODES,
            dtype=np.int32
        )
        _GLOBAL_RS_QUEUE = np.empty(
            _NUM_NODES,
            dtype=np.int32
        )
        _GLOBAL_RS_TOKEN = 1
    else:
        _GLOBAL_RS_TOKEN += 1

        # Defensive overflow handling.  In the present experiments each worker
        # processes far fewer than 2^31 local BFS tasks, so this branch should
        # never be reached, but it keeps the marker scheme exact in general.
        if _GLOBAL_RS_TOKEN >= np.iinfo(np.int32).max:
            _GLOBAL_RS_VISITED.fill(0)
            _GLOBAL_RS_TOKEN = 1

    rs_val, still_growing = (
        numba_calc_rs_k(
            _GLOBAL_INDPTR,
            _GLOBAL_INDICES,
            node_idx,
            k,
            _GLOBAL_RS_VISITED,
            _GLOBAL_RS_QUEUE,
            np.int32(_GLOBAL_RS_TOKEN)
        )
    )

    return (
        node_idx,
        rs_val,
        bool(still_growing)
    )


def worker_final_check(
    cand_idx
):
    global _GLOBAL_INDPTR
    global _GLOBAL_INDICES
    global _NUM_NODES

    (
        _,
        _,
        _,
        ecc
    ) = numba_fast_bfs(
        _GLOBAL_INDPTR,
        _GLOBAL_INDICES,
        _NUM_NODES,
        cand_idx
    )

    return (
        cand_idx,
        ecc
    )


# ============================================================
# Build SCC condensation DAG
# ============================================================

def build_condensation_dag(
    adj,
    scc_labels,
    num_scc
):
    num_nodes = adj.shape[0]

    src_nodes = np.repeat(
        np.arange(
            num_nodes,
            dtype=np.int32
        ),
        np.diff(
            adj.indptr
        )
    )

    dst_nodes = (
        adj.indices
        .astype(
            np.int32,
            copy=False
        )
    )

    src_scc = (
        scc_labels[
            src_nodes
        ]
    )

    dst_scc = (
        scc_labels[
            dst_nodes
        ]
    )

    mask = (
        src_scc
        !=
        dst_scc
    )

    dag_src = (
        src_scc[
            mask
        ]
    )

    dag_dst = (
        dst_scc[
            mask
        ]
    )

    dag = csr_matrix(
        (
            np.ones(
                len(dag_src),
                dtype=np.uint8
            ),
            (
                dag_src,
                dag_dst
            )
        ),
        shape=(
            num_scc,
            num_scc
        )
    )

    dag.sum_duplicates()

    dag.data[:] = 1

    dag.eliminate_zeros()

    del src_nodes
    del src_scc
    del dst_scc
    del mask
    del dag_src
    del dag_dst

    return dag


# ============================================================
# Topological ordering of SCC-DAG
# ============================================================

def topological_order(
    dag
):
    num_scc = (
        dag.shape[0]
    )

    indegree = np.asarray(
        dag.sum(axis=0)
    ).ravel().astype(
        np.int64
    )

    queue = np.empty(
        num_scc,
        dtype=np.int32
    )

    topo = np.empty(
        num_scc,
        dtype=np.int32
    )

    head = 0
    tail = 0

    for c in range(
        num_scc
    ):

        if indegree[c] == 0:

            queue[tail] = c
            tail += 1

    count = 0

    while head < tail:

        c = queue[head]
        head += 1

        topo[count] = c
        count += 1

        for p in range(
            dag.indptr[c],
            dag.indptr[c + 1]
        ):

            d = dag.indices[p]

            indegree[d] -= 1

            if indegree[d] == 0:

                queue[tail] = d
                tail += 1

    if count != num_scc:

        raise RuntimeError(
            "Condensation graph is not a DAG."
        )

    return topo


# ============================================================
# Run one network
# ============================================================

def run_single_network(
    file_name,
    largest_wcc_only=False
):
    file_path = os.path.join(
        ".",
        file_name
    )

    if not os.path.exists(
        file_path
    ):

        print(
            f"[错误] 文件不存在: "
            f"{file_path}"
        )

        return None

    TARGET_SAMPLES = 500

    total_start = time.time()

    print(
        "\n"
        +
        "=" * 80
    )

    run_label = (
        f"{file_name} [largest WCC]"
        if largest_wcc_only
        else file_name
    )

    print(
        f"RCC EXACT TEST WITHOUT 20% GRN: "
        f"{run_label}"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # Load graph
    # ========================================================

    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        header=None,
        names=[
            "src",
            "dst"
        ],
        comment="#"
    )

    unique_nodes = np.unique(
        df[
            [
                "src",
                "dst"
            ]
        ].values
    )

    full_num_nodes = len(
        unique_nodes
    )

    raw_edge_rows = len(
        df
    )

    mapping = {
        node: i
        for i, node
        in enumerate(
            unique_nodes
        )
    }

    src_idx = (
        df["src"]
        .map(mapping)
        .to_numpy()
    )

    dst_idx = (
        df["dst"]
        .map(mapping)
        .to_numpy()
    )

    adj = csr_matrix(
        (
            np.ones(
                raw_edge_rows,
                dtype=bool
            ),
            (
                src_idx,
                dst_idx
            )
        ),
        shape=(
            full_num_nodes,
            full_num_nodes
        )
    )

    # The algorithm runs on the deduplicated CSR graph, so M must be adj.nnz,
    # not the number of raw rows in the input file.
    adj.sum_duplicates()
    adj.eliminate_zeros()

    full_unique_edges = int(
        adj.nnz
    )

    graph_scope = "Full graph"
    weak_component_count = None
    largest_wcc_full_size = None

    if largest_wcc_only:

        (
            weak_component_count,
            weak_labels
        ) = connected_components(
            adj,
            directed=True,
            connection="weak",
            return_labels=True
        )

        weak_sizes = np.bincount(
            weak_labels,
            minlength=weak_component_count
        )

        largest_wcc_label = int(
            np.argmax(
                weak_sizes
            )
        )

        largest_wcc_nodes = np.flatnonzero(
            weak_labels == largest_wcc_label
        )

        largest_wcc_full_size = int(
            largest_wcc_nodes.size
        )

        adj = (
            adj[
                largest_wcc_nodes,
                :
            ][
                :,
                largest_wcc_nodes
            ]
            .tocsr()
        )

        # Preserve canonical CSR form after subgraph extraction.
        adj.sum_duplicates()
        adj.eliminate_zeros()

        graph_scope = "Largest WCC"

        del weak_labels
        del weak_sizes
        del largest_wcc_nodes

    num_nodes = int(
        adj.shape[0]
    )

    # Effective number of directed edges actually processed by the algorithm.
    num_edges = int(
        adj.nnz
    )

    adj_t = (
        adj
        .transpose()
        .tocsr()
    )

    print(
        f"Scope = {graph_scope}"
    )

    if largest_wcc_only:
        print(
            f"Original graph nodes = {full_num_nodes}"
        )
        print(
            f"Original unique directed edges = {full_unique_edges}"
        )
        print(
            f"Weakly connected components = {weak_component_count}"
        )
        print(
            f"Largest WCC nodes = {largest_wcc_full_size}"
        )
    else:
        print(
            f"Raw input edge rows = {raw_edge_rows}"
        )

    print(
        f"Nodes = {num_nodes}"
    )

    print(
        f"Unique directed edges processed = {num_edges}"
    )

    # ========================================================
    # SCC decomposition
    # ========================================================

    scc_start = time.time()

    (
        num_scc,
        scc_labels
    ) = connected_components(
        adj,
        directed=True,
        connection="strong",
        return_labels=True
    )

    scc_labels = (
        scc_labels.astype(
            np.int32,
            copy=False
        )
    )

    scc_sizes = np.bincount(
        scc_labels,
        minlength=num_scc
    ).astype(
        np.int32
    )

    largest_scc = int(
        np.max(
            scc_sizes
        )
    )

    print(
        f"SCC count = "
        f"{num_scc}"
    )

    print(
        f"Largest SCC = "
        f"{largest_scc} "
        f"("
        f"{largest_scc / num_nodes * 100:.2f}%"
        f")"
    )

    # ========================================================
    # Condensation DAG
    # ========================================================

    dag = (
        build_condensation_dag(
            adj,
            scc_labels,
            num_scc
        )
    )

    topo = (
        topological_order(
            dag
        )
    )

    print(
        f"SCC DAG edges = "
        f"{dag.nnz}"
    )

    print(
        f"SCC preprocessing = "
        f"{time.time() - scc_start:.2f} sec"
    )

    # ========================================================
    # CSR arrays
    # ========================================================

    indptr = (
        adj.indptr
        .astype(
            np.int32
        )
    )

    indices = (
        adj.indices
        .astype(
            np.int32
        )
    )

    indptr_t = (
        adj_t.indptr
        .astype(
            np.int32
        )
    )

    indices_t = (
        adj_t.indices
        .astype(
            np.int32
        )
    )

    dag_indptr = (
        dag.indptr
        .astype(
            np.int32
        )
    )

    dag_indices = (
        dag.indices
        .astype(
            np.int32
        )
    )

    # ========================================================
    # Numba warm-up
    # ========================================================

    numba_fast_bfs(
        indptr,
        indices,
        num_nodes,
        0
    )

    warm_rs_visited = np.zeros(
        num_nodes,
        dtype=np.int32
    )
    warm_rs_queue = np.empty(
        num_nodes,
        dtype=np.int32
    )

    numba_calc_rs_k(
        indptr,
        indices,
        0,
        1,
        warm_rs_visited,
        warm_rs_queue,
        np.int32(1)
    )

    del warm_rs_visited
    del warm_rs_queue

    dummy_Q = (
        scc_forward_closure(
            dag_indptr,
            dag_indices,
            num_scc,
            0
        )
    )

    compute_escape_bound(
        dag_indptr,
        dag_indices,
        topo,
        scc_sizes,
        dummy_Q
    )

    max_workers = min(
        cpu_count(),
        6
    )

    # ========================================================
    # Stage 1
    #
    # NO GRN THRESHOLD
    # ========================================================

    print(
        "\nStage 1: "
        "500-node calibration + RCC certification "
        "(NO 20% GRN threshold)"
    )

    # Fixed random seed for reproducibility
    rng = np.random.default_rng(2026)
    sample_indices = rng.choice(
        num_nodes,
        min(
            TARGET_SAMPLES,
            num_nodes
        ),
        replace=False
    )

    with Pool(
        processes=max_workers,
        initializer=init_worker,
        initargs=(
            indptr,
            indices,
            indptr_t,
            indices_t,
            num_nodes
        )
    ) as pool:

        results = pool.map(
            worker_sssp_directed,
            sample_indices
        )

    # ========================================================
    # ALL sampled nodes participate
    # ========================================================

    sum_ssspl = 0.0
    sum_ssspn = 0.0

    initial_lb = 0

    # Rigorous RCC UB.  Exact integer storage prevents floating-point
    # roundoff from ever affecting a pruning decision.
    rcc_ub = np.full(
        num_nodes,
        RCC_INF,
        dtype=np.int64
    )

    same_scc_ever = np.zeros(
        num_nodes,
        dtype=bool
    )

    escape_rcc_ever = np.zeros(
        num_nodes,
        dtype=bool
    )

    sample_count = len(
        results
    )

    samples_with_reachability = 0

    certificate_start = time.time()

    for (
        sentinel,
        ssspl,
        ssspn,
        ecc_s,
        d_to_s
    ) in results:

        # ====================================================
        # 1. Scale estimation:
        # ALL sampled finite reachable pairs EXCLUDING self-pairs.
        # numba_fast_bfs returns ssspn excluding the source itself.
        # ====================================================

        if ssspn > 0:

            sum_ssspl += float(
                ssspl
            )

            sum_ssspn += float(
                ssspn
            )

            samples_with_reachability += 1

        # ====================================================
        # 2. Initial LB:
        # ALL sampled exact eccentricities
        # ====================================================

        if ecc_s > initial_lb:

            initial_lb = (
                ecc_s
            )

        # ====================================================
        # 3. RCC certification:
        # ALL sampled nodes may act as sentinels.
        #
        # Whether a sentinel is useful is decided ONLY by RCC.
        # ====================================================

        Cs = (
            scc_labels[
                sentinel
            ]
        )

        # ----------------------------------------------------
        # Q_s
        # ----------------------------------------------------

        in_Q = (
            scc_forward_closure(
                dag_indptr,
                dag_indices,
                num_scc,
                Cs
            )
        )

        # ----------------------------------------------------
        # Escape bound E_s
        # ----------------------------------------------------

        escape_scc = (
            compute_escape_bound(
                dag_indptr,
                dag_indices,
                topo,
                scc_sizes,
                in_Q
            )
        )

        # ----------------------------------------------------
        # Sentinel score
        # ----------------------------------------------------

        finite = (
            d_to_s
            >=
            0
        )

        # Exact integer certificate value B_s(u) = d(u,s) + epsilon(s).
        # Entries with d_to_s == -1 are masked out by `finite` below.
        score = d_to_s.astype(
            np.int64,
            copy=True
        )
        score[finite] += np.int64(
            ecc_s
        )

        node_scc = (
            scc_labels
        )

        same = (
            node_scc
            ==
            Cs
        )

        outside_Q = (
            in_Q[
                node_scc
            ]
            ==
            0
        )

        escape_node = (
            escape_scc[
                node_scc
            ]
        )

        # ----------------------------------------------------
        # RCC-1:
        # same SCC
        # ----------------------------------------------------

        safe_same = (
            finite
            &
            same
        )

        # ----------------------------------------------------
        # Recursive RCC:
        # escape region is also bounded by score
        # ----------------------------------------------------

        safe_escape = (
            finite
            &
            (~same)
            &
            outside_Q
            &
            (
                escape_node
                <=
                score
            )
        )

        safe = (
            safe_same
            |
            safe_escape
        )

        rcc_ub[
            safe
        ] = np.minimum(
            rcc_ub[
                safe
            ],
            score[
                safe
            ]
        )

        same_scc_ever[
            safe_same
        ] = True

        escape_rcc_ever[
            safe_escape
        ] = True

    del results

    cert_time = (
        time.time()
        -
        certificate_start
    )

    # ========================================================
    # Characteristic path scale
    # ========================================================

    if sum_ssspn > 0:

        lhat = (
            sum_ssspl
            /
            sum_ssspn
        )

    else:

        lhat = 0.0

    # --------------------------------------------------------
    # Keep k >= 1 for RS.
    # --------------------------------------------------------

    raw_k = int(
        np.floor(
            lhat / 2.0
        )
    )

    k = max(
        1,
        raw_k
    )

    # ========================================================
    # Safety of early-termination pruning
    #
    # still_growing=False can only be used for pruning
    # when k <= initial_lb.
    # ========================================================

    early_termination_safe = (
        k <= initial_lb
    )

    finite_rcc = (
        rcc_ub
        !=
        RCC_INF
    )

    same_count = int(
        np.sum(
            same_scc_ever
        )
    )

    escape_count = int(
        np.sum(
            escape_rcc_ever
            &
            (~same_scc_ever)
        )
    )

    total_certified = int(
        np.sum(
            finite_rcc
        )
    )

    print(
        f"Sample count = "
        f"{sample_count}"
    )

    print(
        f"Samples with >=1 non-self reachable node = "
        f"{samples_with_reachability}"
    )

    print(
        f"Lhat (finite sampled distances; self excluded) = "
        f"{lhat:.4f}"
    )

    print(
        f"k = "
        f"{k}"
    )

    print(
        f"Initial LB "
        f"(all sampled eccentricities) = "
        f"{initial_lb}"
    )

    print(
        f"Early-termination pruning safe "
        f"(k <= LB0) = "
        f"{early_termination_safe}"
    )

    print(
        f"Same-SCC certified nodes = "
        f"{same_count}"
    )

    print(
        f"Additional escape-RCC nodes = "
        f"{escape_count}"
    )

    print(
        f"Total RCC coverage = "
        f"{total_certified}/"
        f"{num_nodes} "
        f"("
        f"{total_certified / num_nodes * 100:.2f}%"
        f")"
    )

    print(
        f"RCC construction time = "
        f"{cert_time / 60:.2f} min"
    )

    # ========================================================
    # Stage 2
    # ========================================================

    print(
        "\nStage 2: "
        f"RS-{k} scan"
    )

    with Pool(
        processes=max_workers,
        initializer=init_worker,
        initargs=(
            indptr,
            indices,
            indptr_t,
            indices_t,
            num_nodes
        )
    ) as pool:

        rs_results = pool.map(
            worker_calc_rs,
            [
                (
                    i,
                    k
                )
                for i
                in range(
                    num_nodes
                )
            ],
            chunksize=10000
        )

    # ========================================================
    # Candidate construction
    #
    # RCC pruning is always rigorous.
    #
    # still_growing=False is used as pruning only if
    # k <= initial_lb.
    # ========================================================

    if early_termination_safe:

        candidates = [
            r
            for r in rs_results
            if (
                rcc_ub[
                    r[0]
                ]
                >
                initial_lb
            )
            and
            (
                r[2] is True
            )
        ]

    else:

        # ----------------------------------------------------
        # Extreme safety fallback:
        #
        # if k > LB0, local expansion termination is NOT
        # used to remove nodes.
        # Only RCC-certified upper bounds may prune.
        # ----------------------------------------------------

        candidates = [
            r
            for r in rs_results
            if (
                rcc_ub[
                    r[0]
                ]
                >
                initial_lb
            )
        ]

    # --------------------------------------------------------
    # RS only determines order
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[1]
    )

    candidate_indices = [
        c[0]
        for c in candidates
    ]

    cand_count = len(
        candidate_indices
    )

    cand_ratio = (
        cand_count
        /
        num_nodes
        *
        100.0
    )

    initial_rcc_pruned = int(
        np.sum(
            rcc_ub
            <=
            initial_lb
        )
    )

    if early_termination_safe:

        early_termination_nodes = int(
            np.sum(
                [
                    not r[2]
                    for r in rs_results
                ]
            )
        )

    else:

        early_termination_nodes = 0

    print(
        f"Initial RCC-pruned nodes = "
        f"{initial_rcc_pruned}"
    )

    print(
        f"Safe early-termination nodes = "
        f"{early_termination_nodes}"
    )

    print(
        f"Candidates = "
        f"{cand_count} "
        f"("
        f"{cand_ratio:.2f}%"
        f")"
    )

    # ========================================================
    # Stage 3
    # ========================================================

    print(
        "\nStage 3: "
        "exact verification + RCC dynamic pruning"
    )

    exact_diameter = (
        initial_lb
    )

    verified_stage3 = 0
    dynamic_pruned = 0

    batch_size = (
        max_workers * 4
    )

    with Pool(
        processes=max_workers,
        initializer=init_worker,
        initargs=(
            indptr,
            indices,
            indptr_t,
            indices_t,
            num_nodes
        )
    ) as pool:

        for i in range(
            0,
            cand_count,
            batch_size
        ):

            raw_batch = (
                candidate_indices[
                    i:
                    i + batch_size
                ]
            )

            # ------------------------------------------------
            # Rigorous RCC dynamic pruning
            # ------------------------------------------------

            batch = [
                idx
                for idx
                in raw_batch
                if (
                    rcc_ub[
                        idx
                    ]
                    >
                    exact_diameter
                )
            ]

            dynamic_pruned += (
                len(raw_batch)
                -
                len(batch)
            )

            if not batch:
                continue

            batch_res = (
                pool.map(
                    worker_final_check,
                    batch
                )
            )

            verified_stage3 += (
                len(batch)
            )

            for (
                idx,
                ecc
            ) in batch_res:

                if ecc > exact_diameter:

                    exact_diameter = (
                        ecc
                    )

                    print(
                        f"New diameter LB = "
                        f"{exact_diameter}"
                    )

    # ========================================================
    # Final
    # ========================================================

    total_time = (
        time.time()
        -
        total_start
    ) / 60.0

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "NO-20%-GRN RCC EXACT RESULT"
    )

    print(
        "=" * 80
    )

    print(
        f"Network              : "
        f"{run_label}"
    )

    print(
        f"Diameter             : "
        f"{exact_diameter}"
    )

    print(
        f"Lhat                 : "
        f"{lhat:.4f}"
    )

    print(
        f"k                    : "
        f"{k}"
    )

    print(
        f"Initial LB           : "
        f"{initial_lb}"
    )

    print(
        f"RCC coverage         : "
        f"{total_certified / num_nodes * 100:.2f}%"
    )

    print(
        f"Candidate count      : "
        f"{cand_count}"
    )

    print(
        f"Stage-3 exact SSSP   : "
        f"{verified_stage3}"
    )

    print(
        f"Dynamic RCC-pruned   : "
        f"{dynamic_pruned}"
    )

    sampled_forward_bfs = sample_count
    sampled_reverse_bfs = sample_count
    total_full_bfs = (
        sampled_forward_bfs
        +
        sampled_reverse_bfs
        +
        verified_stage3
    )

    print(
        f"Stage-1 forward BFS  : "
        f"{sampled_forward_bfs}"
    )

    print(
        f"Stage-1 reverse BFS  : "
        f"{sampled_reverse_bfs}"
    )

    print(
        f"Total full BFS       : "
        f"{total_full_bfs}"
    )

    print(
        f"Total time           : "
        f"{total_time:.2f} min"
    )

    print(
        "=" * 80
    )

    return {

        "Network":
            file_name,

        "Scope":
            graph_scope,

        "N":
            num_nodes,

        "Input Edge Rows":
            raw_edge_rows,

        "M":
            num_edges,

        "SCC Count":
            num_scc,

        "Sample Count":
            sample_count,

        "Reachable Samples":
            samples_with_reachability,

        "Lhat":
            round(
                lhat,
                4
            ),

        "k":
            k,

        "Initial LB":
            initial_lb,

        "Early Termination Safe":
            early_termination_safe,

        "Same SCC Certified":
            same_count,

        "Escape RCC Additional":
            escape_count,

        "RCC Coverage (%)":
            round(
                total_certified
                /
                num_nodes
                *
                100.0,
                2
            ),

        "Initial RCC Pruned":
            initial_rcc_pruned,

        "Early Termination Nodes":
            early_termination_nodes,

        "Candidate Count":
            cand_count,

        "Candidate Ratio (%)":
            round(
                cand_ratio,
                2
            ),

        "Stage3 Exact SSSP":
            verified_stage3,

        "Stage3 Exact SSSP / N (%)":
            round(
                verified_stage3
                /
                num_nodes
                *
                100.0,
                4
            ),

        "Dynamic RCC Pruned":
            dynamic_pruned,

        "Stage1 Forward BFS":
            sampled_forward_bfs,

        "Stage1 Reverse BFS":
            sampled_reverse_bfs,

        "Total Full BFS":
            total_full_bfs,

        "Total Full BFS / N (%)":
            round(
                total_full_bfs
                /
                num_nodes
                *
                100.0,
                4
            ),

        "Weak Component Count":
            weak_component_count,

        "Largest WCC Size (full graph)":
            largest_wcc_full_size,

        "Exact Diameter":
            exact_diameter,

        "RCC Build Time (min)":
            round(
                cert_time / 60.0,
                2
            ),

        "Total Time (min)":
            round(
                total_time,
                2
            )
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    freeze_support()

    # ========================================================
    # Final experiment configuration.
    #
    # The main eight-network table uses each complete input graph.
    # Web-Stanford is additionally rerun on its largest weakly connected
    # component as a scope-verification experiment.
    # ========================================================

    NETWORK_FILES = [

        "web-Google.txt",
        "pokec.txt",
         "web-NotreDame.txt",
         "web-Stanford.txt",
         "WikiTalk.txt",
         "twitter.txt",
        "gplus.txt",
        "livejournal.txt",

    ]

    all_results = []
    stanford_wcc_verification = None

    for net in NETWORK_FILES:

        res = run_single_network(
            net
        )

        if res is not None:

            all_results.append(
                res
            )

            pd.DataFrame(
                all_results
            ).to_csv(
                "RCC_no20_results_backup.csv",
                index=False
            )

        # Scope check for Web-Stanford: rerun the same exact algorithm on
        # the largest weakly connected component.  This is a graph-scope
        # verification, not an independent algorithmic validation.
        if os.path.basename(net).lower() == "web-stanford.txt":

            stanford_wcc_verification = run_single_network(
                net,
                largest_wcc_only=True
            )

            if stanford_wcc_verification is not None:

                pd.DataFrame(
                    [stanford_wcc_verification]
                ).to_csv(
                    "Web_Stanford_largest_WCC_verification.csv",
                    index=False
                )

                print(
                    "\n"
                    +
                    "=" * 80
                )
                print(
                    "WEB-STANFORD LARGEST-WCC VERIFICATION OUTPUT"
                )
                print(
                    "=" * 80
                )
                print(
                    f"Largest-WCC nodes    : "
                    f"{stanford_wcc_verification['N']}"
                )
                print(
                    f"Largest-WCC edges    : "
                    f"{stanford_wcc_verification['M']}"
                )
                print(
                    f"Exact diameter       : "
                    f"{stanford_wcc_verification['Exact Diameter']}"
                )
                print(
                    "Saved to              : "
                    "Web_Stanford_largest_WCC_verification.csv"
                )
                print(
                    "=" * 80
                )

    if all_results:

        final_df = pd.DataFrame(
            all_results
        )

        print(
            "\n"
            +
            "=" * 180
        )

        print(
            "FINAL RCC RESULTS WITHOUT 20% GRN THRESHOLD"
        )

        print(
            "=" * 180
        )

        print(
            final_df.to_string(
                index=False
            )
        )

        print(
            "=" * 180
        )

        output_file = (
            "RCC_no20_results.csv"
        )

        final_df.to_csv(
            output_file,
            index=False
        )

        print(
            f"\nResults saved to: "
            f"{os.path.abspath(output_file)}"
        )