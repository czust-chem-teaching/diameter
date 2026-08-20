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

_GLOBAL_RS_VISITED = None
_GLOBAL_RS_QUEUE = None
_GLOBAL_RS_TOKEN = 0

RCC_INF = np.iinfo(np.int64).max // 4


def init_worker(indptr, indices, indptr_t, indices_t, num_nodes):
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

    _GLOBAL_RS_VISITED = None
    _GLOBAL_RS_QUEUE = None
    _GLOBAL_RS_TOKEN = 0


# ============================================================
# Exact BFS
# ============================================================

@njit(fastmath=True)
def numba_fast_bfs(indptr, indices, num_nodes, source_node):
    dists = np.full(num_nodes, -1, dtype=np.int32)
    queue = np.empty(num_nodes, dtype=np.int32)

    head = 0
    tail = 1
    queue[0] = source_node
    dists[source_node] = 0

    max_ecc = 0
    reachable_count = 0
    sum_dists = np.int64(0)

    while head < tail:
        curr = queue[head]
        head += 1
        curr_d = dists[curr]

        for p in range(indptr[curr], indptr[curr + 1]):
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

    return dists, sum_dists, reachable_count, max_ecc


# ============================================================
# RS_k with reusable worker workspace
# ============================================================

@njit(fastmath=True)
def numba_calc_rs_k(indptr, indices, node_idx, k,
                    visited_marker, queue, visit_token):
    head = 0
    tail = 1

    queue[0] = node_idx
    visited_marker[node_idx] = visit_token

    layer_end = tail
    current_step = 0
    still_growing = True

    while head < tail and current_step < k:
        curr = queue[head]
        head += 1

        for p in range(indptr[curr], indptr[curr + 1]):
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
# SCC-DAG routines
# ============================================================

@njit
def scc_forward_closure(dag_indptr, dag_indices, num_scc, source_scc):
    reachable = np.zeros(num_scc, dtype=np.uint8)
    queue = np.empty(num_scc, dtype=np.int32)

    head = 0
    tail = 1
    queue[0] = source_scc
    reachable[source_scc] = 1

    while head < tail:
        c = queue[head]
        head += 1

        for p in range(dag_indptr[c], dag_indptr[c + 1]):
            d = dag_indices[p]
            if reachable[d] == 0:
                reachable[d] = 1
                queue[tail] = d
                tail += 1

    return reachable


@njit
def compute_escape_bound(dag_indptr, dag_indices, topo_order,
                         scc_sizes, in_Q):
    num_scc = len(scc_sizes)
    E = np.zeros(num_scc, dtype=np.int64)

    for pos in range(len(topo_order) - 1, -1, -1):
        c = topo_order[pos]

        if in_Q[c] == 1:
            E[c] = 0
            continue

        best = np.int64(0)
        for p in range(dag_indptr[c], dag_indptr[c + 1]):
            d = dag_indices[p]
            if in_Q[d] == 1:
                continue

            val = np.int64(1) + E[d]
            if val > best:
                best = val

        E[c] = np.int64(scc_sizes[c] - 1) + best

    return E


# ============================================================
# Workers
# ============================================================

def worker_sssp_directed(node_idx):
    global _GLOBAL_INDPTR
    global _GLOBAL_INDICES
    global _GLOBAL_INDPTR_T
    global _GLOBAL_INDICES_T
    global _NUM_NODES

    _, ssspl, ssspn, ecc_out_v = numba_fast_bfs(
        _GLOBAL_INDPTR,
        _GLOBAL_INDICES,
        _NUM_NODES,
        node_idx
    )

    d_to_s, _, _, _ = numba_fast_bfs(
        _GLOBAL_INDPTR_T,
        _GLOBAL_INDICES_T,
        _NUM_NODES,
        node_idx
    )

    return node_idx, ssspl, ssspn, ecc_out_v, d_to_s


def worker_calc_rs(args):
    node_idx, k = args

    global _GLOBAL_INDPTR
    global _GLOBAL_INDICES
    global _NUM_NODES
    global _GLOBAL_RS_VISITED
    global _GLOBAL_RS_QUEUE
    global _GLOBAL_RS_TOKEN

    if _GLOBAL_RS_VISITED is None:
        _GLOBAL_RS_VISITED = np.zeros(_NUM_NODES, dtype=np.int32)
        _GLOBAL_RS_QUEUE = np.empty(_NUM_NODES, dtype=np.int32)
        _GLOBAL_RS_TOKEN = 1
    else:
        _GLOBAL_RS_TOKEN += 1
        if _GLOBAL_RS_TOKEN >= np.iinfo(np.int32).max:
            _GLOBAL_RS_VISITED.fill(0)
            _GLOBAL_RS_TOKEN = 1

    rs_val, still_growing = numba_calc_rs_k(
        _GLOBAL_INDPTR,
        _GLOBAL_INDICES,
        node_idx,
        k,
        _GLOBAL_RS_VISITED,
        _GLOBAL_RS_QUEUE,
        np.int32(_GLOBAL_RS_TOKEN)
    )

    return node_idx, rs_val, bool(still_growing)


def worker_final_check(cand_idx):
    global _GLOBAL_INDPTR
    global _GLOBAL_INDICES
    global _NUM_NODES

    _, _, _, ecc = numba_fast_bfs(
        _GLOBAL_INDPTR,
        _GLOBAL_INDICES,
        _NUM_NODES,
        cand_idx
    )

    return cand_idx, ecc


# ============================================================
# Graph helpers
# ============================================================

def build_condensation_dag(adj, scc_labels, num_scc):
    num_nodes = adj.shape[0]

    src_nodes = np.repeat(
        np.arange(num_nodes, dtype=np.int32),
        np.diff(adj.indptr)
    )
    dst_nodes = adj.indices.astype(np.int32, copy=False)

    src_scc = scc_labels[src_nodes]
    dst_scc = scc_labels[dst_nodes]
    mask = src_scc != dst_scc

    dag_src = src_scc[mask]
    dag_dst = dst_scc[mask]

    dag = csr_matrix(
        (
            np.ones(len(dag_src), dtype=np.uint8),
            (dag_src, dag_dst)
        ),
        shape=(num_scc, num_scc)
    )

    dag.sum_duplicates()
    dag.data[:] = 1
    dag.eliminate_zeros()
    return dag


def topological_order(dag):
    num_scc = dag.shape[0]
    indegree = np.asarray(dag.sum(axis=0)).ravel().astype(np.int64)

    queue = np.empty(num_scc, dtype=np.int32)
    topo = np.empty(num_scc, dtype=np.int32)

    head = 0
    tail = 0

    for c in range(num_scc):
        if indegree[c] == 0:
            queue[tail] = c
            tail += 1

    count = 0
    while head < tail:
        c = queue[head]
        head += 1

        topo[count] = c
        count += 1

        for p in range(dag.indptr[c], dag.indptr[c + 1]):
            d = dag.indices[p]
            indegree[d] -= 1
            if indegree[d] == 0:
                queue[tail] = d
                tail += 1

    if count != num_scc:
        raise RuntimeError("Condensation graph is not a DAG.")

    return topo


# ============================================================
# One-time Numba warm-up
# ============================================================

def compile_numba_kernels_once():
    tiny_indptr = np.array([0, 1, 2, 2], dtype=np.int32)
    tiny_indices = np.array([1, 2], dtype=np.int32)

    numba_fast_bfs(tiny_indptr, tiny_indices, 3, 0)

    visited = np.zeros(3, dtype=np.int32)
    queue = np.empty(3, dtype=np.int32)
    numba_calc_rs_k(
        tiny_indptr, tiny_indices, 0, 2,
        visited, queue, np.int32(1)
    )

    topo = np.array([0, 1, 2], dtype=np.int32)
    sizes = np.ones(3, dtype=np.int32)
    q = scc_forward_closure(tiny_indptr, tiny_indices, 3, 0)
    compute_escape_bound(tiny_indptr, tiny_indices, topo, sizes, q)


# ============================================================
# Fast paired certificate ablation
# ============================================================

def run_fast_ablation(file_name):
    """
    Fast and controlled ablation.

    Common work is done ONCE:
      * graph loading
      * SCC decomposition and condensation DAG
      * the same 500 forward/reverse sentinel BFSs
      * the same RS_k scan

    During the same sentinel pass we build two rigorous UB arrays:
      UB_same : same-SCC certificate only
      UB_full : same-SCC + escape RCC

    We run Stage 3 only for FULL RCC to obtain the exact D.

    Then, using this common exact D, we compare how many nodes remain
    uncertified under the two certificate systems.  This is a pure
    certificate-strength ablation and avoids executing the potentially huge
    SAME_SCC_ONLY exact-verification workload.

    IMPORTANT:
      'Residual at D' is NOT reported as the actual Stage-3 BFS count of the
      same-only algorithm.  It is the number of nodes that remain unresolved
      by that certificate even after the lower bound has reached the true D.
    """

    file_path = os.path.join(".", file_name)
    if not os.path.exists(file_path):
        print(f"file does not exist: {file_path}")
        return None

    TARGET_SAMPLES = 500
    max_workers = min(cpu_count(), 14)

    total_start = time.time()

    print("\n" + "=" * 90)
    print(f"FAST RCC CERTIFICATE ABLATION: {file_name}")
    print("=" * 90)

    # --------------------------------------------------------
    # Load + deduplicate graph
    # --------------------------------------------------------
    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        header=None,
        names=["src", "dst"],
        comment="#"
    )

    unique_nodes = np.unique(df[["src", "dst"]].values)
    num_nodes = len(unique_nodes)
    raw_edge_rows = len(df)

    mapping = {node: i for i, node in enumerate(unique_nodes)}
    src_idx = df["src"].map(mapping).to_numpy()
    dst_idx = df["dst"].map(mapping).to_numpy()

    adj = csr_matrix(
        (
            np.ones(raw_edge_rows, dtype=bool),
            (src_idx, dst_idx)
        ),
        shape=(num_nodes, num_nodes)
    )
    adj.sum_duplicates()
    adj.eliminate_zeros()
    num_edges = int(adj.nnz)

    adj_t = adj.transpose().tocsr()

    # From this point, timing excludes disk parsing.
    algorithm_start = time.time()

    print(f"Nodes = {num_nodes}")
    print(f"Unique directed edges = {num_edges}")

    # --------------------------------------------------------
    # SCC + condensation DAG (needed by FULL RCC)
    # --------------------------------------------------------
    scc_start = time.time()

    num_scc, scc_labels = connected_components(
        adj,
        directed=True,
        connection="strong",
        return_labels=True
    )
    scc_labels = scc_labels.astype(np.int32, copy=False)
    scc_sizes = np.bincount(
        scc_labels,
        minlength=num_scc
    ).astype(np.int32)

    dag = build_condensation_dag(adj, scc_labels, num_scc)
    topo = topological_order(dag)

    print(f"SCC count = {num_scc}")
    print(f"Largest SCC = {int(np.max(scc_sizes))} "
          f"({np.max(scc_sizes) / num_nodes * 100:.2f}%)")
    print(f"SCC DAG edges = {dag.nnz}")
    print(f"SCC preprocessing = {time.time() - scc_start:.2f} sec")

    # CSR arrays
    indptr = adj.indptr.astype(np.int32)
    indices = adj.indices.astype(np.int32)
    indptr_t = adj_t.indptr.astype(np.int32)
    indices_t = adj_t.indices.astype(np.int32)

    dag_indptr = dag.indptr.astype(np.int32)
    dag_indices = dag.indices.astype(np.int32)

    # --------------------------------------------------------
    # Stage 1 ONCE: same sentinel sample builds BOTH UBs
    # --------------------------------------------------------
    print("\nStage 1: one common 500-sentinel pass builds both certificates")

    rng = np.random.default_rng(2026)
    sample_indices = rng.choice(
        num_nodes,
        min(TARGET_SAMPLES, num_nodes),
        replace=False
    )

    ub_same = np.full(num_nodes, RCC_INF, dtype=np.int64)
    ub_full = np.full(num_nodes, RCC_INF, dtype=np.int64)

    sum_ssspl = 0.0
    sum_ssspn = 0.0
    samples_with_reachability = 0
    initial_lb = 0

    cert_start = time.time()

    # Stream results instead of storing 500 d_to_s arrays simultaneously.
    with Pool(
        processes=max_workers,
        initializer=init_worker,
        initargs=(indptr, indices, indptr_t, indices_t, num_nodes)
    ) as pool:

        for sentinel, ssspl, ssspn, ecc_s, d_to_s in pool.imap(
            worker_sssp_directed,
            sample_indices,
            chunksize=1
        ):
            if ssspn > 0:
                sum_ssspl += float(ssspl)
                sum_ssspn += float(ssspn)
                samples_with_reachability += 1

            if ecc_s > initial_lb:
                initial_lb = ecc_s

            Cs = scc_labels[sentinel]

            finite = d_to_s >= 0
            score = d_to_s.astype(np.int64, copy=True)
            score[finite] += np.int64(ecc_s)

            same = scc_labels == Cs
            safe_same = finite & same

            # Same-SCC-only UB.
            ub_same[safe_same] = np.minimum(
                ub_same[safe_same],
                score[safe_same]
            )

            # Full RCC always includes the same-SCC certificate.
            ub_full[safe_same] = np.minimum(
                ub_full[safe_same],
                score[safe_same]
            )

            # Escape RCC.
            in_Q = scc_forward_closure(
                dag_indptr,
                dag_indices,
                num_scc,
                Cs
            )
            escape_scc = compute_escape_bound(
                dag_indptr,
                dag_indices,
                topo,
                scc_sizes,
                in_Q
            )

            outside_Q = in_Q[scc_labels] == 0
            escape_node = escape_scc[scc_labels]

            safe_escape = (
                finite
                & (~same)
                & outside_Q
                & (escape_node <= score)
            )

            ub_full[safe_escape] = np.minimum(
                ub_full[safe_escape],
                score[safe_escape]
            )

    cert_time = time.time() - cert_start

    lhat = (sum_ssspl / sum_ssspn) if sum_ssspn > 0 else 0.0
    k = max(1, int(np.floor(lhat / 2.0)))
    early_safe = (k <= initial_lb)

    finite_same = ub_same != RCC_INF
    finite_full = ub_full != RCC_INF

    same_coverage = int(np.sum(finite_same))
    full_coverage = int(np.sum(finite_full))
    escape_additional = int(np.sum(finite_full & (~finite_same)))

    print(f"Samples with >=1 non-self reachable node = {samples_with_reachability}")
    print(f"Lhat = {lhat:.4f}")
    print(f"k = {k}")
    print(f"Initial LB = {initial_lb}")
    print(f"Early termination safe = {early_safe}")
    print(f"Same-only RCC coverage = {same_coverage}/{num_nodes} "
          f"({same_coverage / num_nodes * 100:.2f}%)")
    print(f"Full RCC coverage = {full_coverage}/{num_nodes} "
          f"({full_coverage / num_nodes * 100:.2f}%)")
    print(f"Escape-RCC additional certified nodes = {escape_additional}")
    print(f"Common Stage-1 + certificate construction time = {cert_time / 60:.2f} min")

    # --------------------------------------------------------
    # Stage 2 ONCE
    # --------------------------------------------------------
    print(f"\nStage 2: one common RS-{k} scan")

    rs_values = np.empty(num_nodes, dtype=np.int32)
    still_growing = np.empty(num_nodes, dtype=bool)

    with Pool(
        processes=max_workers,
        initializer=init_worker,
        initargs=(indptr, indices, indptr_t, indices_t, num_nodes)
    ) as pool:

        iterator = pool.imap(
            worker_calc_rs,
            ((i, k) for i in range(num_nodes)),
            chunksize=10000
        )

        for node_idx, rs_val, growing in iterator:
            rs_values[node_idx] = rs_val
            still_growing[node_idx] = growing

    if early_safe:
        active_local = still_growing
    else:
        active_local = np.ones(num_nodes, dtype=bool)

    cand_same_mask = active_local & (ub_same > initial_lb)
    cand_full_mask = active_local & (ub_full > initial_lb)

    same_candidates = int(np.sum(cand_same_mask))
    full_candidates = int(np.sum(cand_full_mask))

    print(f"Same-only candidates at LB0 = {same_candidates} "
          f"({same_candidates / num_nodes * 100:.2f}%)")
    print(f"Full-RCC candidates at LB0 = {full_candidates} "
          f"({full_candidates / num_nodes * 100:.2f}%)")
    print(f"Candidate reduction from escape RCC = "
          f"{same_candidates - full_candidates}")

    # --------------------------------------------------------
    # Stage 3: run ONLY FULL RCC to obtain exact D
    # --------------------------------------------------------
    print("\nStage 3: exact verification for FULL RCC only")

    full_candidate_indices = np.flatnonzero(cand_full_mask)
    if full_candidate_indices.size > 0:
        order = np.argsort(
            rs_values[full_candidate_indices],
            kind="stable"
        )
        full_candidate_indices = full_candidate_indices[order]

    exact_diameter = initial_lb
    verified_stage3 = 0
    dynamic_pruned = 0
    batch_size = max_workers * 4

    with Pool(
        processes=max_workers,
        initializer=init_worker,
        initargs=(indptr, indices, indptr_t, indices_t, num_nodes)
    ) as pool:

        for i in range(0, len(full_candidate_indices), batch_size):
            raw_batch = full_candidate_indices[i:i + batch_size]

            batch = [
                int(idx)
                for idx in raw_batch
                if ub_full[idx] > exact_diameter
            ]

            dynamic_pruned += len(raw_batch) - len(batch)

            if not batch:
                continue

            batch_res = pool.map(worker_final_check, batch)
            verified_stage3 += len(batch)

            for _, ecc in batch_res:
                if ecc > exact_diameter:
                    exact_diameter = ecc
                    print(f"New diameter LB = {exact_diameter}")

    # --------------------------------------------------------
    # Pure certificate-strength comparison at common exact D
    # --------------------------------------------------------
    # Once LB = D, any active node with UB <= D is certified by the
    # corresponding certificate.  Nodes with UB > D remain unresolved by
    # that certificate and would require exact verification if that
    # certificate were used by itself.
    residual_same_mask = active_local & (ub_same > exact_diameter)
    residual_full_mask = active_local & (ub_full > exact_diameter)

    residual_same = int(np.sum(residual_same_mask))
    residual_full = int(np.sum(residual_full_mask))

    escape_resolved_at_D = int(np.sum(
        active_local
        & (ub_same > exact_diameter)
        & (ub_full <= exact_diameter)
    ))

    sampled_bfs = 2 * len(sample_indices)
    full_total_bfs = sampled_bfs + verified_stage3

    algorithm_time = (time.time() - algorithm_start) / 60.0
    total_time = (time.time() - total_start) / 60.0

    print("\n" + "=" * 90)
    print("FAST CERTIFICATE ABLATION RESULT")
    print("=" * 90)
    print(f"Exact diameter D                         : {exact_diameter}")
    print(f"Same-only RCC coverage                  : {same_coverage / num_nodes * 100:.2f}%")
    print(f"Full RCC coverage                       : {full_coverage / num_nodes * 100:.2f}%")
    print(f"Escape additional certified nodes       : {escape_additional}")
    print(f"Same-only candidates at LB0             : {same_candidates}")
    print(f"Full-RCC candidates at LB0              : {full_candidates}")
    print(f"Candidate reduction                     : {same_candidates - full_candidates}")
    print(f"Same-only residual unresolved at D      : {residual_same}")
    print(f"Full-RCC residual unresolved at D       : {residual_full}")
    print(f"Nodes resolved specifically by escape @D: {escape_resolved_at_D}")
    print(f"Actual FULL-RCC Stage-3 BFS              : {verified_stage3}")
    print(f"Actual FULL-RCC total full BFS           : {full_total_bfs}")
    print(f"Algorithm time                          : {algorithm_time:.2f} min")
    print("=" * 90)

    return {
        "Network": file_name,
        "N": num_nodes,
        "M": num_edges,
        "SCC Count": num_scc,
        "SCC DAG Edges": int(dag.nnz),
        "Sample Count": len(sample_indices),
        "Reachable Samples": samples_with_reachability,
        "Lhat": round(lhat, 4),
        "k": k,
        "LB0": int(initial_lb),
        "D": int(exact_diameter),
        "Early Termination Safe": bool(early_safe),

        "Same-only RCC Coverage": same_coverage,
        "Same-only RCC Coverage (%)": round(same_coverage / num_nodes * 100.0, 2),
        "Full RCC Coverage": full_coverage,
        "Full RCC Coverage (%)": round(full_coverage / num_nodes * 100.0, 2),
        "Coverage Gain (pp)": round((full_coverage - same_coverage) / num_nodes * 100.0, 2),
        "Escape RCC Additional": escape_additional,

        "Same-only Candidates at LB0": same_candidates,
        "Full RCC Candidates at LB0": full_candidates,
        "Candidate Reduction": same_candidates - full_candidates,
        "Candidate Reduction (%)": round(
            (same_candidates - full_candidates) / same_candidates * 100.0,
            2
        ) if same_candidates > 0 else 0.0,

        "Same-only Residual at D": residual_same,
        "Full RCC Residual at D": residual_full,
        "Residual Reduction at D": residual_same - residual_full,
        "Residual Reduction at D (%)": round(
            (residual_same - residual_full) / residual_same * 100.0,
            2
        ) if residual_same > 0 else 0.0,
        "Escape-resolved Nodes at D": escape_resolved_at_D,

        "Full RCC Stage3 Exact SSSP": verified_stage3,
        "Full RCC Dynamic Pruned": dynamic_pruned,
        "Full RCC Total Full BFS": full_total_bfs,
        "Full RCC Total Full BFS / N (%)": round(
            full_total_bfs / num_nodes * 100.0,
            4
        ),

        "Common Certificate Build Time (min)": round(cert_time / 60.0, 4),
        "Algorithm Time (min)": round(algorithm_time, 4),
        "Total Time (min)": round(total_time, 2),
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    freeze_support()
    compile_numba_kernels_once()

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

    for net in NETWORK_FILES:
        res = run_fast_ablation(net)
        if res is not None:
            all_results.append(res)

            pd.DataFrame(all_results).to_csv(
                "RCC_escape_ablation_fast_backup.csv",
                index=False
            )

    if all_results:
        final_df = pd.DataFrame(all_results)

        print("\n" + "=" * 220)
        print("FINAL FAST RCC CERTIFICATE ABLATION")
        print("=" * 220)
        print(final_df.to_string(index=False))
        print("=" * 220)

        output_file = "RCC_escape_ablation_fast.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {os.path.abspath(output_file)}")
