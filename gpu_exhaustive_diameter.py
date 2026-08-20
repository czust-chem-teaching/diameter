import torch
import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm

def compute_exact_directed_diameter_gpu(file_path):
    """
    Exhaustively compute the maximum finite directed shortest-path distance.

    BFS layers for multiple sources are propagated in parallel on the GPU
    using sparse matrix-dense matrix multiplication. Sources with zero
    out-degree are omitted because their reachable eccentricity is zero.
    Self-loops are also omitted because they do not affect shortest-path
    distances between distinct vertices.
    """

    print(f"\n{'='*60}")
    print(f"GPU EXHAUSTIVE DIAMETER VALIDATION: {os.path.basename(file_path)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[Error] CUDA device not available.")
        return None
        
    gpu_name = torch.cuda.get_device_name(0)
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Device: {gpu_name} ({total_memory_gb:.1f} GB GPU memory)")
    start_time = time.time()
    
    # 1. Load the directed edge list
    print("Loading network file...")
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['src', 'dst'], comment='#')
    unique_nodes = np.unique(df[['src', 'dst']].values)
    num_nodes = len(unique_nodes)
    num_edges = len(df)
    
    print(f"Nodes (N): {num_nodes:,} | input edge rows: {num_edges:,}")
    
    mapping = {node: i for i, node in enumerate(unique_nodes)}
    src_mapped = df['src'].map(mapping).values.astype(np.int64)
    dst_mapped = df['dst'].map(mapping).values.astype(np.int64)
    
    # Self-loops do not change shortest-path distances between distinct
    # vertices and can be removed from the validation graph.
    valid_mask = src_mapped != dst_mapped
    src_mapped = src_mapped[valid_mask]
    dst_mapped = dst_mapped[valid_mask]
    
    # Sources with zero out-degree have reachable eccentricity zero, so
    # they do not need to be processed when searching for the maximum.
    out_degrees = np.bincount(src_mapped, minlength=num_nodes)
    valid_sources = np.where(out_degrees > 0)[0]
    total_sources = len(valid_sources)
    print(f"Sources with out-degree > 0: {total_sources:,} / {num_nodes:,}")
    
    # 2. Build the incoming-edge CSR matrix
    #
    # A_in[dst, src] = 1, so frontier propagation is performed as
    # next_frontier = A_in @ frontier.
    print("Building GPU sparse CSR representation...")
    
    # Construct CSR arrays with rows indexed by destination vertices.
    crow_indices = np.zeros(num_nodes + 1, dtype=np.int64)
    in_counts = np.bincount(dst_mapped, minlength=num_nodes)
    crow_indices[1:] = np.cumsum(in_counts)
    
    sort_idx = np.lexsort((src_mapped, dst_mapped))
    col_indices = src_mapped[sort_idx]
    values = np.ones(len(col_indices), dtype=np.float32)
    
    # Transfer CSR arrays to the GPU.
    d_crow = torch.tensor(crow_indices, dtype=torch.int64, device=device)
    d_col = torch.tensor(col_indices, dtype=torch.int64, device=device)
    d_val = torch.tensor(values, dtype=torch.float32, device=device)
    
    # Construct the GPU sparse CSR tensor used for frontier propagation.
    adj_csr = torch.sparse_csr_tensor(d_crow, d_col, d_val, (num_nodes, num_nodes), device=device)
    
    del df, src_mapped, dst_mapped, crow_indices, col_indices, values, in_counts, sort_idx
    torch.cuda.empty_cache()
    print(f"GPU sparse matrix ready in {time.time() - start_time:.2f} s")
    
    compute_start = time.time()
    exact_diameter = 0
    
    # 3. Process source vertices in batches.
    #
    # BATCH_SIZE = 32 was used in the reported validation experiments.
    # It may be adjusted for different GPU memory capacities.
    BATCH_SIZE = 32
    print(f"Batch size: {BATCH_SIZE} sources")
    
    with tqdm(total=total_sources, desc=f"GPU validation [{os.path.basename(file_path)}]", unit="source") as pbar:
        for i in range(0, total_sources, BATCH_SIZE):
            batch_srcs = valid_sources[i : min(i + BATCH_SIZE, total_sources)]
            cur_b = len(batch_srcs)
            
            # visited has shape (current_batch_size, num_nodes).
            visited = torch.zeros((cur_b, num_nodes), dtype=torch.bool, device=device)
            # frontier has shape (num_nodes, current_batch_size).
            frontier = torch.zeros((num_nodes, cur_b), dtype=torch.float32, device=device)
            
            # Initialize the source vertices.
            for col_idx, s in enumerate(batch_srcs):
                visited[col_idx, s] = True
                frontier[s, col_idx] = 1.0
                
            step = 0
            while True:
                # Propagate all current frontiers by sparse matrix
                # multiplication on the GPU.
                next_frontier = torch.sparse.mm(adj_csr, frontier)
                
                # Remove vertices already reached by each source.
                mask = (next_frontier.t() > 0) & (~visited)
                
                if not mask.any():
                    break
                    
                step += 1
                visited |= mask
                
                # Construct the next BFS frontier.
                frontier.zero_()
                frontier = mask.t().to(torch.float32)
                
            if step > exact_diameter:
                exact_diameter = step
                tqdm.write(f"  Updated diameter lower bound: {exact_diameter}")
                
            pbar.update(cur_b)
            pbar.set_postfix({"Current Max Diameter": exact_diameter})

    print(f"\n{'='*60}")
    print(f"GPU EXHAUSTIVE VALIDATION COMPLETE: {os.path.basename(file_path)}")
    print(f"Exact directed diameter: {exact_diameter}")
    print(f"Computation time: {(time.time() - compute_start) / 60.0:.2f} min")
    print(f"{'='*60}\n")
    
    return exact_diameter

if __name__ == "__main__":
    NETWORK_FILES = [
        "livejournal.txt"
    ]
    
    for file_name in NETWORK_FILES:
        file_path = os.path.join(".", file_name)
        if not os.path.exists(file_path):
            file_path = os.path.join("..", file_name)
            
        if os.path.exists(file_path):
            compute_exact_directed_diameter_gpu(file_path)
        else:
            print(f"[Skip] File not found: {file_name}")
