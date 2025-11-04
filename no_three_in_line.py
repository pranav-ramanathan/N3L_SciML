# no_three_in_line.py
# Batched No-Three-in-Line generator + dataset builder (PyTorch)
# - Internal coordinates are ZERO-BASED (x, y)
# - Grids are shaped (N, N) and indexed as [x, y] consistently
# - When saving for Julia, we can export triplets as ONE-BASED (row, col)

from __future__ import annotations
import dataclasses
from collections import Counter
from typing import List, Tuple, Dict, Optional
import time
import numpy as np
import torch
from rich.console import Console

console = Console()


# ============================================
# Line helpers & triplets (internal: 0-based x,y)
# ============================================

def _canon_line(A: int, B: int, C: int) -> Tuple[int, int, int]:
    from math import gcd
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g; B //= g; C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A; B = -B; C = -C
    return (A, B, C)

def canonical_line_from_points_xy(p1: Tuple[int,int], p2: Tuple[int,int]) -> Tuple[int,int,int]:
    # p = (x,y), 0-based
    x1, y1 = p1; x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    return _canon_line(A, B, C)

def generate_triplets_xy0(n: int) -> List[Tuple[int,int,int,int,int,int]]:
    """All collinear triples on an n×n grid, using 0-based (x,y)."""
    coords = [(x, y) for x in range(n) for y in range(n)]
    triplets: List[Tuple[int,int,int,int,int,int]] = []
    for a in range(len(coords) - 2):
        x1, y1 = coords[a]
        for b in range(a + 1, len(coords) - 1):
            x2, y2 = coords[b]
            for c in range(b + 1, len(coords)):
                x3, y3 = coords[c]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    triplets.append((x1, y1, x2, y2, x3, y3))
    return triplets

def triplets_xy0_to_rowcol1(triplets_xy0: List[Tuple[int,int,int,int,int,int]]) -> np.ndarray:
    """Convert (x,y) 0-based to (row,col) 1-based for Julia.
    
    In Python: grid[x, y] where x is first dimension (rows), y is second dimension (cols)
    In Julia: grid[i, j] where i is rows, j is cols
    Therefore: i = x+1, j = y+1
    """
    out = []
    for (x1,y1,x2,y2,x3,y3) in triplets_xy0:
        i1,j1 = x1+1, y1+1  # Fixed: x maps to rows (i), y maps to cols (j)
        i2,j2 = x2+1, y2+1
        i3,j3 = x3+1, y3+1
        out.append((i1,j1,i2,j2,i3,j3))
    return np.asarray(out, dtype=np.int32)


# ============================================
# Dataset labels & masks
# ============================================

def compute_labels(final_binary_grid: torch.Tensor, n: int, triplets_xy0: List[Tuple[int,int,int,int,int,int]]):
    """Compute board quality metrics (thresholded at >0.5). triplets are 0-based (x,y)."""
    grid_cpu = final_binary_grid.detach().to('cpu')
    data_np = grid_cpu.numpy()
    num_points = int((data_np > 0.5).sum())

    num_violations = 0
    if num_points >= 3:
        for (x1, y1, x2, y2, x3, y3) in triplets_xy0:
            if data_np[x1, y1] > 0.5 and data_np[x2, y2] > 0.5 and data_np[x3, y3] > 0.5:
                num_violations += 1

    deficit = max(0, 2 * n - num_points)
    score = num_points - 100.0 * num_violations - 1.0 * deficit
    return {
        "num_points": np.int32(num_points),
        "num_violations": np.int32(num_violations),
        "deficit": np.int32(deficit),
        "score": np.float32(score),
    }

def make_top2n_mask(final_binary_grid: torch.Tensor, n: int):
    """Float32 mask with up to 2n ones at the strongest cells.
       If there are < 2n ones, we just mark the ones."""
    flat = final_binary_grid.flatten().detach().cpu().numpy()
    ones = np.flatnonzero(flat > 0.5)
    mask = np.zeros_like(flat, dtype=np.float32)
    if len(ones) >= 2 * n:
        top_idx = np.argsort(-flat, kind='mergesort')[: 2 * n]
        mask[top_idx] = 1.0
    else:
        mask[ones] = 1.0
    return torch.from_numpy(mask.reshape(n, n))


# ============================================
# Core class
# ============================================

@dataclasses.dataclass
class NoThreeInLine:
    """
    Batched No-Three-in-Line constructor.
    Internal grid shape per item: (N, N), indexed as [x, y] (0-based).
    """
    batch_size: int
    grid_size: int
    max_points: int
    device: str = "cpu"
    aggressive_blocking: bool = False   # if True, block whole row/col/diag once ≥2 points
    post_add_full_update: bool = False  # if True, run full update_forbidden_squares after each add (slower, stronger pruning)

    def __post_init__(self):
        N = self.N = self.grid_size
        assert self.grid_size < 64, "grid_size must fit in int8 coordinate (N < 64)."
        assert self.max_points <= 2 * N, "max_points should be ≤ 2*N."

        # 1: current_constructions ∈ {1(point), 0(empty), -1(forbidden)}
        self.current_constructions = torch.zeros(
            (self.batch_size, N, N), dtype=torch.int8, device=self.device
        )

        # 2: points_list: (batch, max_points, 2) of (x,y) or (-1,-1) for slots
        self.points_list = torch.full(
            (self.batch_size, self.max_points, 2), -1, dtype=torch.int8, device=self.device
        )

        # 3: current_counts (how many points placed)
        self.current_counts = torch.zeros((self.batch_size,), dtype=torch.int16, device=self.device)

        # null token
        self.null_tensor = torch.tensor([-1, -1], dtype=torch.int8, device=self.device)

        # caches
        self._pair_cache: Dict[int, torch.Tensor] = {}
        self.history_buffer: List[torch.Tensor] = []

    # ------------------ optional “aggressive” block along four slope classes
    def _block_entire_lines(self, batch_indices: torch.Tensor, new_points: torch.Tensor):
        if batch_indices.numel() == 0:
            return
        for i, b in enumerate(batch_indices):
            x, y = map(int, new_points[i].tolist())

            # "Row" here = sweep y for fixed x (our axes are [x, y])
            row_count = int(torch.sum(self.current_constructions[b, x, :] == 1).item())
            if row_count >= 2:
                empty = self.current_constructions[b, x, :] == 0
                self.current_constructions[b, x, empty] = -1

            # "Column" = sweep x for fixed y
            col_count = int(torch.sum(self.current_constructions[b, :, y] == 1).item())
            if col_count >= 2:
                empty = self.current_constructions[b, :, y] == 0
                self.current_constructions[b, empty, y] = -1

            # main diag through (x,y)
            sx, sy = x, y
            while sx > 0 and sy > 0:
                sx -= 1; sy -= 1
            diag_coords = []
            cx, cy = sx, sy
            while cx < self.N and cy < self.N:
                diag_coords.append((cx, cy)); cx += 1; cy += 1
            diag_count = sum(self.current_constructions[b, cx, cy] == 1 for (cx, cy) in diag_coords)
            if diag_count >= 2:
                for (cx, cy) in diag_coords:
                    if self.current_constructions[b, cx, cy] == 0:
                        self.current_constructions[b, cx, cy] = -1

            # anti-diag through (x,y)
            sx, sy = x, y
            while sx > 0 and sy < self.N - 1:
                sx -= 1; sy += 1
            adiag_coords = []
            cx, cy = sx, sy
            while cx < self.N and cy >= 0:
                adiag_coords.append((cx, cy)); cx += 1; cy -= 1
            adiag_count = sum(self.current_constructions[b, cx, cy] == 1 for (cx, cy) in adiag_coords)
            if adiag_count >= 2:
                for (cx, cy) in adiag_coords:
                    if self.current_constructions[b, cx, cy] == 0:
                        self.current_constructions[b, cx, cy] = -1

    # ------------------ fast-but-incomplete forbidden marking
    def _update_forbidden_squares_after_add(self, batch_indices, new_points, current_counts):
        if batch_indices.numel() == 0:
            return
        valid_mask = current_counts >= 1
        if not valid_mask.any():
            return

        batch_indices = batch_indices[valid_mask]
        new_points = new_points[valid_mask]
        current_counts = current_counts[valid_mask]

        existing_points_full = self.points_list[batch_indices]  # (m, max_points, 2)

        # Use only the first 'count' points in each batch as existing
        slots = torch.arange(self.max_points, device=self.device).expand(len(current_counts), -1)
        existing_mask = slots < current_counts.unsqueeze(1)
        p1 = new_points.unsqueeze(1)            # (m,1,2)
        p2 = existing_points_full               # (m,max_points,2)

        p3 = 2 * p2 - p1                        # reflection trick (fast, not complete)

        # filter by valid existing slots
        p3_mask = existing_mask
        in_bounds = (p3 >= 0).all(dim=-1) & (p3 < self.N).all(dim=-1)
        p3_mask &= in_bounds

        cand_p3 = p3[p3_mask].to(torch.int)
        if cand_p3.numel() == 0:
            return
        batch_idx_expanded = batch_indices.unsqueeze(1).expand(-1, self.max_points)
        cand_bidx = batch_idx_expanded[p3_mask]

        occ = self.current_constructions[cand_bidx, cand_p3[:, 0], cand_p3[:, 1]]
        is_empty = occ == 0
        if is_empty.any():
            self.current_constructions[cand_bidx[is_empty], cand_p3[is_empty, 0], cand_p3[is_empty, 1]] = -1

        if self.post_add_full_update:
            # Optional: full, correct pruning after adds (costly but thorough)
            self.update_forbidden_squares(self.current_constructions[batch_indices])

    # ------------------ public: add a batch of points
    def add_points(self, points: torch.Tensor):
        points = points.to(dtype=torch.int8, device=self.device)
        points_int = points.to(dtype=torch.int)  # for indexing

        non_null_mask = (points != self.null_tensor).any(dim=-1)
        assert points.shape[0] == self.batch_size
        if non_null_mask.any():
            assert torch.max(self.current_counts[non_null_mask]).item() < self.max_points
        assert (-1 <= points).all() and (points < self.N).all()

        batch_all = torch.arange(self.batch_size, device=self.device)
        valid_idx = batch_all[non_null_mask]
        if valid_idx.numel() == 0:
            return

        coords = points_int[non_null_mask]
        occ = self.current_constructions[valid_idx, coords[:, 0], coords[:, 1]]
        is_available = occ == 0

        final_add_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        final_add_mask[valid_idx[is_available]] = True

        added_point_counts = torch.zeros_like(self.current_counts)

        batched_idxs, batched_pts, batched_counts = [], [], []
        for cur_count in torch.unique(self.current_counts).tolist():
            count_mask = (self.current_counts == cur_count)
            ins_mask = count_mask & final_add_mask
            if not ins_mask.any():
                continue

            ins_idx = batch_all[ins_mask]
            pts_to_add = points[ins_mask]
            pts_to_add_int = points_int[ins_mask]

            self.current_constructions[ins_idx, pts_to_add_int[:, 0], pts_to_add_int[:, 1]] = 1
            self.points_list[ins_idx, cur_count] = pts_to_add
            added_point_counts[ins_idx] = 1

            batched_idxs.append(ins_idx)
            batched_pts.append(pts_to_add)
            batched_counts.append(torch.full_like(ins_idx, cur_count, device=self.device))

        if batched_idxs:
            cat_idx = torch.cat(batched_idxs)
            cat_pts = torch.cat(batched_pts)
            self._block_entire_lines(cat_idx, cat_pts) if self.aggressive_blocking else \
                self._update_forbidden_squares_after_add(cat_idx, cat_pts, torch.cat(batched_counts))

        self.current_counts += added_point_counts

    # ------------------ fully vectorized candidate check
    def check_new_points(self, new_points: torch.Tensor) -> torch.Tensor:
        # new_points: (B, K, 2)
        B, K, _ = new_points.shape
        good = (new_points != self.null_tensor).any(dim=-1)

        in_bounds = (new_points >= 0).all(dim=-1) & (new_points < self.N).all(dim=-1)
        good &= in_bounds

        batch_idx = torch.arange(self.batch_size, device=self.device).unsqueeze(1).expand(-1, K)
        pts = new_points[good].to(torch.int)
        bsel = batch_idx[good]

        occ = self.current_constructions[bsel, pts[:, 0], pts[:, 1]]
        is_occupied = torch.zeros_like(good)
        is_occupied[good] = (occ != 0)
        good &= ~is_occupied

        unique_counts = torch.unique(self.current_counts)
        for cur_count in unique_counts.tolist():
            if cur_count < 2:
                continue
            mask = (self.current_counts == cur_count)
            if not mask.any():
                continue

            batches = mask.nonzero(as_tuple=True)[0]
            existing = self.points_list[batches]   # (m, max_points, 2)
            candidates = new_points[batches]       # (m, K, 2)

            if cur_count not in self._pair_cache:
                self._pair_cache[cur_count] = torch.combinations(torch.arange(cur_count), r=2)
            pairs = self._pair_cache[cur_count].to(self.device)

            p1s = existing[:, pairs[:, 0]].unsqueeze(2)  # (m, P, 1, 2)
            p2s = existing[:, pairs[:, 1]].unsqueeze(2)  # (m, P, 1, 2)
            p3s = candidates.unsqueeze(1)                # (m, 1, K, 2)

            x1, y1 = p1s[..., 0], p1s[..., 1]
            x2, y2 = p2s[..., 0], p2s[..., 1]
            x3, y3 = p3s[..., 0], p3s[..., 1]

            col = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            invalid = (col == 0).any(dim=1)              # (m, K)

            good[batches] &= ~invalid

            # Pre-mark invalids as forbidden to prune future scans
            forb_pts = new_points[batches][invalid].to(torch.int)
            forb_bidx = batches.unsqueeze(1).expand(-1, K)[invalid]
            if forb_pts.numel() > 0:
                self.current_constructions[forb_bidx, forb_pts[:, 0], forb_pts[:, 1]] = -1

        return good

    # ------------------ enumerate empties per batch
    def possible_additions(self, shuffle: bool = False) -> torch.Tensor:
        # returns (B, max_candidates, 2) filled with (-1,-1) when fewer
        nz = (self.current_constructions == 0).nonzero(as_tuple=False).to(torch.int)  # (M, 3): (b, x, y)
        if nz.shape[0] == 0:
            return torch.empty((self.batch_size, 0, 2), device=self.device, dtype=torch.int8)

        if shuffle:
            nz = nz[torch.randperm(nz.size(0), device=self.device)]
            nz = nz[torch.argsort(nz[:, 0])]  # keep grouped by batch

        counts = torch.bincount(nz[:, 0], minlength=self.batch_size)
        max_count = int(counts.max().item())

        result = -torch.ones((self.batch_size, max_count, 2), dtype=torch.int8, device=self.device)
        # fill per-batch slices
        start = 0
        for b in range(self.batch_size):
            c = int(counts[b].item())
            if c == 0:
                continue
            slice_b = nz[start:start + c][:, 1:]  # (c, 2) -> (x,y)
            result[b, :c] = slice_b.to(torch.int8)
            start += c
        return result

    # ------------------ propose one candidate per batch via chunked scan
    def propose_additions_batched(self) -> torch.Tensor:
        proposals = -torch.ones((self.batch_size, 2), dtype=torch.int8, device=self.device)
        candidates = self.possible_additions(shuffle=True)
        if candidates.shape[1] == 0:
            return proposals

        live = (candidates[:, 0] != self.null_tensor).any(dim=-1)
        step = 10

        for k in range(0, candidates.shape[1], step):
            cand_slice = candidates[:, k:k+step]                    # (B, s, 2)
            possible = self.check_new_points(cand_slice)            # (B, s)
            any_true = possible.any(dim=1) & live
            if any_true.any():
                # choose first True in the slice for those batches
                first_idx = possible[any_true].float().argmax(dim=1)  # safe because any_true==True
                b_idx = torch.arange(self.batch_size, device=self.device)[any_true]
                proposals[b_idx] = cand_slice[any_true, first_idx, :]

            # batches that successfully proposed now become inactive
            live = live & ~any_true
            if not live.any():
                break

        return proposals

    # ------------------ saturation (with trajectory)
    @torch.no_grad()
    def saturate(self):
        self.history_buffer = [self.current_constructions.clone().cpu()]
        console.log(f"[saturate] Start: B={self.batch_size}, N={self.N}, max_points={self.max_points}")

        for _ in range(self.max_points):
            before = self.current_counts.clone()
            proposals = self.propose_additions_batched()
            self.add_points(proposals)
            changed = self.current_counts > before
            if torch.any(changed):
                self.history_buffer.append(self.current_constructions.clone().cpu())
                console.log(f"[saturate] Step {len(self.history_buffer)-1}: changed={int(changed.sum().item())}")
            else:
                console.log("[saturate] No changes this step – saturation reached.")
                break
        console.log(f"[saturate] Done: steps={len(self.history_buffer)-1}")

    # ------------------ vectorized forbidden-refresh (complete, slower)
    def update_forbidden_squares(self, grids: torch.Tensor) -> torch.Tensor:
        point_counts = (grids == 1).sum(dim=(1, 2))
        unique_counts = torch.unique(point_counts)

        for count in unique_counts:
            if count < 2:
                continue
            mask = (point_counts == count)
            if not mask.any():
                continue

            idxs = mask.nonzero(as_tuple=True)[0]
            group = grids[idxs]

            nz = (group == 1).nonzero(as_tuple=False)
            if nz.numel() == 0:
                continue
            existing = nz[:, 1:].reshape(group.shape[0], int(count.item()), 2)

            pairs = torch.combinations(torch.arange(int(count.item()), device=grids.device), r=2)
            p1s = existing[:, pairs[:, 0]]  # (g, P, 2)
            p2s = existing[:, pairs[:, 1]]  # (g, P, 2)

            for i, gidx in enumerate(idxs):
                empties = (grids[gidx] == 0).nonzero(as_tuple=False)
                if empties.shape[0] == 0:
                    continue

                x1 = p1s[i, :, 0].unsqueeze(1); y1 = p1s[i, :, 1].unsqueeze(1)
                x2 = p2s[i, :, 0].unsqueeze(1); y2 = p2s[i, :, 1].unsqueeze(1)
                x3 = empties[:, 0].unsqueeze(0); y3 = empties[:, 1].unsqueeze(0)

                col = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                is_col = (col == 0).any(dim=0)
                forb = empties[is_col]
                if forb.numel() > 0:
                    grids[gidx, forb[:, 0], forb[:, 1]] = -1
        return grids

    # ------------------ helpers used by greedy baselines (kept for completeness)
    def add_points_to_grid(self, grids: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        points = points.to(torch.int)
        non_null = (points != self.null_tensor).any(dim=-1)
        if not non_null.any():
            return grids

        bidx = torch.arange(grids.shape[0], device=self.device)[non_null]
        pts = points[non_null]
        gsel = grids[non_null]

        x, y = pts[:, 0], pts[:, 1]
        in_bounds = (x >= 0) & (x < self.N) & (y >= 0) & (y < self.N)
        if not in_bounds.any():
            return grids

        avail = (gsel[torch.arange(gsel.shape[0]), x, y] == 0)
        ok = in_bounds & avail
        if not ok.any():
            return grids

        bidx = bidx[ok]; pts = pts[ok]
        grids[bidx, pts[:, 0], pts[:, 1]] = 1
        return grids

    def available_spaces(self, grid: torch.Tensor) -> torch.Tensor:
        return (grid == 0).nonzero(as_tuple=False).to(torch.int)

    def best_grid(self, grids: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        upd = self.add_points_to_grid(grids, points)
        upd = self.update_forbidden_squares(upd)
        scores = (upd == 0).sum(dim=(1, 2))
        max_score = torch.max(scores)
        best = (scores == max_score).nonzero(as_tuple=True)[0]
        choice = best[torch.randint(0, best.shape[0], (1,)).item()]
        return upd[choice]

    def try_to_add_points(self, points: torch.Tensor):
        can_add = self.check_new_points(points.unsqueeze(1)).squeeze(1)
        points = points.clone()
        points[~can_add] = self.null_tensor
        self.add_points(points)


# ============================================
# Pretty-printing for debug
# ============================================

def print_grid(construction_grid: torch.Tensor, title="Construction"):
    """Text grid: X=point, -=forbidden, .=empty (interprets first axis as rows for printing only)."""
    grid = construction_grid.detach().cpu().numpy()
    N = grid.shape[0]
    print(f"\n{title}")
    for r in range(N):
        row = []
        for c in range(N):
            v = grid[r, c]
            row.append('X' if v == 1 else '-' if v == -1 else '.')
        print(' '.join(row))


# ============================================
# Save dataset (HDF5)
# ============================================

def save_dataset_h5(save_path: str, dataset_entries, triplets,
                    n: int, save_triplets_mode: str = "xy0"):
    """
    Save an HDF5 file with:
      /n            :: Int32
      /triplets     :: Int32[:,6]  (xy0 or rowcol1 depending on mode)
      /dataset/<k>/ initial_grid   Float32[n,n]
                    target_grid    Float32[n,n]
                    mask_top2n     Float32[n,n]
                    score          Float32
                    num_points     Int32
                    num_violations Int32
                    deficit        Int32
                    n              Int32
    save_triplets_mode:
      - "xy0"     => 0-based (x,y) triplets (Python-native)
      - "rowcol1" => 1-based (row,col) triplets (Julia-friendly)
    """
    import h5py
    if save_triplets_mode == "rowcol1":
        trip_arr = triplets_xy0_to_rowcol1(triplets)  # np.int32
    else:
        trip_arr = np.asarray(triplets, dtype=np.int32)

    with h5py.File(save_path, "w") as f:
        f.create_dataset("n", data=np.int32(n))
        f.create_dataset("triplets", data=trip_arr, dtype=np.int32)
        ds = f.create_group("dataset")
        for idx, entry in enumerate(dataset_entries):
            g = ds.create_group(str(idx))
            for k, v in entry.items():
                if isinstance(v, np.ndarray):
                    g.create_dataset(k, data=v, dtype=v.dtype)
                else:
                    g.create_dataset(k, data=np.asarray(v))


# ============================================
# High-level generator
# ============================================

def generate_training_file(n: int, batch_size: int, save_path: str,
                           seed: Optional[int] = 42,
                           save_triplets_mode: str = "rowcol1",
                           post_add_full_update: bool = False,
                           aggressive_blocking: bool = True):
    """
    Build a dataset via saturation and save to HDF5.
    - save_triplets_mode: "rowcol1" (Julia) or "xy0" (Python)
    - post_add_full_update: add thorough pruning after each add (slower)
    - aggressive_blocking: block whole row/col/diagonals after ≥2 points (cheap heuristic)
    """
    console.log(f"[generate_training_file] n={n}, batch_size={batch_size}, save_path={save_path}")
    triplets_xy0 = generate_triplets_xy0(n)
    console.log(f"[generate_training_file] Triplets (xy0): {len(triplets_xy0):,}")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"[generate_training_file] Device: {device}")

    gen = NoThreeInLine(
        batch_size=batch_size,
        grid_size=n,
        max_points=2*n,
        device=device,
        aggressive_blocking=aggressive_blocking,
        post_add_full_update=post_add_full_update,
    )

    t0 = time.time(); gen.saturate(); t1 = time.time()
    console.log(f"[generate_training_file] Saturation took {t1 - t0:.2f}s")

    if not gen.history_buffer:
        raise RuntimeError("history_buffer empty (saturate() didn't record states).")

    console.log("[build_dataset] Stacking history buffer")
    history = torch.stack(gen.history_buffer)  # (T, B, n, n)
    T, B = history.shape[0], history.shape[1]
    console.log(f"[build_dataset] History shape: T={T}, B={B}, n={n}")
    dataset_entries = []

    # Build one entry per final board in batch
    for b in range(B):
        point_counts = (history[:, b] == 1).sum(dim=(1, 2))
        diffs = torch.diff(point_counts, prepend=point_counts[:1])
        last_change_idx = int((diffs != 0).nonzero(as_tuple=True)[0][-1].item())

        # early snapshot at ~30% of trajectory
        snap_idx = max(0, int(last_change_idx * 0.3))
        early_snapshot = history[snap_idx, b]
        final_snapshot = history[last_change_idx, b]

        init_grid_soft = to_soft_grid(early_snapshot)
        final_grid_bin = to_binary_grid(final_snapshot)
        mask_top2n = make_top2n_mask(final_grid_bin.clone(), n)
        labels = compute_labels(final_grid_bin.clone(), n, triplets_xy0)

        entry = {
            "initial_grid": init_grid_soft.numpy().astype("float32"),
            "target_grid":  final_grid_bin.numpy().astype("float32"),
            "mask_top2n":   mask_top2n.numpy().astype("float32"),
            "score":        labels["score"],
            "num_points":   labels["num_points"],
            "num_violations": labels["num_violations"],
            "deficit":      labels["deficit"],
            "n":            np.int32(n),
        }
        dataset_entries.append(entry)

    t2 = time.time()
    console.log(f"[build_dataset] Built {len(dataset_entries)} entries in {t2 - t1:.2f}s")

    save_dataset_h5(save_path, dataset_entries, triplets_xy0, n, save_triplets_mode=save_triplets_mode)
    console.log(f"[generate_training_file] Saved to {save_path}")
    return save_path


# ============================================
# Small tensor helpers (outside class)
# ============================================

def to_soft_grid(grid: torch.Tensor, noise_level: float = 0.3) -> torch.Tensor:
    """{1,0,-1} -> Float32 soft grid: 1->1.0, 0->U[0,noise], -1->0.0"""
    g = grid.detach().to('cpu')
    soft = torch.zeros_like(g, dtype=torch.float32)
    soft[g == 1] = 1.0
    zero_mask = (g == 0)
    if zero_mask.any():
        soft[zero_mask] = torch.rand_like(soft[zero_mask]) * noise_level
    return soft

def to_binary_grid(grid: torch.Tensor) -> torch.Tensor:
    g = grid.detach().to('cpu')
    out = torch.zeros_like(g, dtype=torch.float32)
    out[g == 1] = 1.0
    return out


# ============================================
# CLI test
# ============================================

if __name__ == "__main__":
    N = 10
    B = 1024
    solver = NoThreeInLine(batch_size=B, grid_size=N, max_points=2*N,
                           aggressive_blocking=True, post_add_full_update=False,
                           device="mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time(); solver.saturate(); t1 = time.time()
    print(f"Saturation took {t1-t0:.2f}s.")

    counts = Counter(solver.current_counts.tolist())
    print(counts)

    best_idx = int(torch.argmax(solver.current_counts).item())
    best_grid = solver.current_constructions[best_idx]
    best_count = int(solver.current_counts[best_idx].item())
    print(f"\nBest construction has {best_count} points:")
    print_grid(best_grid, f"Best Grid ({best_count} points)")
