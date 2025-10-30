import dataclasses
from collections import Counter
import torch
import time
import numpy as np
import itertools
import h5py
import random
from rich.console import Console

console = Console()

@dataclasses.dataclass
class NoThreeInLine:
    """
    No-three-in-line.
    """
    batch_size: int
    grid_size: int  
    max_points: int
    device: str = "cpu"
    aggressive_blocking: bool = False  # If True, blocks entire lines when points are placed
    
    def __post_init__(self):
        N = self.N = self.grid_size
        assert self.grid_size < 64, "Grid size must fit in int8 coordinate."
        assert self.max_points < 2*N + 1, "max_points should be less than or equal to 2*N + 1."

        # self.current_constructions: An (N, N) grid for each batch item.
        # Value `1` marks a placed point, `0` is empty, `-1` is forbidden. This is for fast
        # checking of whether a specific coordinate is occupied.
        self.current_constructions = torch.zeros(
            (self.batch_size, N, N),
            dtype=torch.int8,
            device=self.device,
        )

        # self.points_list: A list of coordinates for each batch item.
        # Stores up to `max_points` (x, y) pairs. Unused slots are `[-1, -1]`.
        # This is for efficient retrieval of all points to check for collinearity.
        self.points_list = torch.full(
            (self.batch_size, self.max_points, 2),
            -1,
            dtype=torch.int8,  # N can be < 64
            device=self.device,
        )

        # self.current_counts: Tracks the number of points in each construction.
        self.current_counts = torch.zeros(
            (self.batch_size,),
            dtype=torch.int8,  # max_points can be <= 2*N
            device=self.device,
        )

        # self.null_tensor: A placeholder for an invalid or unused point.
        self.null_tensor = torch.tensor(
            [-1, -1],
            dtype=torch.int8,
            device=self.device,
        )

        # cache for pair indices used in collinearity checks
        self._pair_cache = {}
        # For recording full board trajectories across saturation steps
        self.history_buffer = []

    def _block_entire_lines(self, batch_indices, new_points):
        """
        Block entire lines (rows, columns, diagonals) only when there are already 
        two points in that line, preventing three-in-line violations.
        
        Args:
            batch_indices: 1D tensor of batch indices that were modified
            new_points: 2D tensor (len(batch_indices), 2) of points that were just added
        """
        if batch_indices.numel() == 0:
            return
            
        for i, batch_idx in enumerate(batch_indices):
            x, y = new_points[i]
            x, y = x.item(), y.item()
            
            # Check row - count existing points
            row_count = torch.sum(self.current_constructions[batch_idx, x, :] == 1).item()
            if row_count >= 2:  # Block entire row if 2+ points already exist
                for col in range(self.N):
                    if self.current_constructions[batch_idx, x, col] == 0:
                        self.current_constructions[batch_idx, x, col] = -1
            
            # Check column - count existing points
            col_count = torch.sum(self.current_constructions[batch_idx, :, y] == 1).item()
            if col_count >= 2:  # Block entire column if 2+ points already exist
                for row in range(self.N):
                    if self.current_constructions[batch_idx, row, y] == 0:
                        self.current_constructions[batch_idx, row, y] = -1
            
            # Check main diagonal (top-left to bottom-right)
            # Find the starting point of the diagonal
            start_x, start_y = x, y
            while start_x > 0 and start_y > 0:
                start_x -= 1
                start_y -= 1
            
            # Count points on this diagonal
            diag_count = 0
            curr_x, curr_y = start_x, start_y
            while curr_x < self.N and curr_y < self.N:
                if self.current_constructions[batch_idx, curr_x, curr_y] == 1:
                    diag_count += 1
                curr_x += 1
                curr_y += 1
            
            # Block the entire diagonal if 2+ points already exist
            if diag_count >= 2:
                curr_x, curr_y = start_x, start_y
                while curr_x < self.N and curr_y < self.N:
                    if self.current_constructions[batch_idx, curr_x, curr_y] == 0:
                        self.current_constructions[batch_idx, curr_x, curr_y] = -1
                    curr_x += 1
                    curr_y += 1
            
            # Check anti-diagonal (top-right to bottom-left)
            # Find the starting point of the anti-diagonal
            start_x, start_y = x, y
            while start_x > 0 and start_y < self.N - 1:
                start_x -= 1
                start_y += 1
            
            # Count points on this anti-diagonal
            anti_diag_count = 0
            curr_x, curr_y = start_x, start_y
            while curr_x < self.N and curr_y >= 0:
                if self.current_constructions[batch_idx, curr_x, curr_y] == 1:
                    anti_diag_count += 1
                curr_x += 1
                curr_y -= 1
            
            # Block the entire anti-diagonal if 2+ points already exist
            if anti_diag_count >= 2:
                curr_x, curr_y = start_x, start_y
                while curr_x < self.N and curr_y >= 0:
                    if self.current_constructions[batch_idx, curr_x, curr_y] == 0:
                        self.current_constructions[batch_idx, curr_x, curr_y] = -1
                    curr_x += 1
                    curr_y -= 1

    def _update_forbidden_squares_after_add(self, batch_indices, new_points, current_counts):
        """
        After adding new points, update the constructions to mark newly forbidden squares.
        
        This is called from `add_points` after points have been placed on the grid.
        This is an optimized check that only considers lines formed with the new points.
        
        Args:
            batch_indices: 1D tensor of indices for the batches that were modified.
            new_points: 2D tensor (len(batch_indices), 2) of the points that were just added.
            current_counts: 1D tensor of point counts for the modified batches *before* the new point was added.
        """
        if batch_indices.numel() == 0:
            return

        # We only need to check for new lines with existing points.
        # This requires at least one existing point, so the new count would be >= 2.
        # This means the old count was >= 1.
        valid_mask = current_counts >= 1
        if not valid_mask.any():
            return
            
        # Filter to only the batches that have other points to form a line with
        batch_indices = batch_indices[valid_mask]
        new_points = new_points[valid_mask]
        current_counts = current_counts[valid_mask]

        if batch_indices.numel() == 0:
            return

        # For each modified batch, check the new point against all existing points
        existing_points_full = self.points_list[batch_indices] # (num_modified, max_points, 2)
        
        # Create a mask to select only the valid, existing points for each batch
        counts_expanded = torch.arange(self.max_points, device=self.device).expand(len(current_counts), -1)
        existing_points_mask = counts_expanded < current_counts.unsqueeze(1)

        # p1 is the new point, broadcasted
        p1 = new_points.unsqueeze(1)
        # p2 are all existing points
        p2 = existing_points_full

        # The third point on the line defined by p1 and p2
        p3 = 2 * p2 - p1
        
        # Create a mask to filter out invalid/empty point slots from p2 and resulting p3
        # We only care about p3's that are formed with valid p2's
        p3_valid_mask = existing_points_mask

        # Filter for p3's that are within the grid boundaries
        in_bounds_mask = (p3 >= 0).all(dim=-1) & (p3 < self.N).all(dim=-1)
        p3_valid_mask &= in_bounds_mask
        
        # Further filter: only consider p3's that land on an empty square
        p3_coords_to_check = p3[p3_valid_mask].to(torch.int)
        
        # We need to map these back to their batch indices
        batch_indices_expanded = batch_indices.unsqueeze(1).expand(-1, self.max_points)
        p3_batch_indices = batch_indices_expanded[p3_valid_mask]

        if p3_coords_to_check.numel() > 0:
            occupancy = self.current_constructions[p3_batch_indices, p3_coords_to_check[:, 0], p3_coords_to_check[:, 1]]
            is_empty_mask = (occupancy == 0)
            
            # Final set of points to mark as forbidden
            forbidden_points = p3_coords_to_check[is_empty_mask]
            forbidden_batch_indices = p3_batch_indices[is_empty_mask]

            if forbidden_points.numel() > 0:
                self.current_constructions[forbidden_batch_indices, forbidden_points[:, 0], forbidden_points[:, 1]] = -1

    def add_points(self, points):
        """
        Add points to constructions, updating internal state.
        
        Flow:
        1. Identify valid points (non-null, within bounds, and on available squares).
        2. Group batches by their current number of points.
        3. In a vectorized way, for each group, update both the grid and point list.
        4. Increment point counts for batches where a point was added.
        5. Proactively mark new forbidden squares resulting from the additions.
        
        Args:
            points: tensor of shape (batch_size, 2) with (x, y) coordinates.
        """
        points = points.to(dtype=torch.int8, device=self.device)
        points_int = points.to(dtype=torch.int)  # For indexing

        # Create a mask for non-null points (i.e., rows that really add something)
        non_null_mask = (points != self.null_tensor).any(dim=-1)

        # Assertions for safety
        assert points.shape[0] == self.batch_size
        if non_null_mask.any():
            assert torch.max(self.current_counts[non_null_mask]).item() < self.max_points
        assert (-1 <= points).all() and (points < self.N).all()

        # Get indices for all batches and for only non-null ones
        batch_indices_all = torch.arange(self.batch_size, device=self.device)
        valid_batch_indices = batch_indices_all[non_null_mask]
        
        # If no valid points, exit early
        if valid_batch_indices.numel() == 0:
            return

        # Check occupancy for valid points
        coords_to_check = points_int[non_null_mask]
        occupancy_status = self.current_constructions[valid_batch_indices, coords_to_check[:, 0], coords_to_check[:, 1]]
        is_available = (occupancy_status == 0)

        # Final mask of batches where a point should be added
        final_add_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        # Place `True` at the indices of batches that had a valid point and an available square
        final_add_mask[valid_batch_indices[is_available]] = True

        added_point_counts = torch.zeros_like(self.current_counts)

        # Keep track of which batches get insertions for the forbidden square update
        all_batch_insertion_indices = []
        all_points_to_add = []
        all_initial_counts = []

        # Loop over unique counts for batched updates
        for cur_count in torch.unique(self.current_counts).tolist():
            count_mask = (self.current_counts == cur_count)
            # Find the intersection of points to add and batches with the current count
            batch_insertions_mask = count_mask & final_add_mask
            
            if not batch_insertions_mask.any():
                continue

            # Get the indices and coordinates of points to add for this group
            batch_insertion_indices = batch_indices_all[batch_insertions_mask]
            points_to_add = points[batch_insertions_mask]
            points_to_add_int = points_int[batch_insertions_mask]

            # Update construction grid
            self.current_constructions[batch_insertion_indices, points_to_add_int[:, 0], points_to_add_int[:, 1]] = 1
            
            # Update points list
            self.points_list[batch_insertion_indices, cur_count] = points_to_add
            
            # Mark that we added a point to these batches
            added_point_counts[batch_insertion_indices] = 1

            # Store the info needed for the forbidden square update
            all_batch_insertion_indices.append(batch_insertion_indices)
            all_points_to_add.append(points_to_add)
            all_initial_counts.append(torch.full_like(batch_insertion_indices, cur_count, device=self.device))


        # After all points are added, update the forbidden squares
        if all_batch_insertion_indices:
            batch_indices_cat = torch.cat(all_batch_insertion_indices)
            new_points_cat = torch.cat(all_points_to_add)
            
            if self.aggressive_blocking:
                # Use the new aggressive method (entire lines blocked)
                self._block_entire_lines(batch_indices_cat, new_points_cat)
            else:
                # Use the original method (individual forbidden squares)
                self._update_forbidden_squares_after_add(
                    batch_indices_cat,
                    new_points_cat,
                    torch.cat(all_initial_counts)
                )

        # Update the total counts
        self.current_counts += added_point_counts

    def check_new_points(self, new_points):
        """
        Check which points can be added without creating three-in-a-line.
        
        This implementation is fully vectorized to process all constructions in parallel.
        
        Args:
            new_points: tensor of shape (batch_size, num_candidates, 2) with candidate points.
            verbose: whether to print collinearity info.
        Returns: 
            A boolean mask of shape (batch_size, num_candidates).
        """
        
        num_candidates = new_points.shape[1]

        # 1. Basic validation (vectorized)
        # Check for null points
        good_bools = (new_points != self.null_tensor).any(dim=-1)

        # Check bounds
        in_bounds = (new_points >= 0).all(dim=-1) & (new_points < self.N).all(dim=-1)
        good_bools &= in_bounds
        
        # Check occupancy
        # Create indices for gathering
        batch_indices_int = torch.arange(self.batch_size, device=self.device).unsqueeze(1).expand(-1, num_candidates)
        
        # We only check valid points to avoid index errors
        points_to_check = new_points[good_bools].to(torch.int)
        batch_indices_to_check = batch_indices_int[good_bools]
        
        # Gather occupancy status
        occupancy_status = self.current_constructions[batch_indices_to_check, points_to_check[:, 0], points_to_check[:, 1]]
        
        # Create a full occupancy mask and update it
        is_occupied = torch.zeros_like(good_bools)
        is_occupied[good_bools] = (occupancy_status != 0)
        good_bools &= ~is_occupied

        # 2. Collinearity check (vectorized)
        # Only iterate over counts that actually occur (>1)
        unique_counts = torch.unique(self.current_counts)

        for cur_count in unique_counts.tolist():
            if cur_count < 2:
                continue
            # Find all constructions that currently have `cur_count` points
            batch_mask = (self.current_counts == cur_count)
            if not batch_mask.any():
                continue

            # Get the indices of the relevant batches
            batch_indices = batch_mask.nonzero(as_tuple=True)[0]
            
            # Get existing points and candidates for these batches
            existing_points = self.points_list[batch_indices] # (num_batches, max_points, 2)
            candidates = new_points[batch_indices]            # (num_batches, num_candidates, 2)
            
            # Retrieve or compute cached pair indices for this cur_count
            if cur_count not in self._pair_cache:
                self._pair_cache[cur_count] = torch.combinations(torch.arange(cur_count), r=2)
            pair_indices = self._pair_cache[cur_count].to(self.device)  # (num_pairs, 2)
            p1s = existing_points[:, pair_indices[:, 0]] # (num_batches, num_pairs, 2)
            p2s = existing_points[:, pair_indices[:, 1]] # (num_batches, num_pairs, 2)

            # Reshape for broadcasting
            # p1s/p2s: (num_batches, num_pairs, 1, 2)
            # candidates: (num_batches, 1, num_candidates, 2)
            p1s = p1s.unsqueeze(2)
            p2s = p2s.unsqueeze(2)
            p3s = candidates.unsqueeze(1)

            # Extract coordinates for broadcasting
            x1, y1 = p1s[..., 0], p1s[..., 1] # (num_batches, num_pairs, 1)
            x2, y2 = p2s[..., 0], p2s[..., 1] # (num_batches, num_pairs, 1)
            x3, y3 = p3s[..., 0], p3s[..., 1] # (num_batches, 1, num_candidates)

            # Perform the collinearity check for all pairs and candidates at once
            # Result shape: (num_batches, num_pairs, num_candidates)
            collinearity_check = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            
            # A candidate is invalid if ANY pair is collinear with it
            is_collinear = (collinearity_check == 0)
            invalid_candidates = is_collinear.any(dim=1) # (num_batches, num_candidates)
            
            # Update the good_bools mask for the relevant batches
            good_bools[batch_indices] &= ~invalid_candidates
            
            # Update forbidden positions
            forbidden_points = new_points[batch_indices][invalid_candidates].to(torch.int)
            forbidden_batch_indices = batch_indices.unsqueeze(1).expand(-1, num_candidates)[invalid_candidates]
            
            if forbidden_points.numel() > 0:
                self.current_constructions[forbidden_batch_indices, forbidden_points[:, 0], forbidden_points[:, 1]] = -1

        return good_bools


    def possible_additions(self, shuffle=False):
        """
        Return a tensor of shape (batch_size, k, 2) indicating coordinates of possible additions.
        (Here k is the maximum over all batches of allowable points added.)
        
        Args:
            shuffle: whether to randomize order of candidates
        Returns:
            tensor of shape (batch_size, max_candidates, 2) with coordinates
        """
        
        possible_additions = (self.current_constructions == 0).nonzero(as_tuple=False).to(torch.int)

        if shuffle:
            indices = torch.randperm(possible_additions.size(0))
            possible_additions = possible_additions[indices]
            sorted_indices = torch.argsort(possible_additions[:,0])
            possible_additions = possible_additions[sorted_indices]

        if possible_additions.shape[0] > 0:
            max_newpoints = torch.max(torch.bincount(possible_additions[:,0])) 
        else:
            return torch.empty((self.batch_size,0,2),device=self.device)

        unique_non_zero_first_coords, nonzero_counts = torch.unique(possible_additions[:,0], return_counts=True)

        max_count = nonzero_counts.max().item()

        nonzero_counts = nonzero_counts.to(torch.int)

        counts = torch.zeros((self.batch_size,),dtype=torch.int,device=self.device)
        counts[unique_non_zero_first_coords] = nonzero_counts

        mask = torch.arange(max_count,device=self.device).expand(self.batch_size, max_count) < counts.unsqueeze(1)
        result_tensor = -1 * torch.ones((self.batch_size, max_count, 2),dtype=torch.int8,device=self.device)
        result_tensor[mask] = possible_additions[:, 1:].to(torch.int8)

        return result_tensor

    def propose_additions_batched(self):
        """Propose additions by batching over new possibilities"""
        current_proposals = -1*torch.ones((self.batch_size,2),dtype=torch.int8,device=self.device)

        all_possible_additions = self.possible_additions(shuffle=True)

        if all_possible_additions.shape[1] == 0:
            return current_proposals

        live_batches = (all_possible_additions[:,0] != self.null_tensor).any(dim=-1) # shape (B,)

        sB = 10

        for k in range(0,all_possible_additions.shape[1],sB):

            possible = self.check_new_points(all_possible_additions[:,k:k+sB]) # shape (B,sB)
            batch_fill = torch.arange(self.batch_size,device=self.device).unsqueeze(1).expand(possible.shape)

            if possible.any():
                indices = torch.cat((torch.tensor([0],device=self.device), (torch.diff(batch_fill[possible]) != 0).nonzero(as_tuple=True)[0] + 1))
                current_proposals[batch_fill[possible][indices]] = all_possible_additions[:,k:k+sB][possible][indices]

            successful_batches = possible.any(dim=-1)                          # shape B

            new_successes = torch.logical_and(live_batches,successful_batches).nonzero()

            live_batches[new_successes] = False

            if not live_batches.any():
                break

        return current_proposals
    
    @torch.no_grad()
    def saturate(self):
        """
        Complete all constructions randomly *and* record full trajectories.
        The initial empty state is saved as step 0, and every subsequent step
        that successfully changes **any** board appends a copy of the full batch
        state to ``self.history_buffer``.
        """
        # Reset / initialize history buffer with the all-zero initial state
        self.history_buffer = [self.current_constructions.clone().cpu()]
        console.log(f"[saturate] Start: batch_size={self.batch_size}, n={self.N}, max_points={self.max_points}")

        for _ in range(self.max_points):
            before_counts = self.current_counts.clone()
            proposals = self.propose_additions_batched()
            self.add_points(proposals)

            # Detect if *any* board added at least one new point this step
            changed_mask = self.current_counts > before_counts
            if torch.any(changed_mask):
                self.history_buffer.append(self.current_constructions.clone().cpu())
                console.log(f"[saturate] Step {len(self.history_buffer)-1}: changed={int(changed_mask.sum().item())}")
            else:
                # No boards changed => fully saturated
                console.log("[saturate] No changes this step – saturation reached.")
                break
        console.log(f"[saturate] Done: steps={len(self.history_buffer)-1}")

    @torch.no_grad()
    def greedy_saturate(self):
        """
        Complete all constructions greedily until addition of any more points is impossible.
        """
        for i in range(self.batch_size):
            current_grid = self.current_constructions[i].clone()

            while True:
                # Find all empty spaces on the current grid
                possible_points = self.available_spaces(current_grid)
                
                if possible_points.shape[0] == 0:
                    break  # No more empty spaces
                
                # Set up the inputs for best_grid: one grid for each possible point
                num_candidates = possible_points.shape[0]
                expanded_grids = current_grid.unsqueeze(0).repeat(num_candidates, 1, 1)

                # Find the best grid after trying all possible single-point additions
                best_next_grid = self.best_grid(expanded_grids, possible_points)
            
                # Check if we made progress
                if (best_next_grid == 1).sum() > (current_grid == 1).sum():
                    current_grid = best_next_grid
                    # After adding a point, update the grid to mark any newly formed forbidden squares
                    # We unsqueeze to add a temporary batch dimension for the vectorized function
                    current_grid = self.update_forbidden_squares(current_grid.unsqueeze(0)).squeeze(0)
                else:
                    # No valid point could be added to improve the score, so we're done
                    break
            
            # Update the final construction and its point count
            self.current_constructions[i] = current_grid
            self.current_counts[i] = (current_grid == 1).sum().item()

    @torch.no_grad()
    def greedy_saturate_batched(self):
        """
        Complete all constructions greedily using vectorized operations.
        Exactly mirrors greedy_saturate() but processes all grids simultaneously.
        """
        # Track which grids are still active (have possible moves)
        active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        
        while active_mask.any():
            # Find active grids and collect all their candidates
            all_expanded_grids = []
            all_possible_points = []
            grid_info = []  # (grid_idx, start_pos, num_candidates)
            
            current_pos = 0
            for grid_idx in range(self.batch_size):
                if not active_mask[grid_idx]:
                    continue
                    
                current_grid = self.current_constructions[grid_idx]
                
                # Find all empty spaces on the current grid
                possible_points = self.available_spaces(current_grid)
                
                if possible_points.shape[0] == 0:
                    # No more empty spaces for this grid
                    active_mask[grid_idx] = False
                    continue
                
                # Set up the inputs for best_grid: one grid for each possible point
                num_candidates = possible_points.shape[0]
                expanded_grids = current_grid.unsqueeze(0).repeat(num_candidates, 1, 1)
                
                # Store for batch processing
                all_expanded_grids.append(expanded_grids)
                all_possible_points.append(possible_points)
                grid_info.append((grid_idx, current_pos, num_candidates))
                current_pos += num_candidates
            
            if not all_expanded_grids:
                # No active grids left
                break
            
            # Concatenate all grids and points for batch processing
            batch_expanded_grids = torch.cat(all_expanded_grids, dim=0)
            batch_possible_points = torch.cat(all_possible_points, dim=0)
            
            # Process all candidates at once using existing logic
            batch_updated_grids = self.add_points_to_grid(batch_expanded_grids, batch_possible_points)
            batch_updated_grids = self.update_forbidden_squares(batch_updated_grids)
            
            # Score all updated grids
            batch_scores = (batch_updated_grids == 0).sum(dim=(1, 2))
            
            # For each grid, find its best candidate
            for grid_idx, start_pos, num_candidates in grid_info:
                # Extract this grid's candidates and scores
                end_pos = start_pos + num_candidates
                grid_scores = batch_scores[start_pos:end_pos]
                grid_updated_grids = batch_updated_grids[start_pos:end_pos]
                
                # Find the best grid (same logic as original best_grid method)
                max_score = torch.max(grid_scores)
                best_mask = (grid_scores == max_score)
                
                # Randomly choose among ties
                best_indices = best_mask.nonzero(as_tuple=True)[0]
                chosen_idx = best_indices[torch.randint(0, best_indices.shape[0], (1,)).item()]
                
                best_next_grid = grid_updated_grids[chosen_idx]
                original_grid = self.current_constructions[grid_idx]
                
                # Check if we made progress (same logic as original)
                if (best_next_grid == 1).sum() > (original_grid == 1).sum():
                    self.current_constructions[grid_idx] = best_next_grid
                else:
                    # No valid point could be added to improve the score, so this grid is done
                    active_mask[grid_idx] = False
        
        # Update final counts
        self.current_counts = (self.current_constructions == 1).sum(dim=(1, 2)).to(torch.int8)

    def update_forbidden_squares(self, grids):
        """
        For a batch of grids, this function updates each grid by marking newly 
        forbidden squares with -1. This is the batched version.
        """
        point_counts = (grids == 1).sum(dim=(1, 2))
        
        unique_counts = torch.unique(point_counts)

        for count in unique_counts:
            if count < 2:
                continue
                
            # Identify all grids in the batch that have `count` points
            count_mask = (point_counts == count)
            group_indices = count_mask.nonzero(as_tuple=True)[0]
            group_grids = grids[group_indices]
            
            # This part is tricky to vectorize fully because the number of empty
            # squares can differ for each grid in the group. We iterate through 
            # the grids within the group, which is still much better than iterating 
            # through the entire batch.
            
            # Get existing points for all grids in the group at once
            existing_nz = (group_grids == 1).nonzero(as_tuple=False)
            if existing_nz.numel() == 0: continue
            
            # Reshape to (num_group_grids, count, 2)
            existing_points = existing_nz[:, 1:].reshape(group_grids.shape[0], count.item(), 2)

            # Generate pairs of points; this is the same for every grid in the group
            pair_indices = torch.combinations(torch.arange(count.item(), device=self.device), r=2)
            p1s = existing_points[:, pair_indices[:, 0]]  # (num_group_grids, num_pairs, 2)
            p2s = existing_points[:, pair_indices[:, 1]]  # (num_group_grids, num_pairs, 2)
            
            # Iterate through the group to handle varying numbers of empty squares
            for i, original_grid_idx in enumerate(group_indices):
                grid_p1s = p1s[i]  # (num_pairs, 2)
                grid_p2s = p2s[i]  # (num_pairs, 2)
                
                empty_squares = (grids[original_grid_idx] == 0).nonzero(as_tuple=False)
                if empty_squares.shape[0] == 0:
                    continue
                
                # Reshape for broadcasting
                p3s = empty_squares.unsqueeze(0)  # (1, num_empty, 2)
                
                x1 = grid_p1s[:, 0].unsqueeze(1)
                y1 = grid_p1s[:, 1].unsqueeze(1)
                x2 = grid_p2s[:, 0].unsqueeze(1)
                y2 = grid_p2s[:, 1].unsqueeze(1)
                x3 = p3s[..., 0]
                y3 = p3s[..., 1]
                
                # Check for collinearity
                collinearity_check = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                is_collinear = (collinearity_check == 0).any(dim=0)
                forbidden_points = empty_squares[is_collinear]
                
                if forbidden_points.numel() > 0:
                    grids[original_grid_idx, forbidden_points[:, 0], forbidden_points[:, 1]] = -1
                    
        return grids

    def add_points_to_grid(self, grids, points):
        """
        Add points to grids simultaneously where point i is added to grid i.
        
        Args:
            grids: tensor of shape (batch_size, N, N) 
            points: tensor of shape (batch_size, 2) with coordinates to add
        
        Returns:
            new_grids: tensor of shape (batch_size, N, N) with points added where valid
        """
        points = points.to(torch.int)
        
        # 1. Initial Filtering (null, bounds, occupancy)
        non_null_mask = (points != self.null_tensor).any(dim=-1)
        if not non_null_mask.any(): return grids
        
        batch_indices = torch.arange(grids.shape[0], device=self.device)[non_null_mask]
        valid_points = points[non_null_mask]
        valid_grids = grids[non_null_mask]
        
        x, y = valid_points[:, 0], valid_points[:, 1]
        in_bounds = (x >= 0) & (x < self.N) & (y >= 0) & (y < self.N)
        if not in_bounds.any(): return grids
        
        available = (valid_grids[torch.arange(valid_grids.shape[0]), x, y] == 0)
        final_mask = in_bounds & available
        if not final_mask.any(): return grids

        # Apply final mask to get the set of grids and points we'll actually check
        batch_indices = batch_indices[final_mask]
        valid_points = valid_points[final_mask]
        valid_grids = valid_grids[final_mask]
        
        # 2. Group by point count and check collinearity
        point_counts = (valid_grids == 1).sum(dim=(1, 2))
        can_add_mask = torch.zeros_like(point_counts, dtype=torch.bool)

        unique_counts = torch.unique(point_counts)
        for count in unique_counts:
            # Find all grids in our valid set that have `count` points
            count_mask = (point_counts == count)
            current_indices = count_mask.nonzero(as_tuple=True)[0]
            
            # Case 1: Less than 2 points, no collinearity possible
            if count < 2:
                can_add_mask[current_indices] = True
                continue

            # Case 2: Check for collinearity
            group_grids = valid_grids[current_indices]
            group_candidates = valid_points[current_indices]

            # Efficiently extract existing points for the entire group
            nz = (group_grids == 1).nonzero(as_tuple=False)
            existing_points = nz[:, 1:].reshape(group_grids.shape[0], count.item(), 2)
            
            # Vectorized collinearity check (adapted from check_new_points)
            pair_indices = torch.combinations(torch.arange(count.item()), r=2)
            p1s = existing_points[:, pair_indices[:, 0]]
            p2s = existing_points[:, pair_indices[:, 1]]
            p3s = group_candidates.unsqueeze(1) # a.k.a. the candidate points

            x1, y1 = p1s[..., 0], p1s[..., 1]
            x2, y2 = p2s[..., 0], p2s[..., 1]
            x3, y3 = p3s[..., 0], p3s[..., 1]

            collinearity_check = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            
            # A candidate is invalid if ANY pair is collinear
            invalid_candidates = (collinearity_check == 0).any(dim=1)
            can_add_mask[current_indices] = ~invalid_candidates

        # 3. Add all valid points to their grids in a single operation
        if can_add_mask.any():
            points_to_add = valid_points[can_add_mask]
            indices_to_update = batch_indices[can_add_mask]
            grids[indices_to_update, points_to_add[:, 0], points_to_add[:, 1]] = 1
            
        return grids

    def available_spaces(self, grid):
        """
        Return the coordinates of all empty (value==0) squares in a single grid.
        """
        return (grid == 0).nonzero(as_tuple=False).to(torch.int)

    def best_grid(self, grids, points):
        """
        Given `grids` (a stack of identical grids) and `points` (distinct 
        candidate points), this function returns the grid that is "best" after
        adding its point. The "best" grid is the one with the most available
        (0) squares remaining after the point and all newly-forbidden squares
        have been accounted for.
        If multiple grids are tied for the best score, one is chosen randomly.
        """
        # 1. Get the resulting grids after attempting to add each point
        updated_grids = self.add_points_to_grid(grids, points)

        # 2. For each of these new grids, update their forbidden squares
        updated_grids = self.update_forbidden_squares(updated_grids)
        
        # 3. Score each grid by the number of available spaces left
        scores = (updated_grids == 0).sum(dim=(1, 2))
        
        # 4. Find all indices of grids with the highest score
        max_score = torch.max(scores)
        best_indices = (scores == max_score).nonzero(as_tuple=True)[0]
        
        # 5. Randomly choose one from the best candidates
        random_choice_idx = torch.randint(0, best_indices.shape[0], (1,)).item()
        best_idx = best_indices[random_choice_idx]
        
        # 6. Return the chosen best grid
        return updated_grids[best_idx]

    def try_to_add_points(self, points):
        """
        points is a tensor of shape (batch_size, 2)
        we add each point if it satisfies constraints
        """
        can_add = self.check_new_points(points.unsqueeze(1)).squeeze(1)
        points[~can_add] = self.null_tensor
        self.add_points(points)

    # ======================================================================
    #                      Dataset-building helper methods
    # ======================================================================

    def to_soft_grid(self, grid, noise_level: float = 0.3):
        """Convert an integer grid {1, 0, -1} into a Float32 **soft** grid.
        Rules:
        • 1 stays 1.0
        • 0 becomes Uniform[0, noise_level]
        • -1 becomes 0.0
        Returns a CPU ``torch.FloatTensor`` of shape (n, n).
        """
        grid_cpu = grid.to('cpu')
        soft = torch.zeros_like(grid_cpu, dtype=torch.float32)
        soft[grid_cpu == 1] = 1.0
        zero_mask = (grid_cpu == 0)
        if zero_mask.any():
            soft[zero_mask] = torch.rand_like(soft[zero_mask]) * noise_level
        return soft

    def to_binary_grid(self, grid):
        """Convert an integer grid {1, 0, -1} to a clean Float32 binary grid {0.0, 1.0}."""
        grid_cpu = grid.to('cpu')
        bin_grid = torch.zeros_like(grid_cpu, dtype=torch.float32)
        bin_grid[grid_cpu == 1] = 1.0
        return bin_grid

    def build_dataset(self, n: int, triplets, snapshot_fraction_low: float = 0.3, snapshot_fraction_high: float = 0.4):
        """Create a list of dataset dictionaries – one per board in the current batch.
        The implementation follows the detailed specification in the task prompt.
        """
        import numpy as _np  # local import to keep global namespace clean

        if not self.history_buffer:
            raise RuntimeError("history_buffer is empty – run saturate() first.")

        console.log("[build_dataset] Stacking history buffer")
        history = torch.stack(self.history_buffer)  # (T, B, n, n)
        T, B = history.shape[0], history.shape[1]
        console.log(f"[build_dataset] History shape: T={T}, B={B}, n={n}")
        dataset_entries = []

        for b in range(B):
            # Board-specific trajectory length: identify the last step where the board changed
            point_counts = (history[:, b] == 1).sum(dim=(1, 2))
            diffs = torch.diff(point_counts, prepend=point_counts[:1])
            last_change_idx = int((diffs != 0).nonzero(as_tuple=True)[0][-1].item())
            traj_len = last_change_idx + 1  # +1 because indices start at 0

            # Early snapshot between 30% and 40% of trajectory
            snap_idx = max(0, int(traj_len * snapshot_fraction_low))
            early_snapshot = history[snap_idx, b]
            final_snapshot = history[last_change_idx, b]

            init_grid_soft = self.to_soft_grid(early_snapshot)
            final_grid_bin = self.to_binary_grid(final_snapshot)
            mask_top2n = make_top2n_mask(final_grid_bin.clone(), n)
            labels = compute_labels(final_grid_bin.clone(), n, triplets)

            entry = {
                "initial_grid": init_grid_soft.numpy().astype('float32'),
                "target_grid": final_grid_bin.numpy().astype('float32'),
                "mask_top2n": mask_top2n.numpy().astype('float32'),
                "score": _np.float32(labels["score"]),
                "num_points": _np.int32(labels["num_points"]),
                "num_violations": _np.int32(labels["num_violations"]),
                "deficit": _np.int32(labels["deficit"]),
                "n": _np.int32(n),
            }
            dataset_entries.append(entry)

        console.log(f"[build_dataset] Built {len(dataset_entries)} entries")
        return dataset_entries


def print_grid(construction_grid, title="Construction"):
    """Prints a 2D text representation of a single construction."""
    grid_list = construction_grid.cpu().tolist()
    N = len(grid_list)
    
    print(f"\n{title}")
    for r in range(N):
        row_str = []
        for c in range(N):
            if grid_list[r][c] == 1:
                row_str.append('X')
            elif grid_list[r][c] == -1:
                row_str.append('-')
            else:
                row_str.append('.')
        print(' '.join(row_str))

# ======================================================================
#                           Stand-alone helpers
# ======================================================================

def generate_triplets(n: int):
    """Generate all unique collinear triplets within an n×n grid.
    Returns a list of 6-tuples (i1, j1, i2, j2, i3, j3).
    """
    coords = [(i, j) for i in range(n) for j in range(n)]
    triplets = []
    for idx1 in range(len(coords)):
        x1, y1 = coords[idx1]
        for idx2 in range(idx1 + 1, len(coords)):
            x2, y2 = coords[idx2]
            for idx3 in range(idx2 + 1, len(coords)):
                x3, y3 = coords[idx3]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    triplets.append((x1, y1, x2, y2, x3, y3))
    return triplets


def compute_labels(final_binary_grid: torch.Tensor, n: int, triplets):
    """Compute quality metrics and score for a saturated board."""
    grid_cpu = final_binary_grid.to('cpu')
    num_points = int(grid_cpu.sum().item())
    num_violations = 0
    if num_points >= 3:
        data_np = grid_cpu.numpy()
        for (i1, j1, i2, j2, i3, j3) in triplets:
            if data_np[i1, j1] == 1.0 and data_np[i2, j2] == 1.0 and data_np[i3, j3] == 1.0:
                num_violations += 1
    deficit = max(0, 2 * n - num_points)
    score = num_points - 100.0 * num_violations - 1.0 * deficit
    return {
        "num_points": num_points,
        "num_violations": num_violations,
        "deficit": deficit,
        "score": float(score),
    }


def make_top2n_mask(final_binary_grid: torch.Tensor, n: int):
    """Return a Float32 mask with exactly 2n ones corresponding to the strongest cells."""
    flat = final_binary_grid.flatten()
    flat_np = flat.cpu().numpy()
    top_idx = np.argsort(-flat_np, kind='mergesort')[: 2 * n]
    mask = np.zeros_like(flat_np, dtype=np.float32)
    mask[top_idx] = 1.0
    return torch.from_numpy(mask.reshape(n, n))


def save_dataset_jld2(save_path: str, dataset_entries, triplets, n: int):
    """Save a Julia-friendly .jld2 file with exact schema:
    Layout
    ├─ /n          :: Int32 scalar
    ├─ /triplets   :: Int32[:,6]
    └─ /dataset    :: group
         ├─ "0"    :: group → sample-0
         ├─ "1"    :: group → sample-1
         └─ …

    Each sample group contains
        initial_grid, target_grid, mask_top2n  (Float32 n×n)
        score                                   Float32 scalar
        num_points, num_violations, deficit, n  Int32  scalar
    """
    import h5py

    trip_arr = np.asarray(triplets, dtype=np.int32)

    with h5py.File(save_path, "w") as f:
        # root datasets
        f.create_dataset("n", data=np.int32(n))
        f.create_dataset("triplets", data=trip_arr, dtype=np.int32)

        # per-sample data
        ds_grp = f.create_group("dataset")
        for idx, entry in enumerate(dataset_entries):
            g = ds_grp.create_group(str(idx))
            for key, val in entry.items():
                if isinstance(val, np.ndarray):
                    g.create_dataset(key, data=val, dtype=val.dtype)
                else:
                    g.create_dataset(key, data=np.asarray(val))


def generate_training_file(n: int, batch_size: int, save_path: str):
    """High-level convenience wrapper to generate and save training data."""
    console.log(f"[generate_training_file] n={n}, batch_size={batch_size}, save_path={save_path}")
    triplets = generate_triplets(n)
    console.log(f"[generate_training_file] Triplets generated: {len(triplets):,}")
    device = "mps" if torch.mps.is_available() else "cpu"
    console.log(f"[generate_training_file] Device: {device}")
    gen = NoThreeInLine(
        batch_size=batch_size,
        grid_size=n,
        max_points=2 * n,
        device=device,
        aggressive_blocking=True,
    )
    t0 = time.time()
    gen.saturate()
    t1 = time.time()
    console.log(f"[generate_training_file] Saturation took {t1 - t0:.2f}s")
    dataset_entries = gen.build_dataset(n, triplets)
    t2 = time.time()
    console.log(f"[generate_training_file] Dataset build took {t2 - t1:.2f}s")
    save_dataset_jld2(save_path, dataset_entries, triplets, n)
    console.log(f"[generate_training_file] Saved to {save_path}")
    return save_path


if __name__ == "__main__":

    N = 10
    batch_size = 10000

    solver_batched = NoThreeInLine(batch_size=batch_size, grid_size=N, max_points=2*N, aggressive_blocking=True)
    t0 = time.time()
    solver_batched.saturate()
    t1 = time.time()

    print(f"Saturation took {t1-t0:.2f} seconds.")

    t2 = time.time()
    print("Test completed!")
    print(f"Total time: {t2-t0:.2f} seconds.")
    
    # Count constructions for batched version

    points_counter_batched = Counter(solver_batched.current_counts.tolist())
    print(points_counter_batched)


    # Find the best grid (highest number of points)
    best_idx_batched = torch.argmax(solver_batched.current_counts).item()
    best_grid_batched = solver_batched.current_constructions[best_idx_batched]
    best_count_batched = solver_batched.current_counts[best_idx_batched].item()
    
    print(f"\nBest construction has {best_count_batched} points:")
    print_grid(best_grid_batched, f"Best Grid ({best_count_batched} points)")

    # solver_sequential = NoThreeInLine(batch_size=batch_size, grid_size=N, max_points=2*N)
    # t3 = time.time()
    # solver_sequential.greedy_saturate_batched()
    # t4 = time.time()
    # print(f"Saturation took {t4-t3:.2f} seconds.")

    # points_counter_sequential = Counter(solver_sequential.current_counts.tolist())
    # print(points_counter_sequential)

    # best_idx_sequential = torch.argmax(solver_sequential.current_counts).item()
    # best_grid_sequential = solver_sequential.current_constructions[best_idx_sequential]
    # best_count_sequential = solver_sequential.current_counts[best_idx_sequential].item()

    # print(f"\nBest construction has {best_count_sequential} points:")
    # print_grid(best_grid_sequential, f"Best Grid ({best_count_sequential} points)")

    # print(f"Speedup: {t4-t3:.2f} / {t1-t0:.2f} = {((t4-t3)/(t1-t0)):.2f}x")