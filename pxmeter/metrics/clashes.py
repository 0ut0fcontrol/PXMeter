# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.info.radii import vdw_radius_single
from scipy.spatial import KDTree


def check_clashes_by_vdw(
    atom_array: AtomArray,
    query_mask: Sequence[bool] = None,
    vdw_scale_factor: float = 0.5,
) -> list[tuple[int, int]]:
    """
    Check clashes between atoms in the given atom array.

    Args:
        atom_array (AtomArray): The atom array to check for clashes.
        query_mask (bool, optional): A boolean mask to select atoms to check for clashes.
                   If None, all atoms are checked.
        vdw_scale_factor (float, optional): The scale factor to apply to the Van der Waals radii.
                         Defaults to 0.5.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the indices of atoms that are in clash.
    """
    if query_mask is None:
        # query all atoms
        query_mask = np.ones(len(atom_array), dtype=bool)
    elif not np.any(query_mask):
        # no query atoms, return empty list
        return []

    # Pre-calculate VDW radii, default to 1.7 (radius of Carbon) if undefined
    vdw_radii = np.array([vdw_radius_single(e) or 1.7 for e in atom_array.element])

    # Build KDTree and find all pairs within 3.0A
    query_tree = KDTree(atom_array.coord)
    # query_pairs returns (i, j) with i < j
    pairs = query_tree.query_pairs(3.0)
    pairs = np.array(list(pairs))

    if pairs.size == 0:
        return []

    i = pairs[:, 0]
    j = pairs[:, 1]

    # Matching original logic: a pair (i, j) is checked if the "source" atom is in query_mask.
    # In original loop, if i is in query_mask, it checks all neighbors j.
    # So we need both (i, j) and (j, i) if they satisfy the query_mask condition.
    all_candidate_pairs = []
    # Pairs where i is in query_mask
    mask_i = query_mask[i]
    if np.any(mask_i):
        all_candidate_pairs.append(pairs[mask_i])
    # Pairs where j is in query_mask (add as (j, i))
    mask_j = query_mask[j]
    if np.any(mask_j):
        all_candidate_pairs.append(pairs[mask_j][:, [1, 0]])

    if not all_candidate_pairs:
        return []

    directed_pairs = np.concatenate(all_candidate_pairs, axis=0)

    # Filter out bonded pairs
    # Get all bonds as (min, max) for easy lookup
    bonds = atom_array.bonds.as_array()[:, :2]
    # Use a set of tuples for fast lookup.
    # For large structures, we can use a more efficient way if needed.
    bond_set = set(map(tuple, bonds))
    bond_set.update(set(map(tuple, bonds[:, [1, 0]])))

    # Filter
    is_bonded = np.array([(p[0], p[1]) in bond_set for p in directed_pairs])
    directed_pairs = directed_pairs[~is_bonded]

    if directed_pairs.size == 0:
        return []

    # Final check with distances and VDW sum
    a1 = directed_pairs[:, 0]
    a2 = directed_pairs[:, 1]

    # Vectorized distance calculation
    diff = atom_array.coord[a1] - atom_array.coord[a2]
    dist = np.linalg.norm(diff, axis=1)

    vdw_sum = vdw_radii[a1] + vdw_radii[a2]
    is_clash = dist < vdw_scale_factor * vdw_sum

    clash_pairs = directed_pairs[is_clash]

    return [tuple(p) for p in clash_pairs]
