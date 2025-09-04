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

    if query_mask is None:
        query_mask = np.ones(len(atom_array), dtype=bool)

    query_idx_in_ref = np.where(query_mask)[0]

    vdw_radii = np.array([vdw_radius_single(e) for e in atom_array.element])
    query_vdw_radii = vdw_radii[query_mask]

    clashes = []
    query_tree = KDTree(atom_array.coord)
    for query_idx, nbs_idx in enumerate(
        query_tree.query_ball_point(atom_array.coord[query_mask], r=3.0)
    ):
        query_bonds, _query_bond_types = atom_array.bonds.get_bonds(
            query_idx_in_ref[query_idx]
        )
        query_vdw = query_vdw_radii[query_idx]
        if query_vdw is None:
            # undefined vdw for elem, use 1.7 as "C"
            query_vdw = vdw_radius_single("C")

        for nb_idx in nbs_idx:
            if query_idx_in_ref[query_idx] == nb_idx:
                # clash with self
                continue

            if nb_idx in query_bonds:
                # clash with bonded atoms
                continue

            nb_vdw = vdw_radii[nb_idx]
            if nb_vdw is None:
                # undefined vdw for elem, use 1.7 as "C"
                nb_vdw = vdw_radius_single("C")

            dist = np.linalg.norm(
                atom_array.coord[query_mask][query_idx] - atom_array.coord[nb_idx]
            )
            if dist < vdw_scale_factor * (query_vdw + nb_vdw):
                clashes.append((query_idx_in_ref[query_idx], nb_idx))
    return clashes
