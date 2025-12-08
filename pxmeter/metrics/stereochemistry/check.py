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
import pandas as pd
from biotite.structure import AtomArray
from biotite.structure.info.radii import vdw_radius_single
from scipy.spatial import KDTree

from pxmeter.constants import DNA, NUC_BACKBONE, PROTEIN, PROTEIN_BACKBONE, RNA
from pxmeter.data.struct import Structure
from pxmeter.metrics.stereochemistry.params import (
    ANGLE_DATA,
    BOND_DATA,
    INTER_RES_ANGLE_DATA,
    INTER_RES_BOND_DATA,
)


class StereoChemValidator:
    """
    This class performs preprocessing steps required for downstream
    stereochemical checks, including grouping atoms by residue and
    classifying inter-residue connectivity types. The processed metadata is
    stored for later use by bond-length, bond-angle, and clash validation
    routines.

    Attributes:
        struct (Structure): The original input structure.
        ref_struct (Structure | None): Optional reference structure for
            clash detection. If provided, its atoms must be in strict
            one-to-one order correspondence with those in the query
            structure (same length and indexing), and its bond graph will
            be added to the query bond graph to detect clashes.
        atom_array (AtomArray): Parsed atom array containing coordinates
            and residue annotations.
        entity_poly_type (dict[str, str]): Mapping from entity ID to polymer
            category (e.g., protein, DNA, RNA).
        res_groups (dict): Preprocessed residue-level groupings derived from
            the atom array.
        inter_res_types (dict): Classification of inter-residue bond types
            used to guide stereochemical validation.
    """

    def __init__(
        self,
        struct: Structure,
        ref_struct: Structure | None = None,
    ):
        self.struct = struct
        self.ref_struct = ref_struct
        self.atom_array = struct.atom_array
        self.entity_poly_type = struct.entity_poly_type
        self.res_groups = self._preprocess_residue_groups(struct.atom_array)
        self.inter_res_types = self._classify_inter_residue_types()

    @staticmethod
    def _preprocess_residue_groups(
        atom_array: AtomArray,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Precompute residue-wise atom groups for bond/angle checks.

        This function groups atoms by residue name and, within each
        residue type, assigns a group ID to each residue instance
        identified by ``(chain_id, res_id)``. The result can be reused
        by bond/angle validation functions to avoid per-residue Python
        loops.

        Args:
            atom_array: Biotite ``AtomArray`` containing at least the
                fields ``res_name``, ``chain_id`` and ``res_id``.

        Returns:
            dict[str, dict[str, np.ndarray]]: A mapping from residue
            name (e.g. ``"ALA"``) to a dictionary with the following
            keys:

            - ``"idx_res"`` (np.ndarray, shape (N_res_atoms,), dtype=int):
              Global atom indices in the original ``atom_array`` for all
              atoms of this residue type.
            - ``"group_ids"`` (np.ndarray, shape (N_res_atoms,), dtype=int):
              Residue group index for each atom within this residue
              type. Atoms with the same value correspond to the same
              ``(chain_id, res_id)`` residue instance.
            - ``"n_groups"`` (int): Number of residue instances of this
              residue type in the structure.
        """
        res_name = np.asarray(atom_array.res_name).astype(str)
        chain_id = np.asarray(atom_array.chain_id).astype(str)
        res_id = np.asarray(atom_array.res_id)

        unique_res = np.unique(res_name)
        res_groups: dict[str, dict[str, np.ndarray]] = {}

        for rname in unique_res:
            idx_res = np.where(res_name == rname)[0]
            if idx_res.size == 0:
                continue

            chain_res = chain_id[idx_res]
            resid_res = res_id[idx_res]

            # Build a per-residue key "(chain, res_id)" for grouping
            keys = np.char.add(chain_res, resid_res.astype(str))

            _, group_ids_res = np.unique(keys, return_inverse=True)
            n_groups = int(group_ids_res.max()) + 1

            res_groups[rname] = {
                "idx_res": idx_res,  # Global indices for this residue type
                "group_ids": group_ids_res,  # Residue group index per atom
                "n_groups": n_groups,
            }

        return res_groups

    @staticmethod
    def _compute_dihedral(
        p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
    ) -> float:
        """
        Return dihedral angle (radians) for four points p0-p1-p2-p3.
        """
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2

        # normalize b1 so that it does not influence magnitude of vector
        b1 /= np.linalg.norm(b1)

        # components orthogonal to b1
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        v_norm = np.linalg.norm(v)
        w_norm = np.linalg.norm(w)
        if v_norm < 1e-8 or w_norm < 1e-8:
            return np.nan

        v /= v_norm
        w /= w_norm

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.arctan2(y, x)

    def _classify_single_pro_residue_type(
        self,
        chain_pro: str,
        resid_pro: int,
        entity_types_arr: np.ndarray,
    ) -> str | None:
        """
        Classify a single PRO residue as PRO_CIS / PRO_TRANS.

        Args:
            chain_pro: Chain ID of the PRO residue.
            resid_pro: Residue ID of the PRO residue.
            entity_types_arr: 1D array, same length as atom_array,
                giving per-atom entity type (e.g. PROTEIN / DNA / RNA / OTHER).

        Returns:
            "PRO_CIS", "PRO_TRANS", or None if classification fails.
        """
        atom_array = self.atom_array
        chain_id = self.struct.uni_chain_id
        res_id = atom_array.res_id
        atom_name = atom_array.atom_name
        coords = atom_array.coord

        prev_mask = (
            (chain_id == chain_pro)
            & (res_id < resid_pro)
            & (entity_types_arr == PROTEIN)
        )
        if not np.any(prev_mask):
            return None

        prev_res_ids = np.unique(res_id[prev_mask])
        prev_resid = int(prev_res_ids.max())

        def _find_atom(chain: str, resid: int, name: str) -> int | None:
            mask = (chain_id == chain) & (res_id == resid) & (atom_name == name)
            idx = np.where(mask)[0]
            return int(idx[0]) if idx.size > 0 else None

        idx_ca0 = _find_atom(chain_pro, prev_resid, "CA")
        idx_c0 = _find_atom(chain_pro, prev_resid, "C")
        idx_n1 = _find_atom(chain_pro, resid_pro, "N")
        idx_ca1 = _find_atom(chain_pro, resid_pro, "CA")

        if None in (idx_ca0, idx_c0, idx_n1, idx_ca1):
            return None

        p_ca0 = coords[idx_ca0]
        p_c0 = coords[idx_c0]
        p_n1 = coords[idx_n1]
        p_ca1 = coords[idx_ca1]

        omega = self._compute_dihedral(p_ca0, p_c0, p_n1, p_ca1)
        if np.isnan(omega):
            return None

        # OpenStructure: abs(omega) < 1.57 rad (~90°) -> cis
        if abs(omega) < 1.57:
            return "PRO_CIS"
        else:
            return "PRO_TRANS"

    def _classify_inter_residue_types(self) -> np.ndarray:
        entity_mapping = {PROTEIN: "PEPTIDE", DNA: "NA", RNA: "NA"}

        entity_types = np.array(
            [
                self.entity_poly_type.get(ent_id, "OTHER")
                for ent_id in self.atom_array.label_entity_id
            ],
            dtype=object,
        )
        inter_res_types = np.array(
            [entity_mapping.get(t, "OTHER") for t in entity_types],
            dtype=object,
        )

        # Update GLY types
        gly_mask = (self.atom_array.res_name == "GLY") & (entity_types == PROTEIN)
        inter_res_types[gly_mask] = "GLY"

        # Update PRO_CIS/TRANS types
        pro_mask = (self.atom_array.res_name == "PRO") & (entity_types == PROTEIN)
        if not np.any(pro_mask):
            return inter_res_types

        chain_id = self.struct.uni_chain_id
        res_id = self.atom_array.res_id

        pro_res_keys = sorted(
            set((chain_id[i], int(res_id[i])) for i in np.where(pro_mask)[0]),
            key=lambda x: (x[0], x[1]),
        )

        entity_types_arr = np.asarray(entity_types, dtype=object)

        for chain_pro, resid_pro in pro_res_keys:
            pro_type = self._classify_single_pro_residue_type(
                chain_pro,
                resid_pro,
                entity_types_arr,
            )
            if pro_type is None:
                continue

            mask_pro_res = (chain_id == chain_pro) & (res_id == resid_pro)
            inter_res_types[mask_pro_res] = pro_type

        return inter_res_types

    def find_bad_bonds(
        self,
        bond_data: dict[str, dict[str, tuple[float, float]]] | None = None,
        inter_res_bond_data: dict[str, dict[str, list[float]]] | None = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect all bond length outliers (intra- and inter-residue).

        This is a convenience wrapper that runs both intra-residue and
        inter-residue bond validation and concatenates the results into a single
        table. Bond lengths are compared against reference statistics and those
        whose Z-scores exceed the given threshold are reported.

        Args:
            bond_data (dict[str, dict[str, tuple[float, float]]], optional):
                Reference statistics for intra-residue bond lengths. The outer key
                is typically a residue name (e.g. ``"ALA"``), and the inner key is a
                bond identifier of the form ``"ATOM1_ATOM2"``. Each value is a
                ``(ideal, sigma)`` tuple giving the ideal bond length (Å) and its
                standard deviation. If ``None``, defaults to ``BOND_DATA``.
            inter_res_bond_data (dict[str, dict[str, list[float]]], optional):
                Reference statistics for inter-residue bond lengths. The outer key
                usually describes the inter-residue bond type, and the inner key
                is a bond identifier. Each value is a list of two floats
                ``[ideal, sigma]`` representing the ideal bond length (Å) and its
                standard deviation. If ``None``, defaults to ``INTER_RES_BOND_DATA``.
            z_thresh (float, optional):
                Absolute Z-score threshold above which a bond is considered an
                outlier. Bonds with ``abs(z_score) > z_thresh`` are reported.
                Defaults to ``12.0``.

        Returns:
            pandas.DataFrame:
                A concatenated table of intra- and inter-residue bond length
                outliers. The schema matches the outputs of
                :meth:`find_bad_intra_res_bonds` and
                :meth:`find_bad_inter_res_bonds`, and typically includes residue
                identifiers, atom names, bond identifiers, ideal/observed lengths,
                Z-scores, and atom indices (``idx1``, ``idx2``).

                If no out-of-range bonds are found, an empty ``DataFrame`` is
                returned.
        """
        if bond_data is None:
            bond_data = BOND_DATA
        if inter_res_bond_data is None:
            inter_res_bond_data = INTER_RES_BOND_DATA

        bad_intra_res_bonds = self.find_bad_intra_res_bonds(bond_data, z_thresh)
        bad_inter_res_bonds = self.find_bad_inter_res_bonds(
            inter_res_bond_data, z_thresh
        )
        bad_bonds = pd.concat([bad_intra_res_bonds, bad_inter_res_bonds])
        return bad_bonds

    def find_bad_inter_res_bonds(
        self,
        inter_res_bond_data: dict[str, dict[str, tuple[float, float]]] | None = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect inter-residue bond-length outliers.

        This method checks *inter-residue* bonds (e.g. peptide C-N,
        nucleic-acid O3'-P) using group-specific parameter sets defined
        in ``inter_res_bond_data``.

        The residue type for each atom is precomputed once in
        :meth:`_classify_inter_residue_types` and stored in
        ``self.inter_res_types``. For a covalent bond between atoms
        i and j, the parameter group is determined from their shared
        inter-residue type::

            group_i = self.inter_res_types[idx_i]
            group_j = self.inter_res_types[idx_j]

        Only bonds where ``group_i == group_j`` and the resulting group
        exists in ``inter_res_bond_data`` are considered. In addition,
        the two atoms must belong to different residues (i.e. truly
        inter-residue). Bond keys are matched in an order-independent
        manner by trying both ``ATOM1_ATOM2`` and ``ATOM2_ATOM1``.

        An inter-residue bond is reported as an outlier if its z-score
        exceeds ``z_thresh`` in absolute value::

            z = (length - ideal) / sigma

        Args:
            inter_res_bond_data: Mapping from group name (e.g.
                ``"PEPTIDE"``, ``"GLY"``, ``"NA"``, ``"PRO_CIS"``,
                ``"PRO_TRANS"``) to a mapping
                ``bond_key -> (ideal, sigma)``, where:

                * ``bond_key`` is a string of the form
                  ``"ATOM1_ATOM2"``, e.g. ``"C_N"``, ``"O3'_P"``.
                * ``ideal`` is the ideal bond length in Å.
                * ``sigma`` is the standard deviation used for z-score
                  computation.

                If ``None``, the default :data:`INTER_RES_BOND_DATA`
                will be used.
            z_thresh: Z-score threshold above which a bond is considered
                an outlier. Defaults to ``12.0``.

        Returns:
            pd.DataFrame: A DataFrame describing only the inter-residue
            bonds that exceed the z-score threshold. If no out-of-range
            bonds are found, an empty DataFrame is returned.

            The DataFrame has the following columns:

            * ``"group"``       - parameter group name (e.g. ``"NA"``,
              ``"PEPTIDE"``).
            * ``"bond_key"``    - bond key string used in the parameter
              table.
            * ``"idx1"``        - global atom index of atom 1
              (the lower index in the pair).
            * ``"idx2"``        - global atom index of atom 2
              (the higher index in the pair).
            * ``"res_name1"``   - residue name of atom 1.
            * ``"res_name2"``   - residue name of atom 2.
            * ``"chain_id1"``   - chain ID of atom 1.
            * ``"chain_id2"``   - chain ID of atom 2.
            * ``"res_id1"``     - residue ID of atom 1.
            * ``"res_id2"``     - residue ID of atom 2.
            * ``"atom_name1"``  - atom name of atom 1.
            * ``"atom_name2"``  - atom name of atom 2.
            * ``"ideal"``       - ideal bond length used for this bond.
            * ``"sigma"``       - standard deviation used for this bond.
            * ``"length"``      - observed bond length in Å.
            * ``"z_score"``     - z-score of the observed bond length.
        """
        if inter_res_bond_data is None:
            inter_res_bond_data = INTER_RES_BOND_DATA

        atom_array = self.atom_array
        coords = atom_array.coord

        atom_name_arr = atom_array.atom_name
        res_name_arr = atom_array.res_name
        chain_id_arr = self.struct.uni_chain_id
        res_id_arr = atom_array.res_id

        # Per-atom inter-residue type, e.g. "NA", "PEPTIDE",
        # "GLY", "PRO_CIS", "PRO_TRANS", "OTHER".
        inter_types = self.inter_res_types

        bad_records: list[dict[str, np.ndarray]] = []

        n_atoms = len(atom_array)
        for idx1 in range(n_atoms):
            group1 = inter_types[idx1]
            if group1 not in inter_res_bond_data:
                # This atom does not belong to any parameterized group
                continue

            name1 = atom_name_arr[idx1]

            # All atoms covalently bonded to idx1
            bonded_indices, _bond_types = atom_array.bonds.get_bonds(idx1)
            if len(bonded_indices) == 0:
                continue

            group_params = inter_res_bond_data[group1]

            for idx2 in bonded_indices:
                # Avoid double counting: enforce idx1 < idx2
                if idx1 >= idx2:
                    continue

                group2 = inter_types[idx2]
                if group2 != group1:
                    # Different inter-residue type -> no shared parameter set
                    continue

                # Require different residues => truly inter-residue
                res_key1 = (chain_id_arr[idx1], int(res_id_arr[idx1]))
                res_key2 = (chain_id_arr[idx2], int(res_id_arr[idx2]))
                if res_key1 == res_key2:
                    # Intra-residue bond; handled by intra-res tables
                    continue

                name2 = atom_name_arr[idx2]

                # Try both ATOM1_ATOM2 and ATOM2_ATOM1
                key_12 = f"{name1}_{name2}"
                key_21 = f"{name2}_{name1}"

                if key_12 in group_params:
                    bond_key = key_12
                    ideal, sigma = group_params[key_12]
                    idx1_use, idx2_use = idx1, idx2
                    name1_use, name2_use = name1, name2
                elif key_21 in group_params:
                    bond_key = key_21
                    ideal, sigma = group_params[key_21]
                    # Swap roles to match parameter key order
                    idx1_use, idx2_use = idx2, idx1
                    name1_use, name2_use = name2, name1
                else:
                    # This particular bonded pair is not parameterized
                    continue

                # Compute bond length
                p1 = coords[idx1_use]
                p2 = coords[idx2_use]
                length = float(np.linalg.norm(p1 - p2))

                z = (length - ideal) / sigma
                if np.abs(z) <= z_thresh:
                    continue

                bad_records.append(
                    {
                        "group": np.array([group1], dtype=object),
                        "bond_key": np.array([bond_key], dtype=object),
                        "idx1": np.array([idx1_use], dtype=int),
                        "idx2": np.array([idx2_use], dtype=int),
                        "res_name1": np.array([res_name_arr[idx1_use]], dtype=object),
                        "res_name2": np.array([res_name_arr[idx2_use]], dtype=object),
                        "chain_id1": np.array([chain_id_arr[idx1_use]], dtype=object),
                        "chain_id2": np.array([chain_id_arr[idx2_use]], dtype=object),
                        "res_id1": np.array([res_id_arr[idx1_use]], dtype=int),
                        "res_id2": np.array([res_id_arr[idx2_use]], dtype=int),
                        "atom_name1": np.array([name1_use], dtype=object),
                        "atom_name2": np.array([name2_use], dtype=object),
                        "ideal": np.array([ideal], dtype=float),
                        "sigma": np.array([sigma], dtype=float),
                        "length": np.array([length], dtype=float),
                        "z_score": np.array([z], dtype=float),
                    }
                )

        if not bad_records:
            return pd.DataFrame()

        out: dict[str, np.ndarray] = {}
        for key in bad_records[0].keys():
            out[key] = np.concatenate([r[key] for r in bad_records])

        return pd.DataFrame(out)

    def find_bad_intra_res_bonds(
        self,
        bond_data: dict[str, dict[str, tuple[float, float]]] | None = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect intra-residue bond length outliers.

        This method evaluates all defined intra-residue bonds for each residue
        type, compares observed bond lengths to reference statistics, and reports
        those whose Z-score exceeds a given threshold. Within each residue
        instance, at most one atom of each name is used per bond (i.e., one
        main-atom A/B per residue group).

        Args:
            bond_data (dict[str, dict[str, tuple[float, float]]], optional):
                Reference statistics for intra-residue bond lengths. The outer key
                is typically a residue name (e.g. ``"ALA"``), and the inner key is a
                bond identifier of the form ``"ATOM1_ATOM2"`` (e.g. ``"N_CA"``).
                Each value is a ``(ideal, sigma)`` tuple giving the ideal bond
                length (in Å) and its standard deviation. If ``None``, defaults to
                ``BOND_DATA``.
            z_thresh (float, optional):
                Absolute Z-score threshold above which a bond is considered an
                outlier. Bonds with ``abs(z_score) > z_thresh`` are reported.
                Defaults to ``12.0``.

        Returns:
            pandas.DataFrame:
                A table of intra-residue bond length outliers. Each row
                corresponds to a single bond that violates the Z-score threshold
                and includes at least the following columns:

                * ``"group"``       - parameter group name (e.g. ``"ALA"``,
                    ``"GLN"``).
                * ``"bond_key"``    - bond key string used in the parameter
                    table.
                * ``"idx1"``        - global atom index of atom 1
                    (the lower index in the pair).
                * ``"idx2"``        - global atom index of atom 2
                    (the higher index in the pair).
                * ``"res_name1"``   - residue name of atom 1.
                * ``"res_name2"``   - residue name of atom 2.
                * ``"chain_id1"``   - chain ID of atom 1.
                * ``"chain_id2"``   - chain ID of atom 2.
                * ``"res_id1"``     - residue ID of atom 1.
                * ``"res_id2"``     - residue ID of atom 2.
                * ``"atom_name1"``  - atom name of atom 1.
                * ``"atom_name2"``  - atom name of atom 2.
                * ``"ideal"``       - ideal bond length used for this bond.
                * ``"sigma"``       - standard deviation used for this bond.
                * ``"length"``      - observed bond length in Å.
                * ``"z_score"``     - z-score of the observed bond length.

                If no out-of-range bonds are found, an empty ``DataFrame`` is
                returned.
        """
        if bond_data is None:
            bond_data = BOND_DATA

        atom_array = self.atom_array
        res_groups = self.res_groups

        atom_name = atom_array.atom_name
        chain_id = self.struct.uni_chain_id
        res_id = atom_array.res_id
        coords = atom_array.coord  # (N, 3)

        bad_records: list[dict[str, np.ndarray]] = []

        for rname, bonds in bond_data.items():
            if rname not in res_groups:
                continue

            idx_res = res_groups[rname]["idx_res"]
            group_ids_res = res_groups[rname]["group_ids"]
            n_groups = res_groups[rname]["n_groups"]

            atom_names_res = atom_name[idx_res]

            for bond_key, (ideal, sigma) in bonds.items():
                a_name, b_name = bond_key.split("_")

                mask_a = atom_names_res == a_name
                mask_b = atom_names_res == b_name
                if not (mask_a.any() and mask_b.any()):
                    continue

                idx_a_atoms = idx_res[mask_a]
                gid_a = group_ids_res[mask_a]

                idx_b_atoms = idx_res[mask_b]
                gid_b = group_ids_res[mask_b]

                # Keep at most one A/B atom per residue group (main-atom only)
                idx_a_group = np.full(n_groups, -1, dtype=np.int64)
                idx_b_group = np.full(n_groups, -1, dtype=np.int64)
                idx_a_group[gid_a] = idx_a_atoms
                idx_b_group[gid_b] = idx_b_atoms

                valid = (idx_a_group >= 0) & (idx_b_group >= 0)
                if not valid.any():
                    continue

                ia = idx_a_group[valid]
                ib = idx_b_group[valid]

                vec = coords[ia] - coords[ib]
                length = np.linalg.norm(vec, axis=1)

                # Compute z-score and select out-of-range bonds
                z = (length - ideal) / sigma
                bad = np.abs(z) > z_thresh
                if not bad.any():
                    continue

                ia_bad = ia[bad]
                ib_bad = ib[bad]
                length_bad = length[bad]
                z_bad = z[bad]

                res_name_arr = np.full_like(length_bad, rname, dtype=object)
                atom_name1_arr = np.full_like(length_bad, a_name, dtype=object)
                atom_name2_arr = np.full_like(length_bad, b_name, dtype=object)

                bad_records.append(
                    {
                        "group": np.full_like(length_bad, rname, dtype=object),
                        "res_name1": res_name_arr,
                        "res_name2": res_name_arr,
                        "chain_id1": chain_id[ib_bad],  # Use atom2 (B) as reference
                        "chain_id2": chain_id[ib_bad],
                        "res_id1": res_id[ib_bad],
                        "res_id2": res_id[ib_bad],
                        "atom_name1": atom_name1_arr,
                        "atom_name2": atom_name2_arr,
                        "bond_key": np.full_like(
                            length_bad,
                            bond_key,
                            dtype=object,
                        ),
                        "ideal": np.full_like(length_bad, ideal, dtype=float),
                        "sigma": np.full_like(length_bad, sigma, dtype=float),
                        "length": length_bad,
                        "z_score": z_bad,
                        "idx1": ia_bad,
                        "idx2": ib_bad,
                    }
                )

        if not bad_records:
            # No out-of-range bonds found
            return pd.DataFrame()

        out: dict[str, np.ndarray] = {}
        for key in bad_records[0].keys():
            out[key] = np.concatenate([r[key] for r in bad_records])

        return pd.DataFrame(out)

    def find_bad_inter_res_angles(
        self,
        inter_res_angle_data: dict[str, dict[str, tuple[float, float]]] | None = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect inter-residue backbone angle outliers.

        This method checks *inter-residue* backbone angles such as
        peptide and nucleic-acid linkage angles using residue-type-
        specific parameter sets defined in ``inter_res_angle_data``.

        The residue type for each atom is precomputed once in
        :meth:`_classify_inter_residue_types` and stored in
        ``self.inter_res_types``. For an angle A-B-C, the parameter
        group is determined from the shared inter-residue type of all
        three atoms::

            group_a = self.inter_res_types[idx_a]
            group_b = self.inter_res_types[idx_b]
            group_c = self.inter_res_types[idx_c]

        Only angles where ``group_a == group_b == group_c`` and
        ``group_b`` exists in ``inter_res_angle_data`` are considered.
        In addition, the three atoms must span at least two distinct
        residues (i.e. truly inter-residue).

        Typical groups include:

        * ``"NA"``        - nucleic-acid linkage angles
        * ``"PEPTIDE"``   - generic peptide backbone
        * ``"GLY"``       - glycine-specific peptide parameters
        * ``"PRO_CIS"``   - cis proline peptide bond
        * ``"PRO_TRANS"`` - trans proline peptide bond

        An angle is reported as an outlier if its z-score exceeds
        ``z_thresh`` in absolute value::

            z = (angle_deg - ideal) / sigma

        Args:
            inter_res_angle_data: Mapping from group name (e.g.
                ``"PEPTIDE"``, ``"GLY"``, ``"NA"``) to a mapping
                ``angle_key -> (ideal, sigma)``, where:

                * ``angle_key`` is a string of the form
                  ``"ATOM_A_ATOM_B_ATOM_C"``, e.g. ``"CA_C_N"`` where
                  ``ATOM_B`` is the central atom.
                * ``ideal`` is the ideal bond angle in degrees.
                * ``sigma`` is the standard deviation used for z-score
                  computation.

                If ``None``, the default :data:`INTER_RES_ANGLE_DATA`
                will be used.
            z_thresh: Z-score threshold above which an angle is
                considered an outlier. Defaults to ``12.0``.

        Returns:
            pd.DataFrame: A DataFrame describing only the inter-residue
            angles that exceed the z-score threshold. If no out-of-range
            angles are found, an empty DataFrame is returned.

            The DataFrame has the following columns:

            * ``"group"``        - parameter group name (e.g. ``"NA"``,
              ``"PEPTIDE"``).
            * ``"angle_key"``    - angle key string used in the parameter
              table.
            * ``"idx_a"``        - global atom index of atom A.
            * ``"idx_b"``        - global atom index of atom B (central).
            * ``"idx_c"``        - global atom index of atom C.
            * ``"res_name_a"``   - residue name of atom A.
            * ``"res_name_b"``   - residue name of atom B.
            * ``"res_name_c"``   - residue name of atom C.
            * ``"chain_id_a"``   - chain ID of atom A.
            * ``"chain_id_b"``   - chain ID of atom B.
            * ``"chain_id_c"``   - chain ID of atom C.
            * ``"res_id_a"``     - residue ID of atom A.
            * ``"res_id_b"``     - residue ID of atom B.
            * ``"res_id_c"``     - residue ID of atom C.
            * ``"atom_name_a"``  - atom name of A.
            * ``"atom_name_b"``  - atom name of B.
            * ``"atom_name_c"``  - atom name of C.
            * ``"ideal"``        - ideal angle (degrees) used for this
              triplet.
            * ``"sigma"``        - standard deviation used for this
              angle.
            * ``"angle"``        - observed angle in degrees.
            * ``"z_score"``      - z-score of the observed angle.
        """
        if inter_res_angle_data is None:
            inter_res_angle_data = INTER_RES_ANGLE_DATA

        atom_array = self.atom_array
        coords = atom_array.coord

        atom_name_arr = atom_array.atom_name
        res_name_arr = atom_array.res_name
        chain_id_arr = self.struct.uni_chain_id
        res_id_arr = atom_array.res_id

        # Per-atom inter-residue type, e.g. "NA", "PEPTIDE",
        # "GLY", "PRO_CIS", "PRO_TRANS", "OTHER".
        inter_types = self.inter_res_types

        bad_records: list[dict[str, np.ndarray]] = []

        n_atoms = len(atom_array)
        for idx_b in range(n_atoms):
            group_b = inter_types[idx_b]
            if group_b not in inter_res_angle_data:
                # Central atom does not belong to any parameterized group
                continue

            b_name = atom_name_arr[idx_b]
            group_params = inter_res_angle_data[group_b]

            # All atoms covalently bonded to B
            bonded_indices, _bond_types = atom_array.bonds.get_bonds(idx_b)
            if len(bonded_indices) < 2:
                continue

            # Enumerate all unordered neighbor pairs (A, C) around B
            for i, idx_a in enumerate(bonded_indices):
                for idx_c in bonded_indices[i + 1 :]:
                    group_a = inter_types[idx_a]
                    group_c = inter_types[idx_c]

                    # All three atoms must share the same inter-residue type
                    if group_a != group_b or group_c != group_b:
                        continue

                    # Require at least two distinct residues among A, B, C
                    res_keys = {
                        (chain_id_arr[idx_a], int(res_id_arr[idx_a])),
                        (chain_id_arr[idx_b], int(res_id_arr[idx_b])),
                        (chain_id_arr[idx_c], int(res_id_arr[idx_c])),
                    }
                    if len(res_keys) < 2:
                        # Purely intra-residue angle; handled by intra-res tables
                        continue

                    a_name = atom_name_arr[idx_a]
                    c_name = atom_name_arr[idx_c]

                    # Try both A-B-C and C-B-A keys (angle is symmetric)
                    key_abc = f"{a_name}_{b_name}_{c_name}"
                    key_cba = f"{c_name}_{b_name}_{a_name}"

                    if key_abc in group_params:
                        angle_key = key_abc
                        ideal, sigma = group_params[key_abc]
                        idx_a_use, idx_c_use = idx_a, idx_c
                        a_name_use, c_name_use = a_name, c_name
                    elif key_cba in group_params:
                        angle_key = key_cba
                        ideal, sigma = group_params[key_cba]
                        # Swap A/C to match the parameter key order
                        idx_a_use, idx_c_use = idx_c, idx_a
                        a_name_use, c_name_use = c_name, a_name
                    else:
                        # This particular A-B-C triplet is not parameterized
                        continue

                    # Compute angle at B for A-B-C
                    A = coords[idx_a_use]
                    B = coords[idx_b]
                    C = coords[idx_c_use]

                    v1 = A - B
                    v2 = C - B
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)

                    eps = 1e-8
                    if norm_v1 <= eps or norm_v2 <= eps:
                        continue

                    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle_deg = float(np.degrees(np.arccos(cos_theta)))

                    z = (angle_deg - ideal) / sigma
                    if np.abs(z) <= z_thresh:
                        continue

                    bad_records.append(
                        {
                            "group": np.array([group_b], dtype=object),
                            "angle_key": np.array([angle_key], dtype=object),
                            "idx_a": np.array([idx_a_use], dtype=int),
                            "idx_b": np.array([idx_b], dtype=int),
                            "idx_c": np.array([idx_c_use], dtype=int),
                            "res_name_a": np.array(
                                [res_name_arr[idx_a_use]],
                                dtype=object,
                            ),
                            "res_name_b": np.array(
                                [res_name_arr[idx_b]],
                                dtype=object,
                            ),
                            "res_name_c": np.array(
                                [res_name_arr[idx_c_use]],
                                dtype=object,
                            ),
                            "chain_id_a": np.array(
                                [chain_id_arr[idx_a_use]],
                                dtype=object,
                            ),
                            "chain_id_b": np.array(
                                [chain_id_arr[idx_b]],
                                dtype=object,
                            ),
                            "chain_id_c": np.array(
                                [chain_id_arr[idx_c_use]],
                                dtype=object,
                            ),
                            "res_id_a": np.array(
                                [res_id_arr[idx_a_use]],
                                dtype=int,
                            ),
                            "res_id_b": np.array([res_id_arr[idx_b]], dtype=int),
                            "res_id_c": np.array(
                                [res_id_arr[idx_c_use]],
                                dtype=int,
                            ),
                            "atom_name_a": np.array(
                                [a_name_use],
                                dtype=object,
                            ),
                            "atom_name_b": np.array(
                                [b_name],
                                dtype=object,
                            ),
                            "atom_name_c": np.array(
                                [c_name_use],
                                dtype=object,
                            ),
                            "ideal": np.array([ideal], dtype=float),
                            "sigma": np.array([sigma], dtype=float),
                            "angle": np.array([angle_deg], dtype=float),
                            "z_score": np.array([z], dtype=float),
                        }
                    )

        if not bad_records:
            return pd.DataFrame()

        out: dict[str, np.ndarray] = {}
        for key in bad_records[0].keys():
            out[key] = np.concatenate([r[key] for r in bad_records])

        return pd.DataFrame(out)

    def find_bad_intra_res_angles(
        self,
        angle_data: dict[str, dict[str, tuple[float, float]]] | None = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect bond-angle outliers based on residue-specific statistics.

        For each residue type and each defined angle key in
        ``angle_data``, this method finds the corresponding atom
        triplets within each residue instance and computes the angle at
        the central atom ``B`` for A-B-C. An angle is considered
        out-of-range if its z-score exceeds ``z_thresh`` in absolute
        value::

            z = (angle_deg - ideal) / sigma

        Only angles defined in ``angle_data`` are checked; any atom
        triplets or residues not covered by the definitions are
        implicitly treated as having no violations and are therefore
        omitted from the result.

        Args:
            angle_data: Mapping from residue name (e.g. ``"ALA"``) to a
                mapping ``angle_key -> (ideal, sigma)``, where:

                * ``angle_key`` is a string of the form
                  ``"ATOM_A_ATOM_B_ATOM_C"``, e.g. ``"CA_N_H"`` where
                  ``ATOM_B`` is the central atom.
                * ``ideal`` is the ideal bond angle in degrees.
                * ``sigma`` is the standard deviation used for z-score
                  computation.

                If ``None``, the default :data:`ANGLE_DATA` will be
                used.
            z_thresh: Z-score threshold above which an angle is
                considered an outlier. Defaults to ``12.0``.

        Returns:
            dict[str, np.ndarray]: A dictionary of NumPy arrays
            describing only the angles that exceed the z-score
            threshold. If no out-of-range angles are found, an empty
            dictionary is returned.

            The dictionary contains the following keys (each mapped to a
            1D array of the same length):

            * ``"group"``        - parameter group name (e.g. ``"ALA"``,
              ``"GLN"``).
            * ``"angle_key"``    - angle key string used in the parameter
              table.
            * ``"idx_a"``        - global atom index of atom A.
            * ``"idx_b"``        - global atom index of atom B (central).
            * ``"idx_c"``        - global atom index of atom C.
            * ``"res_name_a"``   - residue name of atom A.
            * ``"res_name_b"``   - residue name of atom B.
            * ``"res_name_c"``   - residue name of atom C.
            * ``"chain_id_a"``   - chain ID of atom A.
            * ``"chain_id_b"``   - chain ID of atom B.
            * ``"chain_id_c"``   - chain ID of atom C.
            * ``"res_id_a"``     - residue ID of atom A.
            * ``"res_id_b"``     - residue ID of atom B.
            * ``"res_id_c"``     - residue ID of atom C.
            * ``"atom_name_a"``  - atom name of A.
            * ``"atom_name_b"``  - atom name of B (central).
            * ``"atom_name_c"``  - atom name of C.
            * ``"ideal"``        - ideal angle (degrees) used for this
              triplet.
            * ``"sigma"``        - standard deviation used for this
              angle.
            * ``"angle"``        - observed angle in degrees.
            * ``"z_score"``      - z-score of the observed angle.
        """
        if angle_data is None:
            angle_data = ANGLE_DATA

        atom_array = self.atom_array
        res_groups = self.res_groups

        atom_name_arr = np.asarray(atom_array.atom_name).astype(str)
        chain_id_arr = np.asarray(atom_array.chain_id).astype(str)
        res_id_arr = np.asarray(atom_array.res_id)
        coords = atom_array.coord

        bad_records: list[dict[str, np.ndarray]] = []

        for rname, angles in angle_data.items():
            if rname not in res_groups:
                continue

            idx_res = res_groups[rname]["idx_res"]
            group_ids_res = res_groups[rname]["group_ids"]
            n_groups = res_groups[rname]["n_groups"]

            atom_names_res = atom_name_arr[idx_res]

            for angle_key, (ideal, sigma) in angles.items():
                a_name, b_name, c_name = angle_key.split("_")

                mask_a = atom_names_res == a_name
                mask_b = atom_names_res == b_name
                mask_c = atom_names_res == c_name
                if not (mask_a.any() and mask_b.any() and mask_c.any()):
                    continue

                idx_a_atoms = idx_res[mask_a]
                gid_a = group_ids_res[mask_a]

                idx_b_atoms = idx_res[mask_b]
                gid_b = group_ids_res[mask_b]

                idx_c_atoms = idx_res[mask_c]
                gid_c = group_ids_res[mask_c]

                idx_a_group = np.full(n_groups, -1, dtype=np.int64)
                idx_b_group = np.full(n_groups, -1, dtype=np.int64)
                idx_c_group = np.full(n_groups, -1, dtype=np.int64)

                idx_a_group[gid_a] = idx_a_atoms
                idx_b_group[gid_b] = idx_b_atoms
                idx_c_group[gid_c] = idx_c_atoms

                valid = (idx_a_group >= 0) & (idx_b_group >= 0) & (idx_c_group >= 0)
                if not valid.any():
                    continue

                ia = idx_a_group[valid]
                ib = idx_b_group[valid]
                ic = idx_c_group[valid]

                A = coords[ia]
                B = coords[ib]
                C = coords[ic]

                v1 = A - B
                v2 = C - B
                v1_norm = np.linalg.norm(v1, axis=1)
                v2_norm = np.linalg.norm(v2, axis=1)

                eps = 1e-8
                len_valid = (v1_norm > eps) & (v2_norm > eps)
                if not len_valid.any():
                    continue

                cos_theta = np.empty_like(v1_norm)
                cos_theta[:] = np.nan

                dot = np.einsum("ij,ij->i", v1[len_valid], v2[len_valid])
                cos_theta[len_valid] = dot / (v1_norm[len_valid] * v2_norm[len_valid])
                cos_theta = np.clip(cos_theta, -1.0, 1.0)

                angle = np.degrees(np.arccos(cos_theta))
                z = (angle - ideal) / sigma
                bad = np.abs(z) > z_thresh

                if not bad.any():
                    continue

                ia_bad = ia[bad]
                ib_bad = ib[bad]
                ic_bad = ic[bad]
                angle_bad = angle[bad]
                z_bad = z[bad]

                res_name_arr = np.full_like(angle_bad, rname, dtype=object)

                bad_records.append(
                    {
                        "group": np.full_like(angle_bad, rname, dtype=object),
                        "res_name_a": res_name_arr,
                        "res_name_b": res_name_arr,
                        "res_name_c": res_name_arr,
                        "chain_id_a": chain_id_arr[ib_bad],
                        "chain_id_b": chain_id_arr[ib_bad],
                        "chain_id_c": chain_id_arr[ib_bad],
                        "res_id_a": res_id_arr[ib_bad],
                        "res_id_b": res_id_arr[ib_bad],
                        "res_id_c": res_id_arr[ib_bad],
                        "atom_name_a": np.full_like(angle_bad, a_name, dtype=object),
                        "atom_name_b": np.full_like(angle_bad, b_name, dtype=object),
                        "atom_name_c": np.full_like(angle_bad, c_name, dtype=object),
                        "angle_key": np.full_like(
                            angle_bad,
                            angle_key,
                            dtype=object,
                        ),
                        "ideal": np.full_like(angle_bad, ideal, dtype=float),
                        "sigma": np.full_like(angle_bad, sigma, dtype=float),
                        "angle": angle_bad,
                        "z_score": z_bad,
                        "idx_a": ia_bad,
                        "idx_b": ib_bad,
                        "idx_c": ic_bad,
                    }
                )

        if not bad_records:
            return pd.DataFrame()

        out: dict[str, np.ndarray] = {}
        for key in bad_records[0].keys():
            out[key] = np.concatenate([r[key] for r in bad_records])

        return pd.DataFrame(out)

    def find_bad_angles(
        self,
        angle_data: dict[str, dict[str, tuple[float, float]]] | None = None,
        inter_res_angle_data: dict[str, dict[str, tuple[float, float]]] | None = None,
        z_thresh: float = 12.0,
    ) -> pd.DataFrame:
        """
        Detect intra- and inter-residue bond angle outliers.

        This method scans all defined bond angles in the structure and flags those
        whose deviation from the reference mean exceeds a given Z-score threshold.
        Intra-residue and inter-residue angles are evaluated separately using
        the corresponding reference tables and then concatenated into a single
        output table.

        Args:
            angle_data (dict[str, dict[str, tuple[float, float]]], optional):
                Reference statistics for intra-residue bond angles. The outer key
                typically corresponds to the residue name (e.g. ``"ALA"``) and the
                inner key to an angle identifier (e.g. ``"N-CA-C"``). Each value is
                a ``(mean, std)`` tuple in degrees. If ``None``, defaults to
                ``ANGLE_DATA``.
            inter_res_angle_data (dict[str, dict[str, tuple[float, float]]], optional):
                Reference statistics for inter-residue bond angles, using the same
                ``(mean, std)`` convention as ``angle_data``. If ``None``, defaults
                to ``INTER_RES_ANGLE_DATA``.
            z_thresh (float, optional):
                Absolute Z-score threshold above which an angle is considered
                an outlier. Defaults to ``12.0``.

        Returns:
            pandas.DataFrame:
                A table of bond angle outliers. Each row corresponds to a single
                intra- or inter-residue angle whose Z-score exceeds ``z_thresh``.
                The exact schema is defined by
                :meth:`find_bad_intra_res_angles` and
                :meth:`find_bad_inter_res_angles`, and typically includes residue
                identifiers, atom names, the observed angle, reference mean,
                Z-score, and a flag indicating intra- vs inter-residue origin.
        """
        if angle_data is None:
            angle_data = ANGLE_DATA
        if inter_res_angle_data is None:
            inter_res_angle_data = INTER_RES_ANGLE_DATA
        bad_intra_res_angles = self.find_bad_intra_res_angles(angle_data, z_thresh)
        bad_inter_res_angles = self.find_bad_inter_res_angles(
            inter_res_angle_data, z_thresh
        )
        return pd.concat([bad_intra_res_angles, bad_inter_res_angles])

    def find_clashes(
        self,
        query_mask: Sequence[bool] | None = None,
        vdw_scale_factor: float | None = None,
        tolerance: float | None = 1.5,
        disulfide_clash_tolerance: float = 1.0,
        cutoff: float = 3.0,
    ) -> pd.DataFrame:
        """
        Identify steric clashes based on van der Waals radii.

        This method finds atom pairs that are closer than an allowed contact
        distance derived from their van der Waals radii. The search is restricted
        to atoms within a given spatial cutoff using a KD-tree, and excludes
        directly bonded atom pairs. The user can either specify a global scaling
        factor for the van der Waals contact distance or a fixed tolerance value
        to subtract from the sum of radii.

        Exactly one of ``vdw_scale_factor`` or ``tolerance`` must be provided.

        Args:
            query_mask (Sequence[bool], optional):
                Boolean mask over all atoms in ``self.atom_array`` specifying the
                subset of atoms to be treated as query atoms. Clashes are reported
                between these query atoms and all atoms in the structure
                (including other query atoms). If ``None``, all atoms are used as
                query atoms.
            vdw_scale_factor (float, optional):
                Global scaling factor applied to the sum of van der Waals radii to
                define the allowed contact distance:
                ``contact_limit = vdw_scale_factor * (r1 + r2)``.
                Must be ``None`` if ``tolerance`` is provided.
            tolerance (float, optional):
                Subtractive tolerance applied to the sum of van der Waals radii:
                ``contact_limit = (r1 + r2) - tolerance``. Distances smaller than
                this limit are reported as clashes. Must be ``None`` if
                ``vdw_scale_factor`` is provided. Defaults to ``1.5``.
            disulfide_clash_tolerance (float, optional):
                The respective tolerance for two atoms can potentially build a disulfide bond.
                Defaults to ``1.0``.
            cutoff (float, optional):
                Maximum distance (in Å) for the KD-tree neighbor search. Atom pairs
                farther apart than this value are ignored. Defaults to ``3.0``.

        Returns:
            pandas.DataFrame:
                A table of detected steric clashes. Each row corresponds to one
                clashing atom pair and contains:

                * ``idx1``, ``idx2``: Atom indices in ``self.atom_array``.
                * ``res_name1``, ``chain_id1``, ``res_id1``: Residue info of atom 1.
                * ``res_name2``, ``chain_id2``, ``res_id2``: Residue info of atom 2.
                * ``element1``, ``element2``: Chemical elements of the two atoms.
                * ``distance``: Observed inter-atomic distance (Å).
                * ``contact_limit``: Allowed contact distance for this pair (Å).
                * ``overlap``: ``contact_limit - distance``; positive values indicate
                the magnitude of the steric overlap.

                If no clashes are found, an empty ``DataFrame`` is returned.
        """

        atom_array = self.atom_array
        n_atoms = len(atom_array)

        if not ((vdw_scale_factor is None) ^ (tolerance is None)):
            raise ValueError(
                "Exactly one of `vdw_scale_factor` or `tolerance` must be provided."
            )

        if query_mask is None:
            query_mask = np.ones(n_atoms, dtype=bool)
        else:
            query_mask = np.asarray(query_mask, dtype=bool)
            if query_mask.shape[0] != n_atoms:
                raise ValueError(
                    f"query_mask length {query_mask.shape[0]} "
                    f"does not match number of atoms {n_atoms}"
                )

        bonds = atom_array.bonds
        if self.ref_struct is not None:
            # Add bonds from reference structure if available
            assert len(self.ref_struct.atom_array) == n_atoms
            bonds += self.ref_struct.atom_array.bonds

        if not np.any(query_mask):
            # no query atoms
            return pd.DataFrame()

        query_idx_in_ref = np.where(query_mask)[0]

        coords = atom_array.coord
        query_coords = coords[query_mask]

        res_name_arr = atom_array.res_name
        chain_id_arr = self.struct.uni_chain_id
        res_id_arr = atom_array.res_id
        element_arr = atom_array.element
        atom_name_arr = atom_array.atom_name

        vdw_radii = np.array(
            [vdw_radius_single(e) for e in atom_array.element],
            dtype=object,
        )
        query_vdw_radii = vdw_radii[query_mask]

        is_cys_sg = (res_name_arr == "CYS") & (atom_name_arr == "SG")

        tree = KDTree(coords)

        idx1_list: list[int] = []
        idx2_list: list[int] = []
        dist_list: list[float] = []
        limit_list: list[float] = []

        for local_q_idx, nb_indices in enumerate(
            tree.query_ball_point(query_coords, r=cutoff)
        ):
            ref_q_idx = query_idx_in_ref[local_q_idx]
            q_vdw = query_vdw_radii[local_q_idx]
            if q_vdw is None:
                q_vdw = vdw_radius_single("C")

            q_coord = query_coords[local_q_idx]

            bonded_indices, _bond_types = bonds.get_bonds(ref_q_idx)
            bonded_set = set(bonded_indices)

            for nb_idx in nb_indices:
                if nb_idx == ref_q_idx:
                    continue
                if nb_idx in bonded_set:
                    continue
                if query_mask[nb_idx] and ref_q_idx > nb_idx:
                    continue

                nb_vdw = vdw_radii[nb_idx]
                if nb_vdw is None:
                    nb_vdw = vdw_radius_single("C")

                dist = np.linalg.norm(q_coord - coords[nb_idx])

                if tolerance is None:
                    contact_limit = vdw_scale_factor * (q_vdw + nb_vdw)
                else:
                    if is_cys_sg[ref_q_idx] and is_cys_sg[nb_idx]:
                        contact_limit = (q_vdw + nb_vdw) - disulfide_clash_tolerance
                    else:
                        contact_limit = (q_vdw + nb_vdw) - tolerance

                if dist < contact_limit:
                    idx1_list.append(ref_q_idx)
                    idx2_list.append(nb_idx)
                    dist_list.append(dist)
                    limit_list.append(contact_limit)

        if not idx1_list:
            return pd.DataFrame()

        idx1 = np.asarray(idx1_list, dtype=int)
        idx2 = np.asarray(idx2_list, dtype=int)
        distance = np.asarray(dist_list, dtype=float)
        contact_limit = np.asarray(limit_list, dtype=float)
        overlap = contact_limit - distance

        out_df = pd.DataFrame(
            {
                "idx1": idx1,
                "idx2": idx2,
                "res_name1": res_name_arr[idx1],
                "chain_id1": chain_id_arr[idx1],
                "res_id1": res_id_arr[idx1],
                "res_name2": res_name_arr[idx2],
                "chain_id2": chain_id_arr[idx2],
                "res_id2": res_id_arr[idx2],
                "atom_name1": atom_name_arr[idx1],
                "atom_name2": atom_name_arr[idx2],
                "element1": element_arr[idx1],
                "element2": element_arr[idx2],
                "distance": distance,
                "contact_limit": contact_limit,
                "overlap": overlap,
            }
        )
        return out_df

    def find_all_violations(
        self,
        clash_tolerance: float = 1.5,
        disulfide_clash_tolerance: float = 1.0,
        z_thresh: float = 12.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run clash, bond, and angle checks and return all violations.

        Args:
            clash_tolerance: Distance tolerance (Å) used for clash detection.
            disulfide_clash_tolerance: Distance tolerance used in disulfide clash detection.
            z_thresh: Absolute Z-score threshold for bond and angle outliers.

        Returns:
            Tuple of three DataFrames: (clash_df, bad_bond_df, bad_angle_df).
        """
        clash_df = self.find_clashes(
            tolerance=clash_tolerance,
            disulfide_clash_tolerance=disulfide_clash_tolerance,
        )
        bad_bond_df = self.find_bad_bonds(z_thresh=z_thresh)
        bad_angle_df = self.find_bad_angles(z_thresh=z_thresh)

        return clash_df, bad_bond_df, bad_angle_df

    def _get_is_backbone_atoms_mask(
        self,
        protein_chains: list[str],
        nuc_chains: list[str],
        chain_ids: list[str],
        atom_names: list[str],
    ) -> np.ndarray:
        is_bb_atom_mask = (
            np.isin(chain_ids, protein_chains) & np.isin(atom_names, PROTEIN_BACKBONE)
        ) | (np.isin(chain_ids, nuc_chains) & np.isin(atom_names, NUC_BACKBONE))
        return is_bb_atom_mask

    def get_valid_atom_mask(
        self,
        clash_tolerance: float = 1.5,
        disulfide_clash_tolerance: float = 1.0,
        z_thresh: float = 12.0,
    ) -> np.ndarray:
        """
        Compute a per-atom validity mask based on stereochemical violations.

        Atoms involved in clashes, bad bonds, or bad angles (as detected by the
        corresponding validators) trigger residue-level invalidation with the
        following rules:

        - If any backbone atom (N, CA, C, O) is involved in an issue,
            the entire residue is marked as invalid.
        - Otherwise, if any sidechain atom is involved, only the sidechain
            atoms of that residue are marked as invalid.

        Args:
            clash_tolerance: Distance tolerance used in clash detection.
            disulfide_clash_tolerance: Distance tolerance used in disulfide clash detection.
            z_thresh: Z-score threshold for bond and angle outliers.

        Returns:
            A boolean array of shape (N_atoms,) where True indicates that the
            atom is valid.
        """
        # Collect all violations
        clash_df, bad_bond_df, bad_angle_df = self.find_all_violations(
            clash_tolerance=clash_tolerance,
            disulfide_clash_tolerance=disulfide_clash_tolerance,
            z_thresh=z_thresh,
        )

        n_atoms = len(self.atom_array)

        # Per-atom basic info
        atom_array = self.atom_array
        all_res_ids = atom_array.res_id
        all_atom_names = atom_array.atom_name
        all_chain_ids = self.struct.uni_chain_id

        # Determine which chains are protein vs nucleotide chains
        entity_id_to_chain_ids = self.struct.get_entity_id_to_chain_ids()
        protein_chains = []
        nuc_chains = []
        for k, v in self.struct.entity_poly_type.items():
            if v == PROTEIN:
                protein_chains.extend(entity_id_to_chain_ids[k])
            elif v in {DNA, RNA}:
                nuc_chains.extend(entity_id_to_chain_ids[k])

        # Backbone / sidechain classification at atom level
        all_is_bb = self._get_is_backbone_atoms_mask(
            protein_chains=protein_chains,
            nuc_chains=nuc_chains,
            chain_ids=all_chain_ids,
            atom_names=all_atom_names,
        )
        all_is_side_chain = ~all_is_bb

        issue_mask = np.zeros(n_atoms, dtype=bool)
        if not clash_df.empty:
            clash_idx = np.concatenate(
                [
                    clash_df["idx1"].to_numpy(dtype=int),
                    clash_df["idx2"].to_numpy(dtype=int),
                ]
            )
            issue_mask[clash_idx] = True

        if not bad_bond_df.empty:
            bond_idx = np.concatenate(
                [
                    bad_bond_df["idx1"].to_numpy(dtype=int),
                    bad_bond_df["idx2"].to_numpy(dtype=int),
                ]
            )
            issue_mask[bond_idx] = True

        if not bad_angle_df.empty:
            angle_idx = np.concatenate(
                [
                    bad_angle_df["idx_a"].to_numpy(dtype=int),
                    bad_angle_df["idx_b"].to_numpy(dtype=int),
                    bad_angle_df["idx_c"].to_numpy(dtype=int),
                ]
            )
            issue_mask[angle_idx] = True

        valid_atom_mask = ~issue_mask
        if not issue_mask.any():
            # No violations at all -> keep everything
            return valid_atom_mask

        # Iterate only over residues that have issues
        issue_idx = np.where(issue_mask)[0]
        issue_chain_ids = all_chain_ids[issue_idx]
        issue_res_ids = all_res_ids[issue_idx].astype(int)

        issue_pairs = zip(issue_chain_ids, issue_res_ids)
        polymer_chain_ids = set(protein_chains + nuc_chains)
        for chain_id, resid in set(issue_pairs):

            if chain_id not in polymer_chain_ids:
                # non-polymer -> drop atoms that have issues
                # already dropped -> skip
                continue

            res_mask = (all_chain_ids == chain_id) & (all_res_ids == resid)

            res_bb_mask = res_mask & all_is_bb
            res_sc_mask = res_mask & all_is_side_chain

            has_bb_issue = (issue_mask & res_bb_mask).any()
            has_sc_issue = (issue_mask & res_sc_mask).any()

            # Backbone issues -> drop entire residue
            if has_bb_issue:
                valid_atom_mask[res_mask] = False
                continue

            # Only sidechain issues -> drop sidechain atoms
            if has_sc_issue:
                valid_atom_mask[res_sc_mask] = False
        return valid_atom_mask
