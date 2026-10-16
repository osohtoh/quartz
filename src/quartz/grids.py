from pathlib import Path

import numpy as np
import numpy.typing as npt
from gridData import OpenDX
from rdkit.Chem.rdchem import Mol
from scipy.spatial import KDTree

from quartz.data import Grid, MoleculeData


def get_grid_size_for_dataset(
    mol_data: list[MoleculeData], padding: float
) -> tuple[list[list[float]], float, float, float]:
    """
    Compute the bounding box and dimensions of a 3D grid that encompasses a dataset of molecules.

    Parameters:
        mol_data (list[MoleculeData]): List of MoleculeData objects, each containing atomic coordinates.
        padding (float): Padding to add to each dimension of the grid (in the same units as coordinates).

    Returns:
        tuple[list[list[float]], float, float, float]:
            - xyzs: List of [min, max] pairs for x, y, and z dimensions, representing the bounding box.
            - x_dist: Distance (float) covered by the grid in the x dimension.
            - y_dist: Distance (float) covered by the grid in the y dimension.
            - z_dist: Distance (float) covered by the grid in the z dimension.
    """
    if len(mol_data) == 0:
        raise ValueError("mol_data must not be empty")

    # Vectorized min/max values
    mins = (
        np.minimum.reduce([mol.coordinates.min(axis=0) for mol in mol_data]) - padding
    )
    maxs = (
        np.maximum.reduce([mol.coordinates.max(axis=0) for mol in mol_data]) + padding
    )

    xyzs: list[list[float]] = [
        [mins[0], maxs[0]],
        [mins[1], maxs[1]],
        [mins[2], maxs[2]],
    ]

    x_dist, y_dist, z_dist = maxs - mins

    return xyzs, x_dist, y_dist, z_dist


# TODO - parse xtb_esp.dat
# Format: x, y, z, val


# FIXME - maximum values on the grid can exceed the maximum values
# of atomic polarizability, probably due to multiple atoms contributing,
def create_grid(
    mol_data: list[MoleculeData],
    props: list[str],
    size: float = 1.0,
    padding: float = 5.0,
    threshold: dict[str, float] | float = 0.01,
    cutoff_dist: float = 10.0,
    clamp_dist: float = 0.3,
    falloff_exp: float = 12.0,
) -> list[Grid]:
    """
    Creates a grid for a given list of molecules and properties.

    Args:
        mol_data (list[MoleculeData]): List of molecules to create the grid for.
        props (list[str]): List of properties to include in the grid.
        size (float, optional): Voxel size. Defaults to 1.0.
        padding (float, optional): Padding around the grid. Defaults to 5.0.
        threshold (dict[str, float] | float, optional): Threshold for property values, either a single float or a dict per property. Defaults to 0.01.
        cutoff_dist (float, optional): Maximum distance for atom contributions. Defaults to 10.0.
        clamp_dist (float, optional): Minimum distance (clamped) to avoid singularities. Defaults to 0.3.
        falloff_exp (float, optional): Exponent for distance falloff. Defaults to 12.0.

    Returns:
        list[Grid]: List of grids, one for each molecule.
    """

    # i forgot to actually use it... no wonder the grids have been fat
    if isinstance(threshold, float):
        # All props have the same cutoff threshold
        threshold = {prop: threshold for prop in props}
    elif isinstance(threshold, dict):
        # All props have a threshold, default to 0.01 if missing
        for prop in props:
            if prop not in threshold:
                threshold[prop] = 0.01
    else:
        raise ValueError("Threshold must be a float or a dict[str, float]")

    # Get grid bounds and distances for the dataset
    xyzs, x_dist, y_dist, z_dist = get_grid_size_for_dataset(mol_data, padding)
    origin = (
        xyzs[0][0],
        xyzs[1][0],
        xyzs[2][0],
    )

    # TODO - padding per axis
    x_len = np.ceil(x_dist)
    y_len = np.ceil(y_dist)
    z_len = np.ceil(z_dist)
    print("Grid lengths: ", x_len, y_len, z_len)
    print("Origin: ", *origin)

    x_gridcount = int(np.ceil(x_len / size))
    y_gridcount = int(np.ceil(y_len / size))
    z_gridcount = int(np.ceil(z_len / size))
    print("Cell count: ", x_gridcount, y_gridcount, z_gridcount)

    n_grid = x_gridcount * y_gridcount * z_gridcount

    grid_props = {
        "bounds_x": xyzs[0],  # [min_x, max_x]
        "bounds_y": xyzs[1],  # [min_y, max_y]
        "bounds_z": xyzs[2],  # [min_z, max_z]
        "origin": origin,
        "size": size,
        "padding": padding,
    }

    prop_map_: list[tuple[int, str]] = list(enumerate(props))

    # Pre-calculate grid points: shape (N_grid, 3)
    np_xs = origin[0] + np.arange(x_gridcount, dtype=np.float32) * size
    np_ys = origin[1] + np.arange(y_gridcount, dtype=np.float32) * size
    np_zs = origin[2] + np.arange(z_gridcount, dtype=np.float32) * size
    gx, gy, gz = np.meshgrid(np_xs, np_ys, np_zs, indexing="ij")
    grid_points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(
        np.float32
    )

    # Build KDTree for grid points once (reused per molecule)
    grid_tree = KDTree(grid_points)

    # FIXME - considering atom contributions results in polarizability maxima being located on bonds, rather than on atoms. Fix and verify results.
    # FIXME - apply cutoff value
    grids = []
    for mol in mol_data:
        print("Processing: ", mol.filename)

        # Atom coordinates: shape (N_atoms, 3)
        np_point_coords = np.asarray(mol.coordinates, dtype=np.float32)

        # Atom grid indices via NumPy, clamped to valid grid bounds
        point_idxs = np.round((np_point_coords - np.array(origin)) / size).astype(int)
        point_idxs[:, 0] = np.clip(point_idxs[:, 0], 0, x_gridcount - 1)
        point_idxs[:, 1] = np.clip(point_idxs[:, 1], 0, y_gridcount - 1)
        point_idxs[:, 2] = np.clip(point_idxs[:, 2], 0, z_gridcount - 1)

        # Vectorized property extraction: shape (N_props, N_atoms)
        rdMol: Mol = mol.rdmol
        n_atoms = rdMol.GetNumAtoms()
        atoms = [rdMol.GetAtomWithIdx(i) for i in range(n_atoms)]
        props_np = np.array(
            [[float(a.GetProp(key)) for a in atoms] for key in props],
            dtype=np.float32,
        )

        # Sparse neighbor query via KDTree: returns a COO sparse matrix of shape
        # (N_grid, N_atoms) containing only distances <= cutoff_dist, avoiding
        # allocation of the full dense distance matrix.
        atom_tree = KDTree(np_point_coords)
        sparse_dist = grid_tree.sparse_distance_matrix(
            atom_tree, cutoff_dist, output_type="coo_matrix"
        )

        # Extract COO data: grid indices, atom indices, distances
        gi = sparse_dist.row  # grid point indices
        ai = sparse_dist.col  # atom indices
        dists = np.asarray(sparse_dist.data, dtype=np.float32)

        # Precompute property-independent falloff
        safe_dists = np.maximum(dists, clamp_dist)
        falloff = 1.0 / (np.power(safe_dists, falloff_exp) + 1e-10)

        grid_prop_dict = {}
        for j, key in prop_map_:
            atom_props = props_np[j, :]  # (N_atoms,)

            # Per-neighbor atom property and capped falloff
            ap = atom_props[ai]
            f = np.minimum(falloff, np.abs(ap))

            # Contributions clipped to sign of atom_prop
            contributions = np.clip(ap * f, np.minimum(0, ap), np.maximum(0, ap))

            # Scatter-add contributions into flat grid
            property_grid_flat = np.zeros(n_grid, dtype=np.float32)
            np.add.at(property_grid_flat, gi, contributions)

            # Reshape and threshold
            property_grid = property_grid_flat.reshape(
                x_gridcount, y_gridcount, z_gridcount
            )
            grid_prop_dict[key] = property_grid

        grids.append(
            Grid(
                npGrid=grid_prop_dict,
                prop_map=prop_map_,
                point_idxs=[tuple(idx) for idx in point_idxs],
                grid_props=grid_props,
                mol_data=mol,
            )
        )

    return grids


def to_dxgrid(
    grids: list[Grid] | Grid | npt.NDArray,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: float = 0.5,
    filename: str = "output",
):
    grids_dir = Path("grids")
    grids_dir.mkdir(exist_ok=True)

    grid_list: list[Grid] = []

    if isinstance(grids, list) and all(isinstance(g, Grid) for g in grids):
        grid_list = grids  # type: ignore
    elif isinstance(grids, Grid):
        grid_list = [grids]
    elif isinstance(grids, np.ndarray):
        if grids.ndim != 3:
            raise ValueError("Input array must be 3D")

        dummy_props = {
            "origin": origin,
            "size": spacing,
        }
        grid_list = [
            Grid(npGrid={"data": grids}, grid_props=dummy_props, mol_data=None)
        ]
    else:
        raise ValueError("Invalid input type")

    for i, grid in enumerate(grid_list):
        props = grid.grid_props

        # Origin
        if props and "origin" in props:
            origin_val = np.array(props["origin"])
        else:
            origin_val = np.array(origin)

        # Spacing/Size / Delta
        if props and "size" in props:
            s = props["size"]
            delta = np.diag([s, s, s])
        else:
            delta = np.diag([spacing, spacing, spacing])

        mol_identifier = ""

        if grid.mol_data:
            mol_identifier = grid.mol_data.filename
        else:
            if len(grid_list) > 1:
                mol_identifier = f"{filename}_{i}"
            else:
                mol_identifier = filename

        for key, val in grid.npGrid.items():
            if val is None:
                continue

            dx = OpenDX.field(mol_identifier)
            dx.add("positions", OpenDX.gridpositions(1, val.shape, origin_val, delta))
            dx.add("connections", OpenDX.gridconnections(2, val.shape))
            dx.add("data", OpenDX.array(3, val))

            safe_identifier = Path(mol_identifier).stem
            output_filename = f"{key}_{safe_identifier}.dx"
            output_file = grids_dir / output_filename

            print(f"Writing {output_file}...")
            dx.write(str(output_file))

            if grid.dxGrid is None:
                grid.dxGrid = dx
