from pathlib import Path

import numpy as np
import numpy.typing as npt
from gridData import OpenDX
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry.rdGeometry import Point3D, UniformGrid3D
from scipy.spatial.distance import cdist

from quartz.data import MoleculeData, Grid

def get_grid_size_for_dataset(
    mol_data: list[MoleculeData], size: float, padding: float
) -> tuple[list[list[float]], float, float, float]:
    """
    Compute the bounding box and dimensions of a 3D grid that encompasses a dataset of molecules.

    Parameters:
        mol_data (list[MoleculeData]): List of MoleculeData objects, each containing atomic coordinates.
        size (float): The voxel size for the grid.
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
    mins = np.minimum.reduce([mol.coordinates.min(axis=0) for mol in mol_data])
    maxs = np.maximum.reduce([mol.coordinates.max(axis=0) for mol in mol_data])

    xyzs: list[list[float]] = [
        [mins[0], maxs[0]],
        [mins[1], maxs[1]],
        [mins[2], maxs[2]],
    ]

    x_dist, y_dist, z_dist = maxs - mins

    return xyzs, x_dist, y_dist, z_dist


# TODO - parse xtb_esp.dat
# Format: x, y, z, val


# FIXME - Check if it works lol, AI thingy
def read_cube_to_array(
    cubefile: str, target_array: npt.NDArray, origin: tuple[float, float, float]
) -> npt.NDArray:
    """
    Parse a Gaussian cube (.cub) file and insert values into a NumPy array.

    Parameters
    ----------
    cubefile : str
        Path to the .cub file.
    target_array : np.ndarray
        Array into which volumetric data will be inserted.
        Must be >= in size compared to cube grid.
    origin : tuple(float, float, float)
        Origin of `target_array` in cartesian coordinates (x0, y0, z0).

    Returns
    -------
    np.ndarray
        target_array with the cube data inserted at the correct location.
    """
    with open(cubefile, "r") as f:
        # Skip title lines
        f.readline()
        f.readline()

        # Number of atoms and file grid origin
        natoms, ox, oy, oz = map(float, f.readline().split())
        natoms = int(natoms)

        # Grid dimensions and spacing, _ = 0.000
        nx, vx_x, _ = map(float, f.readline().split())
        ny, _, vy_y, _ = map(float, f.readline().split())
        nz, _, _, vz_z = map(float, f.readline().split())
        nx, ny, nz = int(nx), int(ny), int(nz)

        # Skip atomic coordinates
        for _ in range(natoms):
            f.readline()

        # Read the volumetric data
        data = []
        for line in f:
            data.extend(line.split())
        data = np.array(data, dtype=float).reshape((nx, ny, nz), order="C")

    # Calculate relative insertion index into target_array
    ox, oy, oz = np.array([ox, oy, oz])
    gx, gy, gz = np.array(origin)

    ix = int(round((ox - gx) / vx_x))
    iy = int(round((oy - gy) / vy_y))
    iz = int(round((oz - gz) / vz_z))

    tx, ty, tz = target_array.shape

    # Insert cube data into target array
    target_array[ix : ix + nx, iy : iy + ny, iz : iz + nz] = data

    return target_array


def create_grid_from_cub():
    pass


# TODO - instead of 0.0 use None, maybe saves memory, sparse matrices
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

    Returns:
        list[Grid]: List of grids, one for each molecule.
    """

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
    xyzs, x_dist, y_dist, z_dist = get_grid_size_for_dataset(mol_data, size, padding)
    origin = (
        xyzs[0][0] - padding,
        xyzs[1][0] - padding,
        xyzs[2][0] - padding,
    )

    # TODO - padding per axis
    x_len = np.ceil(x_dist + 2.0 * padding)
    y_len = np.ceil(y_dist + 2.0 * padding)
    z_len = np.ceil(z_dist + 2.0 * padding)
    print("Grid lengths: ", x_len, y_len, z_len)
    print("Origin: ", *origin)

    x_gridcount = int(np.ceil(x_len / size))
    y_gridcount = int(np.ceil(y_len / size))
    z_gridcount = int(np.ceil(z_len / size))
    print("Cell count: ", x_gridcount, y_gridcount, z_gridcount)

    grid_props = {
        "bounds_x": xyzs[0],  # [min_x, max_x]
        "bounds_y": xyzs[1],  # [min_y, max_y]
        "bounds_z": xyzs[2],  # [min_z, max_z]
        "origin": origin,
        "size": size,
        "padding": padding,
    }

    prop_map_: list[tuple[int, str]] = list(enumerate(props))

    # Pre-calculate grid points using broadcasting
    np_xs = np.arange(origin[0], origin[0] + x_len, size, dtype=np.float32)
    np_ys = np.arange(origin[1], origin[1] + y_len, size, dtype=np.float32)
    np_zs = np.arange(origin[2], origin[2] + z_len, size, dtype=np.float32)

    # We construct grid_points which is (N_grid, 3)
    np_coords = np.meshgrid(np_xs, np_ys, np_zs, indexing="ij")
    # grid_points: Shape (N_grid_points, 3)
    grid_points: npt.NDArray[np.float32] = np.vstack(
        [np_coords[0].ravel(), np_coords[1].ravel(), np_coords[2].ravel()]
    ).T.astype(np.float32)

    grids = []
    for mol in mol_data:
        grid_prop_dict = {}
        prop_map: enumerate[str] = enumerate(props)
        # NOTE - not sure how reliable this is, possible solution is to recenter geometry from 0,0,0
        rd_grid = UniformGrid3D(
            x_len,
            y_len,
            z_len,
            size,
            offSet=Point3D(*origin),
        )
        print("Processing: ", mol.smiles)

        mol_points = [Point3D(j, k, l) for j, k, l in mol.coordinates]
        # np_point_coords: Shape (N_atoms, 3)
        np_point_coords: npt.NDArray[np.float32] = np.array(
            [[j, k, l] for j, k, l in mol.coordinates], dtype=np.float32
        )

        point_idxs: list[tuple[int]] = []
        for j in mol_points:
            idx = rd_grid.GetGridPointIndex(j)
            point_idxs.append(rd_grid.GetGridIndices(idx))

        props_np = []
        rdMol: Mol = mol.rdmol
        for j, key in prop_map:
            temp = []
            for idx in range(rdMol.GetNumAtoms()):
                prop_val = float(rdMol.GetAtomWithIdx(idx).GetProp(key))
                temp.append(prop_val)
            props_np.append(temp)

        # props_np: Shape (N_props, N_atoms)
        props_np = np.array(props_np, dtype=np.float32)

        # Calculate distances using scipy.spatial.distance.cdist
        # grid_points: (N_grid, 3)
        # np_point_coords: (N_atoms, 3)
        # distances: Shape (N_grid, N_atoms)
        distances: npt.NDArray[np.float32] = cdist(grid_points, np_point_coords).astype(
            np.float32
        )

        for j, key in prop_map_:
            atom_props: npt.NDArray[np.float32] = props_np[j, :]  # Shape (N_atoms,)

            # 1. Mask cutoff
            # mask: Shape (N_grid, N_atoms)
            mask = distances < cutoff_dist

            # 2. Clamp distances
            safe_distances = np.where(distances < clamp_dist, clamp_dist, distances)

            # 3. Calculate falloff
            # falloff: Shape (N_grid, N_atoms)
            falloff: npt.NDArray[np.float32] = 1.0 / (
                np.power(safe_distances, falloff_exp) + 1e-10
            )

            # 4. Cap falloff at |atom_prop|
            abs_atom_props = np.abs(atom_props)
            falloff = np.minimum(falloff, abs_atom_props)

            # 5. Calculate contributions
            # contributions: Shape (N_grid, N_atoms)
            contributions: npt.NDArray[np.float32] = atom_props * falloff

            # 6. Clip contributions based on sign of atom_prop
            min_vals = np.minimum(0, atom_props)
            max_vals = np.maximum(0, atom_props)
            contributions = np.clip(contributions, min_vals, max_vals)

            # 7. Apply mask (distances < cutoff)
            contributions[~mask] = 0.0

            # 8. Sum contributions across atoms for each grid point
            # property_grid_flat: Shape (N_grid,)
            property_grid_flat = np.sum(contributions, axis=1)

            # 9. Reshape to grid
            # property_grid: Shape (x_gridcount, y_gridcount, z_gridcount)
            property_grid = property_grid_flat.reshape(
                x_gridcount, y_gridcount, z_gridcount
            ).astype(np.float16)

            # Apply threshold per property and store
            property_grid[np.abs(property_grid) < threshold[key]] = 0.0
            grid_prop_dict[key] = property_grid

        grids.append(
            Grid(
                grid_prop_dict,
                rd_grid,
                prop_map_,
                mol_points,
                point_idxs,
                grid_props,
                mol_data=mol,
            )
        )

    return grids


def to_dxgrid(
    grids: list[Grid]
    | Grid
    | npt.NDArray
    | dict[str, npt.NDArray]
    | list[dict[str, npt.NDArray] | Grid],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: float = 0.5,
    filename: str = "output",
):
    grids_dir = Path("grids")
    grids_dir.mkdir(exist_ok=True)

    if isinstance(grids, np.ndarray):
        val = grids
        if val.ndim != 3:
            raise ValueError("Input array must be 3D")

        origin_val = np.array(origin)
        delta = np.diag([spacing, spacing, spacing])

        dx = OpenDX.field(filename)
        dx.add("positions", OpenDX.gridpositions(1, val.shape, origin_val, delta))
        dx.add("connections", OpenDX.gridconnections(2, val.shape))
        dx.add("data", OpenDX.array(3, val))

        if not filename.endswith(".dx"):
            filename += ".dx"
        output_file = grids_dir / filename
        dx.write(str(output_file))
        return

    if not isinstance(grids, list):
        grids = [grids]

    for i, grid in enumerate(grids):
        npgrid = None
        origin_val = None
        delta = None
        mol_name = ""
        mol_smiles = ""

        if hasattr(grid, "npGrid"):
            # It is a Grid object
            npgrid = grid.npGrid
            props = grid.grid_props
            origin_val = np.array(props["origin"])
            delta = np.diag([props["size"], props["size"], props["size"]])
            mol_name = grid.mol_data.filename
            mol_smiles = grid.mol_data.smiles
        elif isinstance(grid, dict):
            # It is a dictionary of arrays (e.g. from grid_stddev)
            npgrid = grid
            origin_val = np.array(origin)
            delta = np.diag([spacing, spacing, spacing])
            mol_name = f"{filename}_{i}" if len(grids) > 1 else filename
            mol_smiles = filename
        else:
            print(f"Warning: Skipping unsupported grid type: {type(grid)}")
            continue

        for key, val in npgrid.items():
            dx = OpenDX.field(mol_smiles)
            dx.add("positions", OpenDX.gridpositions(1, val.shape, origin_val, delta))
            dx.add("connections", OpenDX.gridconnections(2, val.shape))
            dx.add("data", OpenDX.array(3, val))

            output_filename = f"{key}_{mol_name}.dx"
            output_file = grids_dir / output_filename
            dx.write(str(output_file))

            if hasattr(grid, "dxGrid"):
                grid.dxGrid = dx
