import numpy as np
import numpy.typing as npt
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry.rdGeometry import Point3D, UniformGrid3D
from gridData import Grid, OpenDX
from pathlib import Path

from quartz.data import Grid, MoleculeData

# NOTE - docstrings need some more love in general


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
    xyzs: list[list[float]] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    for i, molecule in enumerate(mol_data):
        xs, ys, zs = (
            molecule.coordinates[:, 0],
            molecule.coordinates[:, 1],
            molecule.coordinates[:, 2],
        )
        if i == 0:
            xyzs = [
                [np.min(xs), np.max(xs)],
                [np.min(ys), np.max(ys)],
                [np.min(zs), np.max(zs)],
            ]
        # NOTE - there's probably a nicer way to do this
        else:
            xyzs[0][0] = np.min(xs) if np.min(xs) < xyzs[0][0] else xyzs[0][0]
            xyzs[0][1] = np.max(xs) if np.max(xs) > xyzs[0][1] else xyzs[0][1]
            xyzs[1][0] = np.min(ys) if np.min(ys) < xyzs[1][0] else xyzs[1][0]
            xyzs[1][1] = np.max(ys) if np.max(ys) > xyzs[1][1] else xyzs[1][1]
            xyzs[2][0] = np.min(zs) if np.min(zs) < xyzs[2][0] else xyzs[2][0]
            xyzs[2][1] = np.max(zs) if np.max(zs) > xyzs[2][1] else xyzs[2][1]

    x_dist: float = np.abs(xyzs[0][0] - xyzs[0][1])
    y_dist: float = np.abs(xyzs[1][0] - xyzs[1][1])
    z_dist: float = np.abs(xyzs[2][0] - xyzs[2][1])

    # print(xyzs)

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
    clamp_falloff: float = 1000.0,
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
        threshold = {prop: threshold for prop in props}
    elif isinstance(threshold, dict):
        # All props have a threshold, default to 0.01 if missing
        for prop in props:
            if prop not in threshold:
                threshold[prop] = 0.01
    else:
        raise ValueError("Threshold must be a float or a dict[str, float]")

    xs, ys, zs = None, None, None
    if len(mol_data) == 1:
        xs, ys, zs = (
            mol_data[0].coordinates[:, 0],
            mol_data[0].coordinates[:, 1],
            mol_data[0].coordinates[:, 2],
        )
        # FIXME - this honestly can break, revise
        x_dist = np.abs(np.max(xs) - np.min(xs))
        y_dist = np.abs(np.max(ys) - np.min(ys))
        z_dist = np.abs(np.max(zs) - np.min(zs))
        print("Molecule lengths: ", x_dist, y_dist, z_dist)
        origin = (
            np.min(xs) - padding,
            np.min(ys) - padding,
            np.min(zs) - padding,
        )
    else:
        xyzs, x_dist, y_dist, z_dist = get_grid_size_for_dataset(
            mol_data, size, padding
        )
        origin = (
            xyzs[0][0] - padding,
            xyzs[1][0] - padding,
            xyzs[2][0] - padding,
        )
        xs = [xyzs[0][0], xyzs[0][1]]
        ys = [xyzs[1][0], xyzs[1][1]]
        zs = [xyzs[2][0], xyzs[2][1]]

    # TODO - padding per axis
    x_len = np.ceil(x_dist + 2.0 * padding)
    y_len = np.ceil(y_dist + 2.0 * padding)
    z_len = np.ceil(z_dist + 2.0 * padding)
    print("Grid lengths: ", x_len, y_len, z_len)

    print("Origin: ", *origin)
    # init = rd_grid.GetGridPointLoc(0)
    # print("Initial point: ", init.x, init.y, init.z)

    # NOTE - not sure why this is here or if it is even needed
    # EncodeShape(mol_data.rdmol, rd_grid, ignoreHs=False)

    grid_props = []

    x_gridcount = int(np.ceil(x_len / size))
    y_gridcount = int(np.ceil(y_len / size))
    z_gridcount = int(np.ceil(z_len / size))
    print("Cell count: ", x_gridcount, y_gridcount, z_gridcount)

    # NOTE - we assume that all supplied MoleculeData have same amount of props
    prop_count = len(mol_data[0].res.keys())

    grid_props = [
        [
            np.min(xs),  # Molecule start
            np.max(xs),  # Molecule end
        ],
        [
            np.min(ys),
            np.max(ys),
        ],
        [
            np.min(zs),
            np.max(zs),
        ],
        origin,
        size,
        padding,
    ]

    prop_map_: list[tuple[int, str]] = list(enumerate(props))

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
        np_point_coords = np.array(
            [[j, k, l] for j, k, l in mol.coordinates], dtype=np.float16
        )
        # print('Mol points: ', [j.x for j in mol_points])
        # print(np_point_coords)

        np_xs = np.arange(origin[0], origin[0] + x_len, size, dtype=np.float16)
        np_ys = np.arange(origin[1], origin[1] + y_len, size, dtype=np.float16)
        np_zs = np.arange(origin[2], origin[2] + z_len, size, dtype=np.float16)
        # print("np shapes: ", np_xs.shape, np_ys.shape, np_zs.shape)

        np_coords = np.meshgrid(np_xs, np_ys, np_zs, indexing="ij")
        grid_points = np.vstack(
            [np_coords[0].ravel(), np_coords[1].ravel(), np_coords[2].ravel()]
        ).T
        # print("coords shape: ", grid_points.shape)
        # print(grid_points)

        point_idxs: list[tuple[int]] = []
        for j in mol_points:
            idx = rd_grid.GetGridPointIndex(j)
            # print('Point idx: ', idx)
            point_idxs.append(rd_grid.GetGridIndices(idx))

        # print(point_idxs)
        # print(prop_map)

        props_np = []
        rdMol: Mol = mol.rdmol
        for j, key in prop_map:
            # print(i, key)
            temp = []
            for idx, coords in zip(range(rdMol.GetNumAtoms()), point_idxs):
                prop_val = float(rdMol.GetAtomWithIdx(idx).GetProp(key))
                # print(idx, coords)
                # print(prop_val)
                # grid[*coords, j] = prop_val # Old way to do this
                temp.append(prop_val)
            props_np.append(temp)

        props_np = np.array(props_np, dtype=np.float16)
        # print("props_np shape: ", props_np.shape)
        # NOTE - reshaping so we can broadcast
        # reshaped_point_coords = np_point_coords[
        #     :, np.newaxis, :
        # ]  # Shape: (n_atoms, 1, 3)
        # reshaped_grid_points = grid_points[
        #     np.newaxis, :, :
        # ]  # Shape: (1, n_grid_points, 3)

        # distances = np.linalg.norm(reshaped_point_coords - reshaped_grid_points, axis=2)
        # distances = np.maximum(distances, 0.1)
        # falloff = 1.0 / (np.power(distances, 6) + np.finfo(np.float16).eps)
        # falloff = np.exp(-0.5 * (distances / 2.0)**2) # Gaussian
        # falloff = np.exp(-distances / 2.0) # Exponential

        for j, key in prop_map_:
            property_grid = np.zeros(
                (x_gridcount, y_gridcount, z_gridcount), dtype=np.float16
            )

            # Calculate contributions for each atom and add to grid
            # NOTE - add some better falloff scaling at close distances
            for atom_idx, atom_point in enumerate(np_point_coords):
                atom_prop = props_np[j, atom_idx]
                prop_precision = str(atom_prop)[::-1].find(".")
                atom_pos = np_point_coords[atom_idx]
                # print(f"{atom_idx}: ", atom_prop)
                # atom_falloff = falloff[atom_idx, :]

                # Vectorized distance
                dx = grid_points[:, 0] - atom_pos[0]
                dy = grid_points[:, 1] - atom_pos[1]
                dz = grid_points[:, 2] - atom_pos[2]
                distances_sq = dx * dx + dy * dy + dz * dz

                mask = distances_sq < (cutoff_dist**2)
                n_nearby = np.sum(mask)

                if n_nearby == 0:
                    continue

                distances = np.sqrt(distances_sq[mask])
                # Clamp the distances at clamp_dist
                safe_distances = np.where(distances < clamp_dist, clamp_dist, distances)

                falloff = 1.0 / (np.power(safe_distances, 6, dtype=np.float32) + 1e-10)
                falloff = np.minimum(falloff, atom_prop)

                point_contributions = atom_prop * falloff
                point_contributions = np.minimum(point_contributions, atom_prop)
                # print("Max point contributions:", np.max(point_contributions))
                # if np.max(point_contributions) > atom_prop:
                #     raise ValueError("Contribution exceeds atom property value")
                # print(point_contributions.shape)

                full_contributions = np.zeros(grid_points.shape[0], dtype=np.float32)
                full_contributions[mask] = point_contributions

                # NOTE - apparently meshgrid indexes weirdly so we reshape with y, x, z
                # point_contributions_shaped = point_contributions.reshape(
                #     y_gridcount, x_gridcount, z_gridcount
                # )
                # point_contributions_shaped = np.transpose(
                #     point_contributions_shaped, (1, 0, 2)
                # )

                contrib_shaped = full_contributions.reshape(
                    x_gridcount, y_gridcount, z_gridcount
                )
                # property_grid += np.transpose(contrib_shaped, (1, 0, 2)).astype(np.float16)

                property_grid += contrib_shaped.astype(np.float16)
                # print("Max property grid value:", np.max(property_grid))

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


def to_dxgrid(grids: list[Grid]):
    if not isinstance(grids, list):
        grids = [grids]

    for grid in grids:
        npgrid = grid.npGrid
        props = grid.grid_props
        origin = np.array(props[3])
        delta = np.diag([props[4], props[4], props[4]])

        grids_dir = Path("grids")
        grids_dir.mkdir(exist_ok=True)

        for key, val in npgrid.items():
            dx = OpenDX.field(grid.mol_data.smiles)
            dx.add("positions", OpenDX.gridpositions(1, val.shape, origin, delta))
            dx.add("connections", OpenDX.gridconnections(2, val.shape))
            dx.add("data", OpenDX.array(3, val))
            output_file = grids_dir / f"{key}_{grid.mol_data.filename}.dx"
            dx.write(str(output_file))
            grid.dxGrid = dx
