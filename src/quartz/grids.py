import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry.rdGeometry import Point3D, UniformGrid3D

from quartz.io import Grid, MoleculeData


def get_grid_size_for_dataset(
    mol_data: list[MoleculeData], size: float, padding: float
) -> tuple[list[list[float]], float, float, float]:
    """
    Input: path to files with molecules
    Output: origin point in xyz space, grid dist
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
            xyzs[0][1] = np.max(xs) if np.max(xs) < xyzs[0][1] else xyzs[0][1]
            xyzs[1][0] = np.min(ys) if np.min(ys) < xyzs[1][0] else xyzs[1][0]
            xyzs[1][1] = np.max(ys) if np.max(ys) < xyzs[1][1] else xyzs[1][1]
            xyzs[2][0] = np.min(zs) if np.min(zs) < xyzs[2][0] else xyzs[2][0]
            xyzs[2][1] = np.max(zs) if np.max(zs) < xyzs[2][1] else xyzs[2][1]

    x_dist: float = np.round(np.abs(xyzs[0][0] - xyzs[0][1]), 3)
    y_dist: float = np.round(np.abs(xyzs[1][0] - xyzs[1][1]), 3)
    z_dist: float = np.round(np.abs(xyzs[2][0] - xyzs[2][1]), 3)

    # print(xyzs)

    return xyzs, x_dist, y_dist, z_dist


# TODO - this whole thing needs refactoring
# TODO - threshold per prop
def create_grid(
    mol_data: list[MoleculeData],
    props: list[str],
    size: float = 1.0,
    padding: float = 5.0,
    threshold: float = 0.01,
):
    xs, ys, zs = None, None, None
    if len(mol_data) == 1:
        xs, ys, zs = (
            mol_data[0].coordinates[:, 0],
            mol_data[0].coordinates[:, 1],
            mol_data[0].coordinates[:, 2],
        )
        # FIXME - this honestly can break, revise
        x_dist = np.round(np.abs(np.max(xs) - np.min(xs)), 3)
        y_dist = np.round(np.abs(np.max(ys) - np.min(ys)), 3)
        z_dist = np.round(np.abs(np.max(zs) - np.min(zs)), 3)
        print("Molecule lengths: ", x_dist, y_dist, z_dist)
        origin = (
            np.round(np.min(xs) - padding, 1),
            np.round(np.min(ys) - padding, 1),
            np.round(np.min(zs) - padding, 1),
        )
    else:
        xyzs, x_dist, y_dist, z_dist = get_grid_size_for_dataset(
            mol_data, size, padding
        )
        origin = (
            np.round(xyzs[0][0] - padding, 1),
            np.round(xyzs[1][0] - padding, 1),
            np.round(xyzs[2][0] - padding, 1),
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
        padding
    ]

    prop_map_: list[tuple[int, str]] = list(enumerate(props))

    grids = []
    for i in mol_data:
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
        print("Processing: ", i.smiles)

        mol_points = [Point3D(j, k, l) for j, k, l in i.coordinates]
        np_point_coords = np.array(
            [[j, k, l] for j, k, l in i.coordinates], dtype=np.float16
        )
        # print('Mol points: ', [j.x for j in mol_points])
        # print(np_point_coords)

        np_xs = np.arange(origin[0], origin[0] + x_len, size, dtype=np.float16)
        np_ys = np.arange(origin[1], origin[1] + y_len, size, dtype=np.float16)
        np_zs = np.arange(origin[2], origin[2] + z_len, size, dtype=np.float16)
        # print("np shapes: ", np_xs.shape, np_ys.shape, np_zs.shape)

        np_coords = np.meshgrid(np_xs, np_ys, np_zs)
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
        mol: Mol = i.rdmol
        for j, key in prop_map:
            # print(i, key)
            temp = []
            for idx, coords in zip(range(mol.GetNumAtoms()), point_idxs):
                prop_val = float(mol.GetAtomWithIdx(idx).GetProp(key))
                # print(idx, coords)
                # print(prop_val)
                # grid[*coords, j] = prop_val # Old way to do this
                temp.append(prop_val)
            props_np.append(temp)

        props_np = np.array(props_np, dtype=np.float16)
        # print("props_np shape: ", props_np.shape)
        # NOTE - reshaping so we can broadcast
        reshaped_point_coords = np_point_coords[
            :, np.newaxis, :
        ]  # Shape: (n_atoms, 1, 3)
        reshaped_grid_points = grid_points[
            np.newaxis, :, :
        ]  # Shape: (1, n_grid_points, 3)

        distances = np.linalg.norm(reshaped_point_coords - reshaped_grid_points, axis=2)
        distances = np.maximum(distances, 0.1)
        falloff = 1.0 / (np.power(distances, 6) + np.finfo(np.float16).eps)

        for j, key in prop_map_:
            property_grid = np.zeros(
                (x_gridcount, y_gridcount, z_gridcount), dtype=np.float16
            )

            # Calculate contributions for each atom and add to grid
            # NOTE - add some better falloff scaling at close distances
            for atom_idx, atom_point in enumerate(np_point_coords):
                atom_prop = props_np[j, atom_idx]
                atom_falloff = falloff[atom_idx, :]
                # print(f"{atom_idx}: ", atom_prop)

                point_contributions = atom_prop * atom_falloff
                point_contributions = np.minimum(point_contributions, atom_prop)
                # print(point_contributions.shape)
                # print(np.max(point_contributions))

                # NOTE - apparently meshgrid indexes weirdly so we reshape with y, x, z
                point_contributions_shaped = point_contributions.reshape(
                    y_gridcount, x_gridcount, z_gridcount
                )
                point_contributions_shaped = np.transpose(
                    point_contributions_shaped, (1, 0, 2)
                )

                property_grid += point_contributions_shaped

            # Apply threshold and store
            property_grid[property_grid < threshold] = 0.0
            grid_prop_dict[key] = property_grid

        grids.append(
            Grid(
                grid_prop_dict,
                rd_grid,
                prop_map_,
                mol_points,
                point_idxs,
                grid_props,
                mol_data=i,
            )
        )

    return grids
