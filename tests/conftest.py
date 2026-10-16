import matplotlib

matplotlib.use("Agg")

import plotly.graph_objects as go
import pytest

from molecule_factory import PATTERNS, PROPS, build_substituted_benzene
from quartz.grids import create_grid

CODES = {
    "benzene": "BEN",
    "chlorobenzene": "CLB",
    "1,2-dichlorobenzene": "D12",
    "1,3-dichlorobenzene": "D13",
    "1,4-dichlorobenzene": "D14",
}
PKIS = {
    "benzene": 5.0,
    "chlorobenzene": 5.5,
    "1,2-dichlorobenzene": 6.0,
    "1,3-dichlorobenzene": 6.5,
    "1,4-dichlorobenzene": 7.0,
}


@pytest.fixture(scope="session")
def substituted_grids():
    """One grid per substituted benzene, plus identical copies of benzene and
    chlorobenzene. All molecules go through a single create_grid call so they
    share one voxel box. Returned dict is keyed by pdb_code."""
    mols = [
        build_substituted_benzene(name, CODES[name], pki=PKIS[name])
        for name in PATTERNS
    ]
    mols.append(build_substituted_benzene("benzene", "BEN_copy", pki=PKIS["benzene"]))
    mols.append(
        build_substituted_benzene(
            "chlorobenzene", "CLB_copy", pki=PKIS["chlorobenzene"]
        )
    )
    grids = create_grid(mols, PROPS, size=1.0, padding=3.0)
    return {g.mol_data.pdb_code: g for g in grids}


@pytest.fixture
def no_show(monkeypatch):
    """Stop plotly figures from opening a browser during tests."""
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None)
