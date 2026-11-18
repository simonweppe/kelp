"""SCHISM reader (readers/schism.py): 2-D depth-averaged velocity on the nodes.

Synthetic SCHISM-like files in tmp_path — no external data. Covers the two
output layouts (combined ``dahv``, split ``depthAverageVelX/Y``), mixed
triangle/quad face tables with SCHISM's padding, geographic and projected
meshes (the grid-to-true rotation measured on the mesh, against a known
rotation), dry values, the variables ``open_dataset`` keeps, the refusals,
and an end-to-end run.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from kelp.readers import get_reader

R = 6_371_000.0
LON0, LAT0 = 176.0, -37.6


def _faces(nx, ny, quad_first=False):
    """1-based SCHISM face table (E, 4), triangles padded with -1."""
    rows = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i + 1
            b, c, d = a + 1, a + nx + 1, a + nx
            if quad_first and i == 0 and j == 0:
                rows.append([a, b, c, d])
            else:
                rows += [[a, b, c, -1], [a, c, d, -1]]
    return np.array(rows, dtype=np.int32)


def _write(path, x, y, faces, uv, *, units, lonlat=None, layout="dahv",
           extra_3d=True, coord_dtype=np.float64):
    n_time, n = uv.shape[0], x.size
    t = np.arange(n_time) * 1800.0
    ds = xr.Dataset(coords={"time": ("time", t, {"units": "seconds since 2020-01-01"})})
    ds["SCHISM_hgrid_node_x"] = ("nSCHISM_hgrid_node", x.astype(coord_dtype), {"units": units})
    ds["SCHISM_hgrid_node_y"] = ("nSCHISM_hgrid_node", y.astype(coord_dtype), {"units": units})
    ds["SCHISM_hgrid_face_nodes"] = (("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
                                     faces, {"start_index": 1})
    ds["depth"] = ("nSCHISM_hgrid_node", np.full(n, 20.0))
    if lonlat is not None:
        ds["longitude"] = ("nSCHISM_hgrid_node", lonlat[0])
        ds["latitude"] = ("nSCHISM_hgrid_node", lonlat[1])
    if layout == "dahv":
        ds["dahv"] = (("time", "nSCHISM_hgrid_node", "two"), uv.astype(np.float32))
    else:
        ds["depthAverageVelX"] = (("time", "nSCHISM_hgrid_node"), uv[..., 0].astype(np.float32))
        ds["depthAverageVelY"] = (("time", "nSCHISM_hgrid_node"), uv[..., 1].astype(np.float32))
    if extra_3d:
        ds["zcor"] = (("time", "nSCHISM_hgrid_node", "nSCHISM_vgrid_layers"),
                      np.zeros((n_time, n, 5), np.float32))
        ds["elev"] = (("time", "nSCHISM_hgrid_node"), np.zeros((n_time, n), np.float32))
    ds.to_netcdf(path)
    return path


def _geographic(tmp_path, uv_true=(0.3, -0.1), n_time=4, layout="dahv", nx=5, ny=4,
                quad_first=False):
    lon = LON0 + 0.01 * np.arange(nx)
    lat = LAT0 + 0.01 * np.arange(ny)
    X, Y = np.meshgrid(lon, lat)
    uv = np.broadcast_to(np.array(uv_true, float), (n_time, X.size, 2)).copy()
    return _write(tmp_path / "geo.nc", X.ravel(), Y.ravel(), _faces(nx, ny, quad_first), uv,
                  units="degrees", layout=layout)


def _projected(tmp_path, alpha_deg, uv_true=(0.6, 0.2), n_time=3, step=500.0, nx=9, ny=7,
               coord_dtype=np.float64):
    """Grid axes turned by alpha from east/north: [x; y] = R(-alpha) [e; n] + offset,
    and the velocity written in the grid frame the same way."""
    e, n = np.meshgrid(step * np.arange(nx), step * np.arange(ny))
    e, n = e.ravel(), n.ravel()
    a = np.radians(alpha_deg)
    x = np.cos(a) * e + np.sin(a) * n + 1.8e6
    y = -np.sin(a) * e + np.cos(a) * n + 5.8e6
    lon = LON0 + np.degrees(e / (R * np.cos(np.radians(LAT0))))
    lat = LAT0 + np.degrees(n / R)
    ue, vn = uv_true
    ug = np.cos(a) * ue + np.sin(a) * vn
    vg = -np.sin(a) * ue + np.cos(a) * vn
    uv = np.broadcast_to(np.array([ug, vg]), (n_time, x.size, 2)).copy()
    return _write(tmp_path / "proj.nc", x, y, _faces(nx, ny), uv, units="m",
                  lonlat=(lon, lat), coord_dtype=coord_dtype)


def _mode(**kw):
    base = dict(Fdrift3D=False, sample_vars=None, var_u=None, var_v=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _load(path, mode=None):
    rd = get_reader("schism")
    mesh = rd.load_mesh(SimpleNamespace(meshpath=str(path), rotation=None))
    ds = rd.open_dataset(str(path), mode_cfg=mode or _mode())
    return rd, mesh, ds


def test_registered():
    assert get_reader("schism").__name__ == "SchismReader"


def test_geographic_dahv_passthrough_node_layout(tmp_path):
    rd, mesh, ds = _load(_geographic(tmp_path))
    assert mesh.n_nodes == 20 and mesh.n_elements == 24
    assert not mesh.metadata["projected"]
    vf = rd.read_velocities(ds, 2, mesh)
    assert vf.uv_layout == "node"
    np.testing.assert_allclose(vf.u, 0.3, atol=1e-6)
    np.testing.assert_allclose(vf.v, -0.1, atol=1e-6)
    assert rd.probe_uv_layout(ds, _mode(), mesh) == "node"


def test_open_dataset_keeps_only_velocity(tmp_path):
    # prefetch windows load every time-dependent variable: zcor/elev must not ride along
    _, _, ds = _load(_geographic(tmp_path))
    assert set(ds.data_vars) == {"dahv"}


def test_split_layout_depth_average_xy(tmp_path):
    rd, mesh, ds = _load(_geographic(tmp_path, layout="split"))
    assert set(ds.data_vars) == {"depthAverageVelX", "depthAverageVelY"}
    vf = rd.read_velocities(ds, 0, mesh)
    np.testing.assert_allclose(vf.u, 0.3, atol=1e-6)
    np.testing.assert_allclose(vf.v, -0.1, atol=1e-6)


def test_var_u_var_v_override(tmp_path):
    path = _geographic(tmp_path, layout="split")
    rd, mesh, ds = _load(path, _mode(var_u="depthAverageVelX", var_v="depthAverageVelY"))
    assert rd.read_velocities(ds, 1, mesh).u[0] == pytest.approx(0.3, abs=1e-6)
    with pytest.raises(ValueError, match="both"):
        _load(path, _mode(var_u="depthAverageVelX"))


def test_mixed_triangles_and_quads(tmp_path):
    _, mesh, _ = _load(_geographic(tmp_path, quad_first=True))
    assert mesh.metadata["n_quads"] == 1
    assert mesh.metadata["n_triangles"] == mesh.n_elements - 1
    assert mesh.elem_nodes.min() == -1                  # 0-based, -1 padded
    assert mesh.elem_nodes.max() == mesh.n_nodes - 1


@pytest.mark.parametrize("alpha", [-3.0, 0.0, 2.5, 30.0])
def test_projected_rotation_recovers_east_north(tmp_path, alpha):
    rd, mesh, ds = _load(_projected(tmp_path, alpha))
    assert mesh.metadata["projected"]
    vf = rd.read_velocities(ds, 1, mesh)
    np.testing.assert_allclose(vf.u, 0.6, atol=2e-6)
    np.testing.assert_allclose(vf.v, 0.2, atol=2e-6)


def test_projected_float32_coordinates_still_rotate(tmp_path):
    # x/y in float32 at ~10^6 m (as SCHISM products ship) step by 0.125-0.5 m:
    # a 200 m element's own angle is off by ~0.1 deg, the fitted rotation by
    # ~0.006 deg here (6e-5 m/s on 0.63 m/s). The fit's error falls with the
    # elements' size and number, so a tiny mesh of small elements cannot be
    # held to this tolerance -- real meshes have 10^4-10^6 elements.
    rd, mesh, ds = _load(_projected(tmp_path, 2.0, step=200.0, nx=21, ny=17,
                                    coord_dtype=np.float32))
    vf = rd.read_velocities(ds, 0, mesh)
    np.testing.assert_allclose(vf.u, 0.6, atol=2e-4)
    np.testing.assert_allclose(vf.v, 0.2, atol=2e-4)


def test_projected_without_lonlat_is_refused(tmp_path):
    path = _projected(tmp_path, 1.0)
    with xr.open_dataset(path) as ds:
        bare = ds.drop_vars(["longitude", "latitude"]).load()
    bare.to_netcdf(tmp_path / "bare.nc")
    with pytest.raises(ValueError, match="longitude/latitude"):
        _load(tmp_path / "bare.nc")


def test_dry_nan_reads_as_zero(tmp_path):
    path = _geographic(tmp_path)
    with xr.open_dataset(path) as ds:
        d = ds.load()
    d["dahv"][:, 3, :] = np.nan
    d.to_netcdf(tmp_path / "dry.nc")
    rd, mesh, ds = _load(tmp_path / "dry.nc")
    vf = rd.read_velocities(ds, 0, mesh)
    assert vf.u[3] == 0.0 and vf.v[3] == 0.0 and np.isfinite(vf.u).all()


def test_lazy_and_loaded_reads_identical(tmp_path):
    rd, mesh, ds = _load(_projected(tmp_path, 2.0))
    a = rd.read_velocities(ds, 2, mesh)
    b = rd.read_velocities(ds.load(), 2, mesh)
    np.testing.assert_array_equal(a.u, b.u)
    np.testing.assert_array_equal(a.v, b.v)


def test_refusals(tmp_path):
    path = _geographic(tmp_path)
    rd = get_reader("schism")
    with pytest.raises(ValueError, match="3-D"):
        rd.open_dataset(str(path), mode_cfg=_mode(Fdrift3D=True))
    with pytest.raises(ValueError, match="sample_vars"):
        rd.open_dataset(str(path), mode_cfg=_mode(sample_vars=["temp"]))
    with pytest.raises(ValueError, match="rotation"):
        rd.load_mesh(SimpleNamespace(meshpath=str(path), rotation="fesom2"))


def test_end_to_end_uniform_flow(tmp_path):
    from kelp.core.config import load_config
    from kelp.tracking import run_tracking
    path = _geographic(tmp_path, uv_true=(0.1, 0.0), n_time=4, nx=8, ny=6)
    seeds = tmp_path / "seeds.txt"
    seeds.write_text(f"{LON0 + 0.02} {LAT0 + 0.02}\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""mode: {{Fdrift2D: true}}
numerics: {{integrator: rk2, dt: 600, temporal_interp: true, uv_interp: barycentric}}
mesh: {{type: schism, meshpath: {path}}}
data: {{ncinput: {path}, inicloud: {seeds}, output_path: "", time_averaging: instant}}
time: {{start_time_idx: 0, n_steps: 6}}
""")
    a = run_tracking(load_config(str(cfg)))[0].to_arrays()
    dlon_m = (a["lon"][-1, 0] - (LON0 + 0.02)) * 111_120.0 * np.cos(np.radians(a["lat"][-1, 0]))
    assert dlon_m == pytest.approx(0.1 * 3600.0, rel=2e-3)
    assert a["lat"][-1, 0] == pytest.approx(LAT0 + 0.02, abs=1e-9)
