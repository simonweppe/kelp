"""SCHISM model reader (2-D, depth-averaged velocity on the nodes).

SCHISM (Zhang et al. 2016) writes its horizontal grid as UGRID-like arrays
(``SCHISM_hgrid_node_x/_y``, ``SCHISM_hgrid_face_nodes``) and its velocity on
the nodes, so a run tracks node-layout u/v with ``numerics.uv_interp:
barycentric``.  Two output layouts are read, detected from the variable names:

* the combined layout (``schout_*.nc`` and products derived from it): the
  depth-averaged velocity as ``dahv (time, node, two)``;
* the split layout of SCHISM >= 5.10 (``out2d_*.nc``):
  ``depthAverageVelX`` / ``depthAverageVelY (time, node)``.

``mode.var_u`` / ``mode.var_v`` name any other ``(time, node)`` pair.

Coordinates.  A geographic run (``ics = 2``) stores lon/lat in the node
x/y and east/north velocity.  A projected run (``ics = 1``) stores metres and
velocity along the projection's grid axes; the reader then takes the node
positions from ``longitude`` / ``latitude`` (written beside x/y by the usual
post-processing), and turns the velocity from grid north to true north with
the local rotation between the two coordinate sets, measured on the mesh
itself (per element, from the map of grid metres onto local east/north
metres; averaged on the nodes).  No projection library is needed and any
projection is handled; the rotation is computed once, at mesh load.

Efficiency.  ``open_dataset`` keeps the velocity variable(s) and nothing
else, so ``numerics.prefetch_size`` reads one contiguous slab of velocity
per window -- not SCHISM's 3-D ``zcor`` / ``hvel`` next to it.  Each record
then costs one read and a vectorised rotation with cached cos/sin.

Not supported (refused): 3-D tracking, ``mesh.rotation``, ``sample_vars``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from ..core.io import VelocityField
from ..core.mesh import (Mesh, build_node_to_elems, build_search,
                         validate_quad_convexity)
from . import register, _detect_engine, _read_kh_fields

_X, _Y = "SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y"
_FACES = "SCHISM_hgrid_face_nodes"
_NODE_DIM = "nSCHISM_hgrid_node"

#: earth radius for the local east/north metres of the rotation measurement
#: (it cancels in the angle; any value works)
_R = 6_371_000.0


def _is_geographic(x, y, units: str) -> bool:
    if "degree" in units.lower():
        return True
    if units.lower().strip() in ("m", "meter", "meters", "metre", "metres"):
        return False
    return bool(np.nanmax(np.abs(x)) <= 360.0 and np.nanmax(np.abs(y)) <= 90.0)


def _connectivity(ds) -> np.ndarray:
    """0-based (E, 4) int32, -1 padded; triangles and quads."""
    var = ds[_FACES]
    faces = np.asarray(var.to_masked_array().filled(-1)
                       if hasattr(var, "to_masked_array") else var.values)
    faces = np.where(np.isfinite(faces), faces, -1).astype(np.int64)
    start = int(var.attrs.get("start_index", 1))
    conn = np.where(faces >= start, faces - start, -1)
    if conn.shape[1] < 4:
        conn = np.pad(conn, ((0, 0), (0, 4 - conn.shape[1])),
                      constant_values=-1)
    return np.ascontiguousarray(conn[:, :4], dtype=np.int32)


def _elem_centres(conn, lon, lat):
    valid = conn >= 0
    n = valid.sum(axis=1)
    idx = np.where(valid, conn, 0)
    lo, la = lon[idx], lat[idx]
    # anchor on the first vertex and average wrapped deltas: exact off the
    # antimeridian, seam-safe on it
    ref = lo[:, :1]
    d = np.where(valid, ((lo - ref + 180.0) % 360.0) - 180.0, 0.0)
    clon = ((ref[:, 0] + d.sum(axis=1) / n + 180.0) % 360.0) - 180.0
    clat = np.where(valid, la, 0.0).sum(axis=1) / n
    return clon, clat


def _grid_rotation(conn, x, y, lon, lat):
    """cos and sin, per node, of the angle from grid north to true north.

    Per element (first three vertices), the 2x2 map M from grid metres
    (dx, dy) onto local east/north metres (dE, dN); its rotation part is
    atan2(M10 - M01, M00 + M11) (the polar decomposition of a near-conformal
    map).  Nodes take the circular mean of their elements' angles.
    """
    a, b, c = (conn[:, k].astype(np.int64) for k in range(3))
    lat0 = np.radians(lat[a])
    k = np.radians(1.0) * _R
    dE1 = ((lon[b] - lon[a] + 180.0) % 360.0 - 180.0) * k * np.cos(lat0)
    dE2 = ((lon[c] - lon[a] + 180.0) % 360.0 - 180.0) * k * np.cos(lat0)
    dN1 = (lat[b] - lat[a]) * k
    dN2 = (lat[c] - lat[a]) * k
    dx1, dx2 = x[b] - x[a], x[c] - x[a]
    dy1, dy2 = y[b] - y[a], y[c] - y[a]
    det = dx1 * dy2 - dx2 * dy1
    ok = np.abs(det) > 0
    det = np.where(ok, det, 1.0)
    # M = [dE; dN] @ inv([dx; dy])
    m00 = (dE1 * dy2 - dE2 * dy1) / det
    m01 = (-dE1 * dx2 + dE2 * dx1) / det
    m10 = (dN1 * dy2 - dN2 * dy1) / det
    m11 = (-dN1 * dx2 + dN2 * dx1) / det
    theta = np.arctan2(m10 - m01, m00 + m11)
    # The angle of one element is only as good as its vertices: SCHISM files
    # often hold projected x/y in float32 (~0.1 m at 10^6 m), which turns a
    # few-metre harbour element's angle by a degree or more. The rotation of
    # a map projection is smooth over a model domain, so fit it: a cubic in
    # the (centred, scaled) grid coordinates, weighted by element area (the
    # error of an element's angle falls as its size grows).
    area = 0.5 * np.abs(det)
    w = np.where(ok, area, 0.0)
    xc, yc = (x[a] + x[b] + x[c]) / 3.0, (y[a] + y[b] + y[c]) / 3.0
    x0, y0 = np.average(xc, weights=w), np.average(yc, weights=w)
    sc = max(np.ptp(x), np.ptp(y), 1.0)

    def basis(px, py):
        u, v = (px - x0) / sc, (py - y0) / sc
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v,
                         u ** 3, u * u * v, u * v * v, v ** 3], axis=1)

    # unwrap around the weighted circular mean so the fit never meets the seam
    ref = np.arctan2(np.sum(w * np.sin(theta)), np.sum(w * np.cos(theta)))
    t = ((theta - ref + np.pi) % (2 * np.pi)) - np.pi
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(basis(xc, yc) * sw[:, None], t * sw, rcond=None)
    theta_n = ref + basis(x, y) @ coef
    return np.cos(theta_n), np.sin(theta_n)


def _record(da, timestep):
    """One time record of ``da`` as a NumPy array.

    In memory (a prefetch window, or a dataset the caller loaded) with time
    first, as SCHISM writes it: a plain index, no xarray indexing machinery
    (which costs more than the record itself on a 10^5-node mesh). Lazy:
    one read through ``isel``.
    """
    data = da.variable._data if hasattr(da.variable, "_data") else None
    if isinstance(data, np.ndarray) and da.dims and da.dims[0] == "time":
        return data[timestep]
    return np.asarray(da.isel(time=timestep).values)


def _finite(a):
    """float64, dry/missing values (NaN) as 0 -- copied only when needed."""
    a = a.astype(np.float64, copy=False)
    bad = np.isnan(a)
    if bad.any():
        a = np.where(bad, 0.0, a)
    return a


def _uv_names(ds, mode_cfg):
    vu = getattr(mode_cfg, "var_u", None) if mode_cfg else None
    vv = getattr(mode_cfg, "var_v", None) if mode_cfg else None
    if vu or vv:
        if not (vu and vv):
            raise ValueError("SCHISM reader: set both mode.var_u and mode.var_v, or neither")
        return vu, vv
    if "dahv" in ds:
        return "dahv", None
    if "depthAverageVelX" in ds and "depthAverageVelY" in ds:
        return "depthAverageVelX", "depthAverageVelY"
    raise KeyError(
        "SCHISM reader: no depth-averaged velocity found (dahv, or "
        "depthAverageVelX/depthAverageVelY); name the pair with "
        "mode.var_u / mode.var_v")


@register("schism")
class SchismReader:
    @staticmethod
    def load_mesh(mesh_cfg):
        if getattr(mesh_cfg, "rotation", None) is not None:
            raise ValueError("mesh.rotation is not supported by the SCHISM reader")
        path = Path(mesh_cfg.meshpath).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SCHISM mesh file not found: {path}")

        with xr.open_dataset(path, engine=_detect_engine(path),
                             mask_and_scale=True, decode_times=False) as ds:
            for name in (_X, _Y, _FACES):
                if name not in ds:
                    raise KeyError(f"SCHISM mesh {path}: no {name}")
            x = np.asarray(ds[_X].values, dtype=np.float64)
            y = np.asarray(ds[_Y].values, dtype=np.float64)
            conn = _connectivity(ds)
            geographic = _is_geographic(x, y, str(ds[_X].attrs.get("units", "")))
            if geographic:
                lon, lat = x, y
            elif "longitude" in ds and "latitude" in ds:
                lon = np.asarray(ds["longitude"].values, dtype=np.float64)
                lat = np.asarray(ds["latitude"].values, dtype=np.float64)
            else:
                raise ValueError(
                    f"SCHISM mesh {path} is projected (node x/y in metres) and "
                    "has no longitude/latitude variables: add them (e.g. with "
                    "pyproj) so the reader can place the nodes and rotate the "
                    "velocity to east/north")
            depth = (np.asarray(ds["depth"].values, dtype=np.float64)
                     if "depth" in ds else np.zeros_like(lon))

        validate_quad_convexity(conn, lon, lat, source=f"SCHISM mesh {path}")
        clon, clat = _elem_centres(conn, lon, lat)
        n2e, n2e_count = build_node_to_elems(lon.shape[0], conn)
        meta = {"path": str(path), "projected": not geographic,
                "n_triangles": int((conn[:, 3] < 0).sum()),
                "n_quads": int((conn[:, 3] >= 0).sum())}
        if not geographic:
            cos_n, sin_n = _grid_rotation(conn, x, y, lon, lat)
            meta["_rot_cos"] = cos_n
            meta["_rot_sin"] = sin_n
            print(f"    SCHISM: projected mesh; grid-to-true rotation "
                  f"{np.degrees(np.arctan2(sin_n, cos_n)).min():+.3f} to "
                  f"{np.degrees(np.arctan2(sin_n, cos_n)).max():+.3f} deg")
        mesh = Mesh(
            name="schism", nodes_lon=lon, nodes_lat=lat, depth=depth,
            elem_nodes=conn, elem_center_lon=clon, elem_center_lat=clat,
            node_to_elems=n2e, node_to_elems_count=n2e_count,
            sigma=np.array([0.0]), vertical_coord="sigma",
            metadata=meta,
        )
        build_search(mesh)
        return mesh

    @staticmethod
    def open_dataset(path, v_path=None, w_path=None, mode_cfg=None):
        if getattr(mode_cfg, "Fdrift3D", False):
            raise ValueError("SCHISM reader: 3-D tracking is not supported (2-D, depth-averaged only)")
        if getattr(mode_cfg, "sample_vars", None):
            raise ValueError("SCHISM reader: mode.sample_vars is not supported")
        if v_path or w_path:
            raise ValueError("SCHISM reader: one file holds both velocity components "
                             "(data.ncinput_v / ncinput_w are not used)")
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        ds = xr.open_dataset(p, engine=_detect_engine(p))
        u_name, v_name = _uv_names(ds, mode_cfg)
        keep = [u_name] + ([v_name] if v_name else [])
        # Only the velocity: prefetch windows load every time-dependent
        # variable of the dataset, and SCHISM files carry zcor/hvel beside it.
        out = ds[keep]
        out.attrs["schism_uv"] = ",".join(keep)
        return out

    @staticmethod
    def read_velocities(ds, timestep, mesh, mode_cfg=None, diffusion=None,
                        sample_vars=None, velocity_float32=False):
        vd = np.float32 if velocity_float32 else np.float64
        names = ds.attrs.get("schism_uv", "").split(",")
        if names == [""]:
            u_name, v_name = _uv_names(ds, mode_cfg)
            names = [u_name] + ([v_name] if v_name else [])
        if len(names) == 1:
            uv = _record(ds[names[0]], timestep)
            u, v = uv[:, 0], uv[:, 1]
        else:
            u = _record(ds[names[0]], timestep)
            v = _record(ds[names[1]], timestep)
        u = _finite(u)
        v = _finite(v)
        c = mesh.metadata.get("_rot_cos")
        if c is not None:
            s = mesh.metadata["_rot_sin"]
            u, v = c * u - s * v, s * u + c * v
        kh = _read_kh_fields(ds, timestep, mesh, diffusion)
        kh_z, kh_m, kh_layout = kh if kh is not None else (None, None, None)
        return VelocityField(
            u=u.astype(vd, copy=False), v=v.astype(vd, copy=False),
            eta=np.zeros(mesh.n_nodes, dtype=np.float64),
            kh_zonal=kh_z, kh_meridional=kh_m, kh_layout=kh_layout,
            uv_layout="node")

    @staticmethod
    def probe_uv_layout(ds, mode_cfg, mesh) -> str:
        return "node"
