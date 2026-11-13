"""FVCOM output: an unstructured mesh of triangles, with values on its nodes
(node) or on its elements (nele), the triangles' nodes in nv (three, nele,
1-based), and sigma layers (siglay, siglev) as the vertical.

Which output is FVCOM's is told by the mesh's dimensions and nv.

(Ported from oceanval's viewer_mesh.py.)

The plotting uses pyfvcom2's approach
(https://github.com/PlymouthMarineLaboratory/pyfvcom2, plotting/plot.py,
FVCOMPlotter): the triangles are nv transposed, less 1 (its
Triangulation(x, y, nv.T - 1)), on the nodes' longitudes and latitudes, and
a field is coloured over them as its tripcolor colours it: a value on the
nodes shaded linearly across each triangle, a value on the elements filling
its triangle. As Bokeh cannot draw a mesh of many triangles quickly, that is
done here onto a regular grid (Raster) and drawn as an image, pixels outside
the mesh missing. Which triangle each pixel
is in, and its weights, are worked out once for a mesh, so each map after
the first costs one weighted sum. Only numpy is needed.
"""

import numpy as np

# the variables that describe the mesh rather than the ocean, which are not mapped
MESH_VARIABLES = {
    "nv", "nbe", "nbve", "nbsn", "ntve", "ntsn", "partition", "lon", "lat", "lonc", "latc",
    "x", "y", "xc", "yc", "siglay", "siglev", "siglay_center", "siglev_center", "Itime", "Itime2",
    "Times", "time", "iint", "a1u", "a2u", "aw0", "awx", "awy", "art1", "art2",
}

# the longest side of a map's image, in pixels
MAX_SIDE = 1000
# triangle-pixel pairs looked at at once, as the raster is made
PAIRS_AT_ONCE = 4_000_000


def looks_like_fvcom(ds):
    """Whether an open dataset (xarray, netCDF4 or anything with ``.dims`` or
    ``.dimensions``, and ``.variables``) is raw FVCOM output: the unstructured
    mesh, which output that has been regridded no longer has."""
    dims = ds.dimensions if hasattr(ds, "dimensions") else ds.dims
    return {"node", "nele"} <= set(dims) and "nv" in ds.variables


def mesh_location(dims):
    """"node" or "nele", whichever of the mesh's dimensions dims has, or None."""
    return "node" if "node" in dims else "nele" if "nele" in dims else None


def _array(ds, name):
    """The variable as a float64 numpy array, NaN where it is missing."""
    return np.asarray(ds.variables[name].values, dtype="float64")


class Mesh:
    """The mesh of an FVCOM file: where its nodes are (x, y: longitude and
    latitude, or metres when the file has no longitudes), and its triangles
    (nele, 3), 0-based."""

    def __init__(self, x, y, triangles, cartesian=False):
        self.x = np.asarray(x, dtype="float64")
        self.y = np.asarray(y, dtype="float64")
        self.triangles = np.asarray(triangles, dtype="int64")
        self.cartesian = cartesian
        self._rasters = {}

    @classmethod
    def from_dataset(cls, ds):
        """The mesh of an open xarray Dataset of FVCOM output."""
        if "nv" not in ds.variables:
            raise ValueError("The dataset has no nv, so it has no FVCOM mesh.")
        nv = np.asarray(ds.variables["nv"].values).astype("int64")
        lon = _array(ds, "lon") if "lon" in ds.variables else None
        lat = _array(ds, "lat") if "lat" in ds.variables else None
        cartesian = lon is None or lat is None or not (np.nanmax(np.abs(lon)) > 0 or np.nanmax(np.abs(lat)) > 0)
        if cartesian:
            if "x" not in ds.variables or "y" not in ds.variables:
                raise ValueError("The FVCOM mesh has neither longitudes and latitudes nor x and y.")
            x, y = _array(ds, "x"), _array(ds, "y")
        else:
            x, y = np.where(lon > 180, lon - 360, lon), lat
        # nv is (three, nele) and counts from 1
        triangles = (nv.T if nv.shape[0] == 3 else nv) - 1
        return cls(x, y, triangles, cartesian)

    @classmethod
    def read(cls, path):
        """The mesh of the FVCOM file at path."""
        import xarray as xr

        # only the mesh is read: the file's times need not be decoded
        with xr.open_dataset(path, decode_times=False, decode_timedelta=False) as ds:
            return cls.from_dataset(ds[[name for name in ("nv", "lon", "lat", "x", "y") if name in ds.variables]])

    @property
    def nodes(self):
        return len(self.x)

    @property
    def elements(self):
        return len(self.triangles)

    def raster(self, max_side=MAX_SIDE):
        """The Raster of this mesh, at most ``max_side`` pixels on its longest
        side (made the first time it is asked for)."""
        if max_side not in self._rasters:
            self._rasters[max_side] = Raster(self, max_side)
        return self._rasters[max_side]


class Raster:
    """A regular grid over a mesh, at most MAX_SIDE pixels on its longest
    side, and for each pixel inside the mesh the triangle it is in and the
    weights of the triangle's three nodes there (barycentric coordinates)."""

    def __init__(self, mesh, max_side=MAX_SIDE):
        self.mesh = mesh
        x, y, triangles = mesh.x, mesh.y, mesh.triangles
        used = np.unique(triangles)
        x0, x1 = float(np.nanmin(x[used])), float(np.nanmax(x[used]))
        y0, y1 = float(np.nanmin(y[used])), float(np.nanmax(y[used]))
        width, height = max(x1 - x0, 1e-12), max(y1 - y0, 1e-12)
        if not mesh.cartesian:
            # a degree of longitude is shorter than one of latitude
            width *= max(np.cos(np.radians((y0 + y1) / 2)), 0.05)
        if width >= height:
            nx, ny = max_side, max(2, round(max_side * height / width))
        else:
            nx, ny = max(2, round(max_side * width / height)), max_side
        self.shape = (ny, nx)
        dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
        # the centres of the pixels
        self.x_axis = x0 + (np.arange(nx) + 0.5) * dx
        self.y_axis = y0 + (np.arange(ny) + 0.5) * dy

        corners_x, corners_y = x[triangles], y[triangles]
        # the pixels whose centres are in each triangle's bounding box
        i_low = np.ceil((corners_x.min(axis=1) - x0) / dx - 0.5).astype("int64").clip(0, nx - 1)
        i_high = np.floor((corners_x.max(axis=1) - x0) / dx - 0.5).astype("int64").clip(-1, nx - 1)
        j_low = np.ceil((corners_y.min(axis=1) - y0) / dy - 0.5).astype("int64").clip(0, ny - 1)
        j_high = np.floor((corners_y.max(axis=1) - y0) / dy - 0.5).astype("int64").clip(-1, ny - 1)
        widths = np.maximum(i_high - i_low + 1, 0)
        counts = widths * np.maximum(j_high - j_low + 1, 0)
        counts[~np.isfinite(corners_x).all(axis=1) | ~np.isfinite(corners_y).all(axis=1)] = 0

        pixels, owners, weights = [], [], []
        start = 0
        ends = np.cumsum(counts)
        while start < len(triangles):
            # triangles enough for PAIRS_AT_ONCE pairs (at least one)
            done = ends[start - 1] if start else 0
            stop = max(start + 1, int(np.searchsorted(ends, done + PAIRS_AT_ONCE, side="right")))
            chunk = np.arange(start, min(stop, len(triangles)))
            start = chunk[-1] + 1
            chunk = chunk[counts[chunk] > 0]
            if not chunk.size:
                continue
            owner = np.repeat(chunk, counts[chunk])
            offsets = np.repeat(np.cumsum(counts[chunk]) - counts[chunk], counts[chunk])
            k = np.arange(owner.size) - offsets
            ii = i_low[owner] + k % widths[owner]
            jj = j_low[owner] + k // widths[owner]
            px, py = x0 + (ii + 0.5) * dx, y0 + (jj + 0.5) * dy
            ax, ay = corners_x[owner, 0], corners_y[owner, 0]
            bx, by = corners_x[owner, 1], corners_y[owner, 1]
            cx, cy = corners_x[owner, 2], corners_y[owner, 2]
            denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            with np.errstate(divide="ignore", invalid="ignore"):
                wa = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
                wb = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
            wc = 1 - wa - wb
            eps = -1e-9
            inside = (denominator != 0) & (wa >= eps) & (wb >= eps) & (wc >= eps)
            pixels.append((jj * nx + ii)[inside])
            owners.append(owner[inside])
            weights.append(np.stack([wa, wb, wc], axis=1)[inside].clip(0, 1))
        pixels = np.concatenate(pixels) if pixels else np.zeros(0, "int64")
        owners = np.concatenate(owners) if owners else np.zeros(0, "int64")
        weights = np.concatenate(weights) if weights else np.zeros((0, 3))
        # a pixel on an edge two triangles share goes to the first
        pixels, first = np.unique(pixels, return_index=True)
        self.pixels = pixels
        self.elements = owners[first]
        self.weights = weights[first]

    def image(self, values, location="node"):
        """The values of the mesh (one for each node, or for each element:
        location) as a (ny, nx) float64 array: NaN outside the mesh, and
        where any of a triangle's nodes is missing."""
        values = np.asarray(values, dtype="float64").ravel()
        out = np.full(self.shape[0] * self.shape[1], np.nan)
        if location == "nele":
            out[self.pixels] = values[self.elements]
        else:
            corners = values[self.mesh.triangles[self.elements]]
            out[self.pixels] = (corners * self.weights).sum(axis=1)
        return out.reshape(self.shape)
