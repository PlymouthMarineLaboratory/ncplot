"""Grids that cannot be drawn as they are, drawn on a regular longitude-latitude
grid instead: an unstructured grid (ICON's or FESOM's cells along one dimension,
with a longitude and latitude each) and a curvilinear one that a quadmesh pulls
out of shape (the seam of a global ocean grid, where the longitudes jump by 360
degrees, and the north fold of a tripolar one, where two rows of cells are in
the same places), as the native grids of the CMIP6 ocean models are.

(Ported from oceanval's viewer_grid.py.)

Each pixel of the image takes the value of the cell whose centre is nearest
(on the sphere, so the seam and the fold do not matter), if it is near enough
to be in that cell: within its radius, times RADIUS_FACTOR (half the diagonal
of a square cell is 0.71 of its side). A curvilinear grid's cell's radius is
the distance to the farthest of its neighbours along the rows and columns; an
unstructured grid's, the distance to its fourth nearest other cell. Cells with
no longitude or latitude, or piled up at one place (NEMO puts the cells of the
land processors it leaves out at one point; more than the two cells of a north
fold or a repeated column), are left out. A pixel whose cell has no value
takes the next nearest cell it is in that has one (OWNERS are looked at), on
a grid folded over itself only (has_folds: a ROMS grid's lines crossing),
which can put land cells over sea cells. Which cell each pixel takes is worked out once for a grid (Raster) and
kept (raster_for), so each map after the first costs one take. The cells'
corners (CMIP's vertices) are not needed.
"""

import hashlib
import warnings
from collections import OrderedDict

import numpy as np

# the longest side of the image, in pixels (a quarter of a degree, round the globe)
MAX_SIDE = 1440
# the pixels are about this fraction of the cells' narrowest side (the median of the nearest neighbours)
PIXEL_FACTOR = 0.5
# a pixel is in its nearest cell if within this times the distance from that cell to its 4th nearest
RADIUS_FACTOR = 0.75
# how many of a cell's nearest others are looked at for its radius (piled-up cells are skipped)
NEIGHBOURS = 8
# a pixel's cells looked at, nearest first: the first with a value that the pixel is in gives it its value
OWNERS = 16
# rasters kept, for the grids drawn last
KEPT = 4
_kept = OrderedDict()


def _xyz(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _wrap(lon):
    """lon in [-180, 180)."""
    return (np.asarray(lon, dtype="float64") + 180) % 360 - 180


def has_seam(lon):
    """Whether neighbouring cells of a 2-D longitude are more than 180 degrees
    apart (the seam of a global grid, or a regional one across the dateline)."""
    lon = _wrap(lon)
    if lon.ndim != 2:
        return False
    with np.errstate(invalid="ignore"):
        return bool(
            (np.abs(np.diff(lon, axis=0)) > 180).any() or (np.abs(np.diff(lon, axis=1)) > 180).any()
        )


def has_folds(lon, lat):
    """Whether some cells of a 2-D grid are turned over: the quadrilateral of
    four neighbouring centres goes round the other way from most (a ROMS
    grid's lines crossing over land), which a quadmesh draws as spikes
    across the map."""
    lon, lat = np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
    if lon.ndim != 2 or min(lon.shape) < 2:
        return False
    # longitudes as steps from a neighbour, so a seam is no fold
    along = _wrap(lon[:-1, 1:] - lon[:-1, :-1]), lat[:-1, 1:] - lat[:-1, :-1]
    across = _wrap(lon[1:, :-1] - lon[:-1, :-1]), lat[1:, :-1] - lat[:-1, :-1]
    with np.errstate(invalid="ignore"):
        turn = along[0] * across[1] - along[1] * across[0]
        turn = turn[np.isfinite(turn) & (turn != 0)]
    if not turn.size:
        return False
    return bool(min((turn > 0).sum(), (turn < 0).sum()) > 0)


def _grid_radius(xyz):
    """For each cell of a curvilinear grid (xyz: (ny, nx, 3), NaN where a cell
    is left out), the farthest and the nearest of its neighbours along the
    grid's rows and columns (chords on the unit sphere), 0 if it has none."""
    distances = []
    for axis in (0, 1):
        step = np.linalg.norm(np.diff(xyz, axis=axis), axis=-1)
        pad = [(0, 0), (0, 0)]
        pad[axis] = (1, 0)
        distances.append(np.pad(step, pad, constant_values=np.nan))
        pad[axis] = (0, 1)
        distances.append(np.pad(step, pad, constant_values=np.nan))
    distances = np.stack(distances)
    # a cell in the same place as its neighbour (a repeated column) is no measure of its size
    distances[distances <= 1e-7] = np.nan
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        radius = np.nanmax(distances, axis=0)
        nearest = np.nanmin(distances, axis=0)
    return np.nan_to_num(radius, nan=0.0), np.nan_to_num(nearest, nan=0.0)


def _cloud_radius(tree, points):
    """For each of the cells at points (an unstructured grid's, (n, 3) on the
    unit sphere, whose cKDTree is tree), the chord to its 4th nearest other
    cell (or the farthest of those looked at), and to its nearest."""
    k = min(NEIGHBOURS, len(points))
    distances, _ = tree.query(points, k=k, workers=-1)
    apart = np.where(distances > 1e-7, distances, np.nan)
    counted = np.sum(np.isfinite(apart), axis=1)
    apart.sort(axis=1)
    which = np.clip(np.minimum(counted, 4) - 1, 0, None)
    radius = np.where(counted > 0, apart[np.arange(len(apart)), which], 0.0)
    return radius, np.where(counted > 0, apart[:, 0], 0.0)


def _degrees(chord):
    return np.degrees(2 * np.arcsin(np.clip(chord / 2, 0, 1)))


class Raster:
    """For the cells at lon, lat (arrays of one shape: a curvilinear grid's 2-D
    ones, or an unstructured grid's 1-D ones), a regular longitude-latitude
    grid (x_axis, y_axis: the pixels' centres) over them, at most max_side
    pixels on its longest side, and for each pixel the cell it is in, if any."""

    def __init__(self, lon, lat, max_side=MAX_SIDE):
        from scipy.spatial import cKDTree

        lon = np.asarray(lon, dtype="float64")
        lat = np.asarray(lat, dtype="float64")
        self.cells = lon.size
        self.grid_shape = lon.shape
        usable = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90)
        # cells piled up in one place (more than the two of a fold or a grid's repeated columns): fill values
        rounded = np.where(usable, np.round(_wrap(lon), 5) + 1j * np.round(lat, 5), np.nan)
        _, where, counts = np.unique(rounded.ravel(), return_inverse=True, return_counts=True)
        piled = usable & (counts[where] > 2).reshape(lon.shape)
        # not shown, whatever their values: a model's filler (CESM's land processors at -1, -1 have tos -1)
        self.piled = piled.ravel()
        usable &= ~piled
        xyz = np.full(lon.shape + (3,), np.nan)
        xyz[usable] = _xyz(lon[usable], lat[usable])
        tree = None
        if lon.ndim == 2:
            radius, nearest = _grid_radius(xyz)
        else:
            radius, nearest = np.zeros(lon.shape), np.zeros(lon.shape)
            if usable.sum() >= 2:
                tree = cKDTree(xyz[usable])
                radius[usable], nearest[usable] = _cloud_radius(tree, xyz[usable])
        # the two cells in one place (across a fold) are as big as the bigger: the nearest of them is either
        group = where.reshape(lon.shape)[usable]
        largest = np.zeros(counts.size)
        np.maximum.at(largest, group, radius[usable])
        radius[usable] = largest[group]
        keep = (usable & (radius > 0)).ravel()
        if keep.sum() < 2:
            raise ValueError("The grid's longitudes and latitudes place too few cells.")
        placed = np.flatnonzero(keep)
        points, radius = xyz.reshape(-1, 3)[keep], radius.ravel()[keep]
        lon, lat = lon.ravel(), lat.ravel()
        self.kept = placed
        # the cells' size (the farthest neighbour), for the margin; their narrowest side (the nearest), for the
        # pixels, so a cell narrow one way (near a pole) still has a pixel of its own, MAX_SIDE allowing
        self.spacing = float(_degrees(np.median(radius)))
        narrowest = nearest.ravel()[keep]
        self.narrowest = float(_degrees(np.median(narrowest[narrowest > 0]))) if (narrowest > 0).any() else self.spacing
        radius = radius * RADIUS_FACTOR

        lons, lats = _wrap(lon[self.kept]), lat[self.kept]
        # longitudes from -180 or from 0, whichever covers the cells in less (a grid across the dateline),
        # -180 unless 0 saves more than a cell
        shifted = lons % 360
        if np.ptp(shifted) < np.ptp(lons) - self.spacing:
            lons = shifted
        pad = self.spacing / 2
        x0, x1 = float(lons.min()) - pad, float(lons.max()) + pad
        y0, y1 = max(float(lats.min()) - pad, -90.0), min(float(lats.max()) + pad, 90.0)
        if x1 - x0 > 360 - self.spacing:
            # round the globe
            x0, x1 = (-180.0, 180.0) if lons.min() < 0 else (0.0, 360.0)
        step = max(self.narrowest * PIXEL_FACTOR, max(x1 - x0, y1 - y0) / max_side)
        nx, ny = max(2, int(np.ceil((x1 - x0) / step))), max(2, int(np.ceil((y1 - y0) / step)))
        self.shape = (ny, nx)
        self.step = step
        self.x_axis = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        self.y_axis = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny

        if tree is None or tree.n != len(points):
            tree = cKDTree(points)
        px, py = np.meshgrid(self.x_axis, self.y_axis)
        # pixels farther than any cell's radius from every cell are in none: the search stops there (without the
        # bound, a pixel far from a dense mesh, on land, searches most of the tree)
        # only a grid folded over itself looks past a pixel's nearest cell: elsewhere a coast's sea would spread
        # into its land cells
        k = min(OWNERS if has_folds(lon.reshape(self.grid_shape), lat.reshape(self.grid_shape)) else 1, len(points))
        distance, nearest = tree.query(
            _xyz(px.ravel(), py.ravel()), k=k, workers=-1, distance_upper_bound=float(radius.max()) * (1 + 1e-9)
        )
        distance, nearest = distance.reshape(-1, k), nearest.reshape(-1, k)
        found = nearest < len(points)
        inside = np.zeros(found.shape, dtype=bool)
        inside[found] = distance[found] <= radius[nearest[found]]
        self.pixels = np.flatnonzero(inside.any(axis=1))
        # for each pixel, the cells it is in, nearest first (-1: no more); the first is its own cell
        owners = np.where(inside, self.kept[np.where(found, nearest, 0)], -1)[self.pixels]
        # each pixel's cells in order, those it is in first
        order = np.argsort(owners < 0, axis=1, kind="stable")
        owners = np.take_along_axis(owners, order, axis=1).astype("int32" if self.cells < 2**31 else "int64")
        # no more columns than the most cells a pixel is in
        self.candidates = owners[:, : max(1, int((owners >= 0).sum(axis=1).max()))]
        self.owners = self.candidates[:, 0]

    def image(self, values):
        """The values of the cells (of the grid's shape, or flattened) as a
        (ny, nx) float64 array: NaN where no cell is, or where its value is
        missing. A pixel whose cell's value is missing takes that of the next
        nearest cell it is in that has one (a grid folded over itself puts
        land cells over sea cells, which they would hide)."""
        values = np.asarray(values, dtype="float64").ravel()
        if values.size != self.cells:
            raise ValueError("The values are not of the grid they are drawn on.")
        taken = values[self.owners]
        for column in range(1, self.candidates.shape[1]):
            missing = ~np.isfinite(taken)
            if not missing.any():
                break
            cells = self.candidates[missing, column]
            taken[missing] = np.where(cells >= 0, values[np.maximum(cells, 0)], np.nan)
        out = np.full(self.shape[0] * self.shape[1], np.nan)
        out[self.pixels] = taken
        return out.reshape(self.shape)

    def placed_share(self, values):
        """The share of the cells with values that the image shows (some, at
        least, of its pixels), cells piled up in one place left out."""
        values = np.asarray(values, dtype="float64").ravel()
        valued = np.isfinite(values) & ~self.piled
        if not valued.any():
            return 1.0
        shown = np.zeros(values.size, dtype=bool)
        shown[self.kept] = True
        return float((shown & valued).sum() / valued.sum())


def raster_for(lon, lat):
    """The Raster of the grid at lon, lat, made the first time and kept."""
    lon = np.ascontiguousarray(lon, dtype="float64")
    lat = np.ascontiguousarray(lat, dtype="float64")
    key = (lon.shape, hashlib.sha1(lon.tobytes()).hexdigest(), hashlib.sha1(lat.tobytes()).hexdigest())
    if key in _kept:
        _kept.move_to_end(key)
        return _kept[key]
    raster = Raster(lon, lat)
    _kept[key] = raster
    while len(_kept) > KEPT:
        _kept.popitem(last=False)
    return raster
