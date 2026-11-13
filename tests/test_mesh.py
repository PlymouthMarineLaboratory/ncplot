"""FVCOM's mesh, drawn on a regular grid."""

import numpy as np
import pytest

from ncplot.loading import open_one
from ncplot.mesh import Mesh, looks_like_fvcom, mesh_location

from conftest import CMIP6_FILES, FVCOM_FILE


@pytest.fixture
def two_triangles():
    # a unit square cut along its diagonal
    return Mesh([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [[0, 1, 2], [0, 2, 3]], cartesian=True)


def test_node_values_are_shaded_across_each_triangle(two_triangles):
    image = two_triangles.raster().image([0.0, 1.0, 2.0, 1.0], "node")
    x = two_triangles.raster().x_axis
    y = two_triangles.raster().y_axis
    # the nodes' values, 0, 1, 2 and 1, are x + y at the corners of the square
    expected = np.add.outer(y, x)
    inside = np.isfinite(image)
    assert inside.mean() > 0.95
    np.testing.assert_allclose(image[inside], expected[inside], atol=1e-6)


def test_element_values_fill_their_triangle(two_triangles):
    raster = two_triangles.raster()
    image = raster.image([10.0, 20.0], "nele")
    assert set(np.unique(image[np.isfinite(image)])) == {10.0, 20.0}
    # the first triangle (0, 1, 2) is the lower right of the square
    x, y = np.meshgrid(raster.x_axis, raster.y_axis)
    lower_right = (x - y) > 0.05
    assert np.all(image[lower_right & np.isfinite(image)] == 10.0)


def test_outside_the_mesh_is_missing():
    mesh = Mesh([0.0, 1.0, 0.0, 5.0, 6.0, 5.0], [0.0, 0.0, 1.0, 5.0, 5.0, 6.0], [[0, 1, 2], [3, 4, 5]], cartesian=True)
    image = mesh.raster().image(np.ones(6), "node")
    assert np.isnan(image[image.shape[0] // 2, image.shape[1] // 2])
    assert np.isfinite(image).any()


def test_a_missing_node_makes_its_triangles_missing(two_triangles):
    full = two_triangles.raster().image([0.0, 1.0, 1.0, 1.0], "node")
    image = two_triangles.raster().image([np.nan, 1.0, 1.0, 1.0], "node")
    # node 0 is in both triangles
    assert np.isfinite(full).sum() > 0 and np.isfinite(image).sum() == 0


def test_the_mesh_of_the_test_file():
    ds = open_one(FVCOM_FILE)
    assert looks_like_fvcom(ds)
    mesh = Mesh.from_dataset(ds)
    assert (mesh.nodes, mesh.elements) == (1469, 2654)
    assert mesh.triangles.min() >= 0 and mesh.triangles.max() < mesh.nodes
    assert not mesh.cartesian
    assert -5.01 < mesh.x.min() and mesh.x.max() < -3.0 and 50.0 <= mesh.y.min() and mesh.y.max() <= 51.0
    # the same from the path
    assert Mesh.read(FVCOM_FILE).nodes == mesh.nodes


def test_nodes_and_elements_of_the_test_file_are_drawn():
    ds = open_one(FVCOM_FILE)
    mesh = Mesh.from_dataset(ds)
    raster = mesh.raster()
    node = raster.image(ds["temp"].isel(time=0, siglay=0).values, "node")
    element = raster.image(ds["u"].isel(time=0, siglay=0).values, "nele")
    assert np.isfinite(node).sum() == np.isfinite(element).sum() > 0
    assert np.nanmin(node) >= float(ds["temp"].min()) - 1e-6
    assert mesh_location(ds["temp"].dims) == "node" and mesh_location(ds["u"].dims) == "nele"


def test_other_data_is_not_fvcom():
    assert not looks_like_fvcom(open_one(CMIP6_FILES[0]))
