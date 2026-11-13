import sys

import pytest

import ncplot.command_line as command_line

from conftest import CMIP6_FILES


def run(monkeypatch, *args):
    calls = []
    monkeypatch.setattr(command_line, "view", lambda x: calls.append(x))
    monkeypatch.setattr(sys, "argv", ["ncplot", *args])
    return command_line.main(), calls


def test_no_file(monkeypatch):
    assert run(monkeypatch)[0] == "Please supply a file"


def test_a_file_that_is_not_there(monkeypatch):
    message, calls = run(monkeypatch, "nowhere.nc")
    assert message == "nowhere.nc does not exist!" and calls == []


def test_one_file_is_viewed_as_it_is(monkeypatch):
    message, calls = run(monkeypatch, CMIP6_FILES[0])
    assert message is None and calls == [CMIP6_FILES[0]]


def test_several_files_are_viewed_together(monkeypatch):
    _, calls = run(monkeypatch, *CMIP6_FILES[:2])
    assert calls == [CMIP6_FILES[:2]]


def test_a_pattern_the_shell_left_alone(monkeypatch):
    _, calls = run(monkeypatch, CMIP6_FILES[0].replace("cmip6_ij", "cmip6_i?"))
    assert calls == [CMIP6_FILES[0]]


@pytest.mark.parametrize("url", ["https://example.org/data/file.nc", "http://localhost:8080/x.nc"])
def test_urls(monkeypatch, url):
    assert run(monkeypatch, url)[1] == [url]
