import hashlib

from reference_notebook import NOTEBOOK, cells

SHA256 = "127b44d4d858efbc2d90d128815ce75a7cce2882702f7209134354a22e2b13c7"


def test_the_reference_notebook_is_the_upstream_copy():
    assert hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() == SHA256


def test_the_reference_notebook_is_1_20_8_with_47_cells():
    assert len(cells()) == 47
    assert "Notebook_version = '1.20.8'" in "".join(cells()[8]["source"])
