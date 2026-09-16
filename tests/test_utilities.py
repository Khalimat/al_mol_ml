import pytest

from jcim.utilities import require_file, str2bool


def test_str2bool_accepts_common_truthy_and_falsy_strings():
    assert str2bool("yes") is True
    assert str2bool("Y") is True
    assert str2bool("no") is False
    assert str2bool(False) is False


def test_str2bool_rejects_garbage():
    with pytest.raises(Exception):
        str2bool("maybe")


def test_require_file_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_file(tmp_path, "nope.csv")


def test_require_file_returns_path_for_present_file(tmp_path):
    target = tmp_path / "present.csv"
    target.write_text("a,b\n1,2\n")
    result = require_file(tmp_path, "present.csv")
    assert result == target
