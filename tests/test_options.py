import pytest

from transformercvn.options import Options


@pytest.mark.parametrize("value, expected", [(True, True), (False, False), (1, True), (0, False)])
def test_boolean_overrides_keep_boolean_type(value, expected):
    options = Options()
    options.update_options({"verbose_output": value, "batch_size": 12.0})
    assert options.verbose_output is expected
    assert options.batch_size == 12
    assert type(options.batch_size) is int


@pytest.mark.parametrize("value", ["false", "true", 2, None])
def test_invalid_boolean_overrides(value):
    with pytest.raises(ValueError, match="verbose_output"):
        Options().update_options({"verbose_output": value})


def test_load_preserves_extension_keys(tmp_path):
    config = tmp_path / "options.json"
    config.write_text('{"verbose_output": false, "experiment_label": "sample"}')
    options = Options.load(str(config))
    assert options.verbose_output is False
    assert options.experiment_label == "sample"
