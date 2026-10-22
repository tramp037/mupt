"""Tests for the MuPT -> AutoREACTER YAML translator.

Run from the repository root:
    python -m pytest tests/reactions/autoreacter/ -v

This module only tests translation. The current arx_run() function does not
launch AutoREACTER or LAMMPS.
"""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from mupt.reactions.autoreacter.make_ar import arx_run, _build_arx_dict


HERE = Path(__file__).resolve().parent
SAMPLE_YAML = HERE / "reacter_smoke.yaml"


@pytest.fixture
def recipe():
    """Fresh dictionary for each test."""
    return yaml.safe_load(SAMPLE_YAML.read_text(encoding="utf-8"))


@pytest.fixture
def yaml_path(tmp_path):
    """Place the YAML in a temporary directory to test output_dir."""
    path = tmp_path / "reactions.yaml"
    path.write_text(SAMPLE_YAML.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def test_example_yaml_is_valid_mapping(recipe):
    assert recipe["reaction_engine"]["name"] == "reacter"
    assert len(recipe["monomers"]) == 2
    assert len(recipe["system_parameters"]) == 2


def test_yaml_file_uses_parent_as_output_dir(yaml_path):
    output = arx_run(str(yaml_path))
    assert output["output_dir"] == yaml_path.parent.resolve()
    assert output["simulation_name"] == "TMC_MPD_MuPT_Test"
    assert output["force_field"] == "PCFF"


def test_path_object_is_accepted(yaml_path):
    output = arx_run(yaml_path)
    assert output["output_dir"] == yaml_path.parent.resolve()


def test_dictionary_uses_current_directory(recipe, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = arx_run(recipe)
    assert output["output_dir"] == Path.cwd()


def test_ratio_mode_mapping(recipe):
    output = _build_arx_dict(recipe, Path("test-output"))
    simulation = output["simulations"][0]
    assert simulation == {
        "tag": "ratio_10k_300K",
        "temperature": 300,
        "density": 0.8,
        "total_atoms": 10000,
        "monomer_ratios": {"tmc": 1.0, "mpd": 1.5},
    }
    assert "monomer_counts" not in simulation


def test_count_mode_mapping(recipe):
    output = _build_arx_dict(recipe, Path("test-output"))
    simulation = output["simulations"][1]
    assert simulation == {
        "tag": "counts_300K",
        "temperature": 300,
        "density": 0.8,
        "monomer_counts": {"tmc": 220, "mpd": 330},
    }
    assert "total_atoms" not in simulation
    assert "monomer_ratios" not in simulation


def test_monomer_names_and_smiles_preserved(recipe):
    output = arx_run(recipe)
    assert output["monomers"] == recipe["monomers"]


def test_missing_force_field_raises_value_error(recipe):
    del recipe["reaction_engine"]["inputs"]["force_field"]
    with pytest.raises(ValueError, match="Missing 'force_field'"):
        arx_run(recipe)


def test_missing_smiles_raises_value_error(recipe):
    del recipe["monomers"][0]["smiles"]
    with pytest.raises(ValueError, match="has no 'smiles'"):
        arx_run(recipe)


def test_missing_total_atoms_in_ratio_mode_raises(recipe):
    del recipe["system_parameters"][0]["total_atoms"]
    with pytest.raises(ValueError, match="missing 'total_atoms'"):
        arx_run(recipe)


def test_missing_composition_mode_raises(recipe):
    del recipe["system_parameters"][1]["monomer_counts"]
    with pytest.raises(ValueError, match="must define either"):
        arx_run(recipe)


def test_missing_dirname_currently_raises_keyerror(recipe):
    """Documents current behavior; can be updated when validation improves."""
    del recipe["dirname"]
    with pytest.raises(KeyError, match="dirname"):
        arx_run(recipe)


def test_both_composition_modes_currently_prioritizes_ratio(recipe):
    """Documents current behavior, not a recommendation for valid configs."""
    recipe = deepcopy(recipe)
    recipe["system_parameters"][0]["monomer_counts"] = {"tmc": 10, "mpd": 15}
    output = arx_run(recipe)
    assert "monomer_ratios" in output["simulations"][0]
    assert "monomer_counts" not in output["simulations"][0]


def test_reactions_are_not_yet_forwarded_to_autoreacter(recipe):
    """The current adapter translates species/simulations only."""
    output = arx_run(recipe)
    assert "reactions" not in output
