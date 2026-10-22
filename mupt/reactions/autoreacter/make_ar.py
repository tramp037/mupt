'''Wrapper for invoking the REACTER reaction engine from a MuPT reactions config.'''
__author__ = 'Janitha Manhanthe'
__email__ = 'jmanhanth@stevens.edu'


import yaml
from pathlib import Path

def arx_run(inputs: str) -> dict:
    '''
    Launch the AutoREACTER engine with the given MuPT reactions config.

    Parameters
    ----------
    inputs : dict
        The reactions config as a dictionary.

    Returns
    -------
    dict
        The results from AutoREACTER.
    '''
    if isinstance(inputs, dict):
        recipe = inputs
        in_dir = Path.cwd()  # Default to current working directory if dict is passed
    else:
        abs_path = Path(inputs).resolve()
        recipe = yaml.safe_load(abs_path.read_text())
        in_dir = abs_path.parent
        
    # 2. Translate YAML dict to AutoREACTER dict format
    arx_dict = _build_arx_dict(recipe, in_dir)
    return arx_dict

def _build_arx_dict(recipe: dict, in_dir: Path) -> dict:
    # 1. Base details
    simulation_name = recipe["dirname"]
    
    # Extract force field from the reaction_engine block
    try:
        force_field = recipe["reaction_engine"]["inputs"]["force_field"]
    except KeyError:
        raise ValueError("Missing 'force_field' in recipe['reaction_engine']['inputs']")

    # 2. Process monomers
    monomers_out = []
    for entry in recipe.get("monomers", []):
        name = entry["name"]
        smiles = entry.get("smiles")
        if smiles is None:
            raise ValueError(
                f"Monomer {name!r} has no 'smiles' in the recipe and none was "
                "supplied in smiles_by_name."
            )
        monomers_out.append({"name": name, "smiles": smiles})

    # 3. Process simulations (Supports both Ratio Mode and Count Mode)
    simulations_out = []
    for entry in recipe.get("system_parameters", []):
        sim = {
            "tag": entry["tag"],
            "temperature": entry["temperature"],
            "density": entry["density"],
        }
        
        # Handle Ratio Mode
        if "monomer_ratios" in entry:
            if "total_atoms" not in entry:
                raise ValueError(f"Simulation '{entry['tag']}' uses 'monomer_ratios' but is missing 'total_atoms'.")
            sim["total_atoms"] = entry["total_atoms"]
            sim["monomer_ratios"] = entry["monomer_ratios"]
            
        # Handle Count Mode
        elif "monomer_counts" in entry:
            sim["monomer_counts"] = entry["monomer_counts"]
            
        else:
            raise ValueError(f"Simulation '{entry['tag']}' must define either 'monomer_ratios' or 'monomer_counts'.")

        simulations_out.append(sim)

    # 4. Construct final payload
    return {
        "output_dir": Path(in_dir),
        "simulation_name": simulation_name,
        "force_field": force_field,
        "simulations": simulations_out,
        "monomers": monomers_out
    }