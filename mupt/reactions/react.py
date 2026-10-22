'''Engine-agnostic entry point for reaction workflows described by a MuPT reactions config'''

__author__ = 'Salman Bin Kashif, Janitha Manhanthe'
__email__ = 'salmanbinkashif@gmail.com, jmanhanth@stevens.edu'

import yaml


# All suported reaction engines are listed here, so that run_reactions can dispatch to the right one.
# SUPPORTED_ENGINES = MuPT can generate run-ready project directories for the chosen engine that users will run on their own.
# PLANNED_ENGINES = Work is underway to support these engines, but MuPT does not yet generate run-ready project directories for them.
SUPPORTED_ENGINES = ('polymerizeit', "reacter")
REQUIRED_REACTION_KEYS = {
    'reactants':      [True, True],
    'react_template': [True, False],
    'react_idx':      [True, False],
    'products':       [True, False],
    'prod_template':  [True, False],
    'prod_idx':       [True, False]
}
REQUIRED_MONOMER_KEYS = ('name', 'smiles')


def run_reactions(inputs):
    '''
    Run the reaction workflow named by a MuPT reactions config.

    The config is common to every reaction engine; `reaction_engine['name']` selects which one runs.

    Parameters
    ----------
    inputs : dict or str or Path
        The reactions config as a dict, or a path to the YAML file holding it.

    Returns
    -------
    dict
        The config dict, enriched in place by the engine's workflow.

    Raises
    ------
    ValueError
        If the config is malformed, names no engine, or names an unrecognized one.
    NotImplementedError
        If the config selects a known engine that MuPT does not yet drive.
    '''
    config = inputs
    if not isinstance(inputs, dict):
        with open(inputs, 'r') as f:
            config = yaml.safe_load(f)

    engine = config['reaction_engine']['name']
    engine = str(engine).strip().lower()

    # engine = inputs['reaction_engine']['name']
    # engine = str(engine).strip().lower()

    if engine not in SUPPORTED_ENGINES:
        raise ValueError(
            f'Unrecognized reaction engine {engine!r}. '
            f'Expected one of {sorted(SUPPORTED_ENGINES)}.'
        )
    validate_config(config, engine=engine)
    # Engine names are compared case-insensitively so that the shared schema accepts both the
    # lower-case form used by MuPT's examples and the upper-case form used in REACTER's.
    
    if engine == 'polymerizeit':
        # Imported here rather than at module scope so that dispatching to another engine, or
        # rejecting an unknown one, does not pay for PolymerizeIt!'s RDKit-backed import chain.
        

        from mupt.reactions.polymerizeit.make_pi import make_pi

        return make_pi(inputs)

    if engine == 'reacter':
        from mupt.reactions.autoreacter.make_ar import arx_run
        # Pass the ORIGINAL inputs (which is the file path string) 
        # so arx_run can extract the abs_path.parent properly.
        return arx_run(inputs)

    known = sorted(SUPPORTED_ENGINES)
    raise ValueError(
        f'Unrecognized reaction engine {engine!r}. '
        f'Expected one of {known}.'
    )


# Keys every reactions config carries, whichever engine runs it. The per-reaction keys are the reaction
# template: which molecules react, the SMARTS patterns matching their reactive groups, and which atom of
# each pattern reacts. The *molecule* atom indices are derived from these, not supplied.

def validate_config(inputs, engine):
    '''Check the engine-agnostic structure of a reactions config.

    Reports every problem at once rather than the first, and runs before dispatch so a malformed config
    fails immediately instead of part-way through a reaction workflow. Engine-specific requirements are
    checked by the engine itself.

    Parameters
    ----------
    inputs : dict
        The reactions config.

    Raises
    ------
    ValueError
        If any required key is missing or malformed, listing all of them.
    '''
    problems = []

    engine_idx = {
        'polymerizeit': 0,
        'reacter': 1
    }[engine]

    monomers = inputs.get('monomers')
    if not isinstance(monomers, list) or not monomers:
        problems.append("'monomers' must be a non-empty list of the species the reaction starts from")
    else:
        for i, monomer in enumerate(monomers):
            if not isinstance(monomer, dict):
                problems.append(f'monomers[{i}] must be a mapping')
                continue
            for key in REQUIRED_MONOMER_KEYS:
                if key not in monomer:
                    problems.append(f"monomers[{i}] is missing '{key}'")

    reactions = inputs.get('reactions')
    if not isinstance(reactions, dict) or not reactions:
        problems.append("'reactions' must be a non-empty mapping of reaction name to reaction template")
    else:
        for name, reaction in reactions.items():
            if not isinstance(reaction, dict):
                problems.append(f'reactions.{name} must be a mapping')
                continue

            for key, required in REQUIRED_REACTION_KEYS.items():
                if required[engine_idx] and key not in reaction:
                    problems.append(f"reactions.{name} is missing '{key}'")

    engine_config = inputs.get('reaction_engine')
    if not isinstance(engine_config, dict):
        problems.append("'reaction_engine' must be a mapping naming the engine and its inputs")
    else:
        if 'name' not in engine_config:
            problems.append("reaction_engine is missing 'name'")
        if not isinstance(engine_config.get('inputs'), dict):
            problems.append("reaction_engine is missing an 'inputs' mapping")

    if problems:
        raise ValueError('Invalid reactions config:\n  - ' + '\n  - '.join(problems))
