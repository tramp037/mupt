from mupt.reactions.polymerizeit import react_pi
from mupt.reactions.polymerizeit import write_pi

import yaml
import argparse

def make_pi(inputs_file=None):

    if inputs_file is None:
        parser = argparse.ArgumentParser(description='Generate input files for PolymerizeIt!')
        parser.add_argument('-i', '--inputs', type=str, help='Path to the input YAML file')
        args = parser.parse_args()
        inputs_file = args.inputs
    
    if inputs_file is None:
        print("No input file provided. Please provide a YAML file with the necessary inputs using the -i or --inputs flag.")
        return

    with open(inputs_file, 'r') as f:
        inputs = yaml.safe_load(f)

    for reaction in inputs['reactions']:

        # react the monomers to form the dimer and create primitives and RDKit molecules
        react_pi.react_molecules_to_product(inputs, reaction)

        # find atom indices corresponding to reactive sites in monomers and dimer
        react_pi.identify_reactive_sites(inputs, reaction)

        # match the atom indices with monomer A or B
        react_pi.map_product_atoms_to_reactants(inputs, reaction)


    return inputs
    

