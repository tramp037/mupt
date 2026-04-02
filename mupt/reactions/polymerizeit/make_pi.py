from mupt.reactions.polymerizeit import react_pi
from mupt.reactions.polymerizeit import write_pi

import yaml
import argparse

def make_pi():

    parser = argparse.ArgumentParser(description='Generate input files for PolymerizeIt!')
    parser.add_argument('-i', '--inputs', type=str, help='Path to the input YAML file')
    args = parser.parse_args()
    inputs_file = args.inputs
    
    if inputs_file is None:
        print("No input file provided. Please provide a YAML file with the necessary inputs using the -i or --inputs flag.")
        return

    with open(inputs_file, 'r') as f:
        inputs = yaml.safe_load(f)


    # React the monomers to form the dimer and create primitives and RDKit molecules
    inputs['mon_A_prim'], inputs['mon_A_mol'], inputs['mon_B_prim'], inputs['mon_B_mol'], inputs['dim_prim'], inputs['dim_mol'], inputs['dim_smi'] = react_pi.react_monomers_to_dimer(inputs['mon_A_smi'], inputs['mon_B_smi'], inputs['react_template_A'], inputs['react_template_B'], inputs['prod_template'])
    
    # find atom indices corresponding to reactive sites in monomers and dimer
    inputs['mon_A_react'], inputs['mon_B_react'], inputs['mon_A_prod'], inputs['mon_B_prod'] = react_pi.identify_reactive_sites(inputs['mon_A_mol'], inputs['mon_B_mol'], inputs['dim_mol'], 
                                                                                                                                inputs['react_template_A'], inputs['react_idx_A'], 
                                                                                                                                inputs['react_template_B'], inputs['react_idx_B'], 
                                                                                                                                inputs['prod_template'], inputs['prod_idx_A'], inputs['prod_idx_B'])
    
    # match the atom indices with monomer A or B
    inputs['mon_A_indices'], inputs['mon_B_indices'] = react_pi.map_dimer_atoms_to_monomers(inputs['dim_prim'], inputs['mon_A_prod'], inputs['mon_B_prod'])

    print(inputs)
    

