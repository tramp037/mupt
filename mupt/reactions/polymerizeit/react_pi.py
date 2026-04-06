from mupt.interfaces.rdkit import primitive_from_rdkit, primitive_to_rdkit
from mupt.interfaces.smiles import primitive_from_smiles


from mupt.geometry.shapes import PointCloud
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdChemReactions

import networkx as nx
import numpy as np

import shutil
import periodictable

def react_monomers_to_dimer(mon_A_smi, mon_B_smi, react_template_A, react_template_B, prod_template):
    mon_A_prim = primitive_from_smiles(mon_A_smi, ensure_explicit_Hs=True, embed_positions=True)
    mon_A = primitive_to_rdkit(mon_A_prim)

    mon_B_prim = primitive_from_smiles(mon_B_smi, ensure_explicit_Hs=True, embed_positions=True)
    mon_B = primitive_to_rdkit(mon_B_prim)

    # note: currently assumes all dimers are identical and takes the first product only
    rxn = rdChemReactions.ReactionFromSmarts(f'{react_template_A}.{react_template_B}>>{prod_template}')
    reacts = (Chem.MolFromSmiles(Chem.MolToSmiles(mon_A)),Chem.MolFromSmiles(Chem.MolToSmiles(mon_B)))
    products = rxn.RunReactants(reacts)

    dim_smi = Chem.MolToSmiles(products[0][0])

    # Create the primitive and RDKit molecule for the dimer
    dim_prim = primitive_from_smiles(dim_smi, ensure_explicit_Hs=True, embed_positions=True)
    dim = primitive_to_rdkit(dim_prim)

    return mon_A_prim, mon_A, mon_B_prim, mon_B, dim_prim, dim, dim_smi

def identify_reactive_sites(mon_A, mon_B, dim, react_template_A, react_idx_A, react_template_B, react_idx_B, prod_template, prod_idx_A, prod_idx_B, mon_names=['Monomer A', 'Monomer B']):
    matches = mon_A.GetSubstructMatches(Chem.MolFromSmarts(react_template_A))
    all_matches = []

    for match in matches:
        all_matches.append(match[react_idx_A])
    print("Matches for primary amine nitrogens in " + mon_names[0] + ":", all_matches)

    mon_A_react = all_matches[0]+1  # first match corresponds to monomer A


    matches = mon_B.GetSubstructMatches(Chem.MolFromSmarts(react_template_B))
    all_matches = []

    for i, match in enumerate(matches):
        all_matches.append(match[react_idx_B])
    print("Matches for acid chloride carbons in " + mon_names[1] + ":", all_matches)
    mon_B_react = all_matches[0]+1  # first match corresponds to monomer B

    # find N bonded to C=O in dimer
    matches = dim.GetSubstructMatches(Chem.MolFromSmarts(prod_template))
    all_matches_A = []
    all_matches_B = []

    for i, match in enumerate(matches):
        all_matches_A.append(match[prod_idx_A])
        all_matches_B.append(match[prod_idx_B])

    print("Matches for amide nitrogens in DIM:", all_matches_A)
    print("Matches for carbonyl carbons in DIM:", all_matches_B)
    mon_A_prod = all_matches_A[0]+1  # first match corresponds to monomer A
    mon_B_prod = all_matches_B[0]+1  # first match corresponds to monomer B

    return mon_A_react, mon_B_react, mon_A_prod, mon_B_prod

def map_dimer_atoms_to_monomers(dim_prim, mon_A_prod, mon_B_prod):
    mon_A_indices = []
    mon_B_indices = []

    nodelist = list(dim_prim.topology.nodes)
    for atom in nodelist:
        path_A = nx.shortest_path(dim_prim.topology, source=atom, target=nodelist[mon_A_prod-1])
        path_B = nx.shortest_path(dim_prim.topology, source=atom, target=nodelist[mon_B_prod-1])
        if len(path_A) < len(path_B):
            mon_A_indices.append(atom[1]+1)
        else:
            mon_B_indices.append(atom[1]+1) 

    print("Monomer A atom indices in dimer:", mon_A_indices)
    print("Monomer B atom indices in dimer:", mon_B_indices)

    return mon_A_indices, mon_B_indices

def openff_atom_typing(mon_names, mon_A_smi, mon_B_smi, dim_smi):
    from openff.toolkit import ForceField, Molecule, Topology
    from openff.units import unit
    mon_A_mol = Molecule.from_smiles(mon_A_smi)
    mon_B_mol = Molecule.from_smiles(mon_B_smi)
    dim_mol = Molecule.from_smiles(dim_smi)

    for mol in [mon_A_mol, mon_B_mol, dim_mol]:
        mol.generate_conformers(n_conformers=1)

    mon_A_top = mon_A_mol.to_topology()
    mon_B_top = mon_B_mol.to_topology()
    dim_top = dim_mol.to_topology()

    for top in [mon_A_top, mon_B_top, dim_top]:
        top.box_vectors = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]) * unit.nanometer

    print("Performing atom typing. This may take a few minutes...")
    # # load the forcefield .xml file
    forcefield = ForceField("openff-2.2.0.offxml")
    # # create an openmm system first with the atom types assigned
    # # this will take a few minutes
    mon_A_system = forcefield.create_openmm_system(mon_A_top)
    mon_B_system = forcefield.create_openmm_system(mon_B_top)

    # # do the same for the dimer
    # # will also take a few minutes
    dim_system = forcefield.create_openmm_system(dim_top)

    # # finally create the interchangers and write to GROMACS files
    mon_A_int = forcefield.create_interchange(mon_A_top)
    mon_A_int.to_gromacs(prefix=mon_names[0].lower(), _merge_atom_types=True)
    mon_B_int = forcefield.create_interchange(mon_B_top)
    mon_B_int.to_gromacs(prefix=mon_names[1].lower(), _merge_atom_types=True)
    dim_int = forcefield.create_interchange(dim_top)
    dim_int.to_gromacs(prefix="dim", _merge_atom_types=True)
    print("Atom typing complete!")

def combine_atom_types(mon_names):
    shutil.copyfile(f'{mon_names[0].lower()}.top', f'{mon_names[0].lower()}_temp.top')
    shutil.copyfile(f'{mon_names[1].lower()}.top', f'{mon_names[1].lower()}_temp.top')
    shutil.copyfile('dim.top', 'dim_temp.top')

    # collect atom types and create mapping
    atom_types_dict = {}
    atom_types_map = {}
    atom_count = 1

    with open(f'{mon_names[0].lower()}_temp.top', 'r') as file:
        mon_A_top_content = file.read()
    in_atoms = False
    atom_types_map['mon_A'] = {}
    for line in mon_A_top_content.splitlines():
        if line.strip() == '[ atomtypes ]':
            in_atoms = True
        elif line.strip().startswith('[') and line.strip() != '[ atomtypes ]':
            in_atoms = False
        elif in_atoms and len(line.strip().split()) > 0 and not line.strip().startswith(';'):
            parts = line.split()
            if parts[1:] not in atom_types_dict.values():
                element = periodictable.elements[int(parts[1])].symbol
                atom_types_dict[f'{element}{atom_count}'] = parts[1:]
                atom_count += 1
            for key, value in atom_types_dict.items():
                if parts[1:] == value:
                    atom_types_map['mon_A'][parts[0]] = key
                    break

    with open(f'{mon_names[1].lower()}_temp.top', 'r') as file:
        mon_B_top_content = file.read()
    in_atoms = False
    atom_types_map['mon_B'] = {}
    for line in mon_B_top_content.splitlines():
        if line.strip() == '[ atomtypes ]':
            in_atoms = True
        elif line.strip().startswith('[') and line.strip() != '[ atomtypes ]':
            in_atoms = False
        elif in_atoms and len(line.strip().split()) > 0 and not line.strip().startswith(';'):
            parts = line.split()
            if parts[1:] not in atom_types_dict.values():
                element = periodictable.elements[int(parts[1])].symbol
                atom_types_dict[f'{element}{atom_count}'] = parts[1:]
                atom_count += 1
            for key, value in atom_types_dict.items():
                if parts[1:] == value:
                    atom_types_map['mon_B'][parts[0]] = key
                    break

    with open('dim_temp.top', 'r') as file:
        dim_top_content = file.read()
    in_atoms = False
    atom_types_map['dim'] = {}
    for line in dim_top_content.splitlines():
        if line.strip() == '[ atomtypes ]':
            in_atoms = True
        elif line.strip().startswith('[') and line.strip() != '[ atomtypes ]':
            in_atoms = False
        elif in_atoms and len(line.strip().split()) > 0 and not line.strip().startswith(';'):
            parts = line.split()
            if parts[1:] not in atom_types_dict.values():
                element = periodictable.elements[int(parts[1])].symbol
                atom_types_dict[f'{element}{atom_count}'] = parts[1:]
                atom_count += 1
            for key, value in atom_types_dict.items():
                if parts[1:] == value:
                    atom_types_map['dim'][parts[0]] = key
                    break

    # replace atom types in the top files with the mapped ones
    for original_type, mapped_type in atom_types_map['mon_A'].items():
        mon_A_top_content = mon_A_top_content.replace(original_type, mapped_type)
    with open(f'{mon_names[0].lower()}.top', 'w') as file:
        file.write(mon_A_top_content)
    for original_type, mapped_type in atom_types_map['mon_B'].items():
        mon_B_top_content = mon_B_top_content.replace(original_type, mapped_type)
    with open(f'{mon_names[1].lower()}.top', 'w') as file:
        file.write(mon_B_top_content)
    for original_type, mapped_type in atom_types_map['dim'].items():
        dim_top_content = dim_top_content.replace(original_type, mapped_type)
    with open('dim.top', 'w') as file:
        file.write(dim_top_content)
    
    return atom_types_dict

def write_itp(mon_names, atom_types_dict):
    with open(f'{mon_names[0].lower()}.top', 'r') as file:
        mpd_top_content = file.read()
    with open(f'{mon_names[1].lower()}.top', 'r') as file:
        tmc_top_content = file.read()
    with open('dim.top', 'r') as file:
        dim_top_content = file.read()

    # write forcefield.itp
    with open('forcefield.itp', 'w') as f:
        f.write('; Forcefield parameters from openff-2.2.0.offxml\n\n')
        f.write('[ defaults ]\n')
        f.write('; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ\n')
        f.write('1        2          yes        0.5      0.8333\n\n')
        f.write('#include "nonbond.itp"\n')
        f.write('#include "bonded.itp"\n')

    # combine atom types into a single .itp file
    with open('nonbond.itp', 'w') as f:
        f.write('; Combined atom types\n\n')
        f.write('[ atomtypes ]\n')
        for key, value in atom_types_dict.items():
            f.write(f'{key:<6} {"   ".join(value)}\n')
        f.write('\n')

    with open('bonded.itp', 'w') as f:
        f.write('; Combined bonded parameters\n\n')

    # create molecule .itp files
    for name, top_content in zip([mon_names[0].lower(), mon_names[1].lower(), 'dim'], [mpd_top_content, tmc_top_content, dim_top_content]):
        with open(f'{name}.itp', 'w') as f:
            f.write(f'; {name.upper()} molecule parameters from {name}.top\n\n')
            in_moleculetype = False
            for line in top_content.splitlines():
                if line.strip().startswith('[ moleculetype ]'):
                    in_moleculetype = True
                if line.strip().startswith('[ system ]'):
                    in_moleculetype = False
                if in_moleculetype:
                    f.write(line + '\n')
                if in_moleculetype and line.strip() == '':
                    f.write('\n')