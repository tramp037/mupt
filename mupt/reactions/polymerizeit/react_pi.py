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

def react_molecules_to_product(inputs, reaction, ex_h_template=False):

    react_mols = []
    reaction_info = inputs['reactions'][reaction]
    for reactant in reaction_info['reactants']:
        # verify that reactant molecules have SMILES strings defined
        if 'smi' not in inputs['molecules'][reactant]:
            print(f"Error: Reactant {reactant} does not have a SMILES string defined. Please add a 'smi' entry for this reactant in the input file.")
            return
        # generate the primitive and RDKit molecule for the reactant if not already done
        if 'prim' in inputs['molecules'][reactant]:
            print(f"Primitive already exists for reactant {reactant}. Skipping primitive generation.")
        else:
            primitive = primitive_from_smiles(inputs['molecules'][reactant]['smi'], ensure_explicit_Hs=True, embed_positions=True)
            inputs['molecules'][reactant]['prim'] = primitive
        if 'mol' in inputs['molecules'][reactant]:
            print(f"RDKit molecule already exists for reactant {reactant}. Skipping RDKit molecule generation.")
        else:
            mol = primitive_to_rdkit(inputs['molecules'][reactant]['prim'])
            inputs['molecules'][reactant]['mol'] = mol
        if not ex_h_template:
            react_mols.append(Chem.MolFromSmiles(Chem.MolToSmiles(inputs['molecules'][reactant]['mol'])))
        else:
            react_mols.append(inputs['molecules'][reactant]['mol'])
        
    # create the reaction SMARTS string from the templates in the input file
    reactants_template = ".".join(reaction_info['react_template'])
    products_template = ".".join(reaction_info['prod_template'])

    # react to form products
    rxn = rdChemReactions.ReactionFromSmarts(f'{reactants_template}>>{products_template}')
    products = rxn.RunReactants(tuple(react_mols))

    # note: currently assumes all products are identical and takes the first product only
    for i, product in enumerate(reaction_info['products']):
        if product not in inputs['molecules']:
            inputs['molecules'][product] = {}
        else:
            print(f"Warning: Product {product} already in molecules. Overwriting with new product information.")
        prod_smi = Chem.MolToSmiles(products[0][i])
        prod_prim = primitive_from_smiles(prod_smi, ensure_explicit_Hs=True, embed_positions=True)
        prod_mol = primitive_to_rdkit(prod_prim)
        inputs['molecules'][product]['smi'] = prod_smi
        inputs['molecules'][product]['prim'] = prod_prim
        inputs['molecules'][product]['mol'] = prod_mol
    return

def identify_reactive_sites(inputs, reaction): #mon_A, mon_B, dim, react_template_A, react_idx_A, react_template_B, react_idx_B, prod_template, prod_idx_A, prod_idx_B, mon_names=['Monomer A', 'Monomer B']):
    
    react_idxs = []
    reaction_info = inputs['reactions'][reaction]
    for i, reactant in enumerate(reaction_info['reactants']):
        if 'mol' not in inputs['molecules'][reactant]:
            print(f"Error: Reactant {reactant} does not have an RDKit molecule defined. Please ensure that the reactant has been processed to generate the RDKit molecule before identifying reactive sites.")
            return

        matches = inputs['molecules'][reactant]['mol'].GetSubstructMatches(Chem.MolFromSmarts(reaction_info['react_template'][i]))
        all_matches = []

        for match in matches:
            all_matches.append(match[reaction_info['react_idx'][i]])

        print("Reacting groups in " + reactant + ":", all_matches)
        react_idxs.append(all_matches[0]+1)

    # note: assumes that the first product is the "main" product and identifies reactive sites in that product only
    product = reaction_info['products'][0]
    if 'mol' not in inputs['molecules'][product]:
        print(f"Error: Product {product} does not have an RDKit molecule defined. Please ensure that the product has been processed to generate the RDKit molecule before identifying reactive sites.")
        return

    matches = inputs['molecules'][product]['mol'].GetSubstructMatches(Chem.MolFromSmarts(reaction_info['prod_template'][0]))
    all_matches = [[] for _ in range(len(reaction_info['prod_idx']))]

    for match in matches:
        for j, idx in enumerate(reaction_info['prod_idx']):
            all_matches[j].append(match[idx])

        print("Product groups in " + product + ":", all_matches)
        prod_idxs = [all_matches[j][0]+1 for j in range(len(all_matches))]

    inputs['reactions'][reaction]['reactant_indices'] = react_idxs
    inputs['reactions'][reaction]['product_indices'] = prod_idxs
    
    return

def map_product_atoms_to_reactants(inputs, reaction):

    reaction_info = inputs['reactions'][reaction]
    mapped_indices = [[] for _ in range(len(reaction_info['reactants']))]

    # note: assumes that the first product is the "main" product
    topology = inputs['molecules'][reaction_info['products'][0]]['prim'].topology
    nodelist = list(topology.nodes)
    for atom in nodelist:
        paths = []
        for i, reactant in enumerate(reaction_info['reactants']):
            path = nx.shortest_path(topology, source=atom, target=nodelist[reaction_info['product_indices'][i]-1])
            paths.append(path)
        
        mapped_indices[np.argmin([len(path) for path in paths])].append(atom[1]+1)

    for i, reactant in enumerate(reaction_info['reactants']):
        print(f"Reactant {reactant} atom indices in product:", mapped_indices[i])
    
    inputs['reactions'][reaction]['mapped_indices'] = mapped_indices

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