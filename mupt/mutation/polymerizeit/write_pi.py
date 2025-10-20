import networkx as nx

def write_pi_input_gro(monomer_list, repeat_unit_list, monomer_filenames, repeat_unit_filenames, reactions, reactive_groups, distance_cutoff, outname, **kwargs):
    print("Writing the input file for PolymerizeIt! with GRO files ...")
    with open(outname, 'w') as f:
        f.write("; Input file for PolymerizeIt!\n")
        f.write(f"polymer_name={kwargs.get('polymer_name', 'polymer')}\n")
        f.write(f"md_engine=gromacs\n\n")
        f.write("; Monomers and repeat units\n")
        for i, monomer in enumerate(monomer_list):
            f.write(f"mon_{['A','B'][i]}={monomer},{monomer_filenames[i]}.gro\n")
        for i, repeat_unit in enumerate(repeat_unit_list):
            f.write(f"repeat_unit_{i+1}={repeat_unit},{repeat_unit_filenames[i]}.gro\n")

        # generate graphs for monomers and repeat units
        graph_dict = {}
        for i, monomer in enumerate(monomer_list):
            graph_dict[monomer] = generate_graph_gromacs(monomer_filenames[i]+'.gro', monomer_filenames[i]+'.top')
        for i, repeat_unit in enumerate(repeat_unit_list):
            graph_dict[repeat_unit] = generate_graph_gromacs(repeat_unit_filenames[i]+'.gro', repeat_unit_filenames[i]+'.top')

        f.write("\n; Reference reactions\n")
        for r, reaction in enumerate(reactions):
            # identify reacting atoms
            reacting_atoms = []
            for i, mol in enumerate(reaction[:2]):
                indices = []
                for atom in graph_dict[mol].nodes(data=True):
                    if atom[1]['atom_type'].startswith(reactive_groups[r][i][0]):
                        neighbors = list((neighbor, graph_dict[mol].nodes[neighbor]['atom_type']) for neighbor in graph_dict[mol].neighbors(atom[0]))
                        print(f"Neighbors of atom {atom[0]} ({atom[1]['atom_type']}) in {mol}: {neighbors}")
                        neighbor_match = []
                        for rg in reactive_groups[r][i][1:]:
                            for neighbor in neighbors:
                                if neighbor[1].startswith(rg) and neighbor[0] not in neighbor_match:
                                    neighbor_match.append(neighbor[0])
                                    break
                        if len(neighbor_match) == len(reactive_groups[r][i]) - 1:
                            indices.append(atom[0])
                if not indices:
                    raise ValueError(f"No matching atom found in {mol} for reactive group {reactive_groups[r][i]}")
                reacting_atoms.append(indices[0])  # take the first match for simplicity
            print(reacting_atoms)

            # identify corresponding atoms in the product
            product_atoms = []
            mon_idx = [[],[]]
            for i, mol in enumerate(reaction[:2]):
                for atom in graph_dict[reaction[2]].nodes(data=True):
                    # need to find reactive_groups[r][0][0] and [r][1][0] that are bonded to each other, [r][0][1:] but not [r][2][:], and [r][1][1:] but not [r][3][:], respectively
                    if atom[1]['atom_type'].startswith(reactive_groups[r][i][0]):
                        neighbor_list = []
                        neighbor_list.append(reactive_groups[r][(i+1)%2][0])
                        for rg in reactive_groups[r][i][1:]:
                            neighbor_list.append(rg)
                        for rg in reactive_groups[r][i+2]:
                            if rg in neighbor_list:
                                neighbor_list.remove(rg)
                        neighbors = list((neighbor, graph_dict[reaction[2]].nodes[neighbor]['atom_type']) for neighbor in graph_dict[reaction[2]].neighbors(atom[0]))
                        print(f"Neighbors of atom {atom[0]} ({atom[1]['atom_type']}) in {reaction[2]}: {neighbors}")
                        neighbor_match = []
                        for rg in neighbor_list:
                            for neighbor in neighbors:
                                if neighbor[1].startswith(rg) and neighbor[0] not in neighbor_match:
                                    neighbor_match.append(neighbor[0])
                                    break
                        if len(neighbor_match) == len(neighbor_list):
                            mon_idx[i].append((atom[0], neighbor_match[0]))

            if len(mon_idx[0]) > len(mon_idx[1]):
                temp_mon_idx = []
                # too many matches for mon A
                for idx_pair in mon_idx[1]:
                    for idx_pair2 in mon_idx[0]:
                        if idx_pair[1] == idx_pair2[0]:
                            temp_mon_idx.append(idx_pair2)
                mon_idx[0] = temp_mon_idx

            elif len(mon_idx[1]) > len(mon_idx[0]):
                temp_mon_idx = []
                # too many matches for mon B
                for idx_pair in mon_idx[0]:
                    for idx_pair2 in mon_idx[1]:
                        if idx_pair[1] == idx_pair2[0]:
                            temp_mon_idx.append(idx_pair2)
                mon_idx[1] = temp_mon_idx

            if len(mon_idx[0]) != len(mon_idx[1]):
                print("Error in:", mon_idx)
                raise ValueError(f"Mismatch in number of matching atoms between monomers and product for reaction {reaction}")

            for idx_pair in range(len(mon_idx[0])):
                if mon_idx[0][idx_pair][1] != mon_idx[1][idx_pair][0] and mon_idx[0][idx_pair][0] == mon_idx[1][idx_pair][1]:
                    mon_idx[0][idx_pair] = (mon_idx[1][idx_pair][1], mon_idx[1][idx_pair][0])
                elif mon_idx[0][idx_pair][0] != mon_idx[1][idx_pair][1] and mon_idx[0][idx_pair][1] == mon_idx[1][idx_pair][0]:
                    mon_idx[1][idx_pair] = (mon_idx[0][idx_pair][1], mon_idx[0][idx_pair][0])
                elif mon_idx[0][idx_pair][1] != mon_idx[1][idx_pair][0] and mon_idx[0][idx_pair][0] != mon_idx[1][idx_pair][1]:
                    print("Error in:", mon_idx)
                    raise ValueError(f"Mismatch in atom connectivity between monomers and product for reaction {reaction}")

            product_atoms = [mon_idx[0][0][0], mon_idx[1][0][0]]
            print(product_atoms)

            f.write(f"reference_reaction_{r+1}={reaction[0]}({reacting_atoms[0]}) & {reaction[1]}({reacting_atoms[1]}) : {reaction[2]}({product_atoms[0]}) & {reaction[2]}({product_atoms[1]})\n")

            f.write(f"\n; Product indices\n")
            mon_A_indices = []
            mon_B_indices = []
            for atom in graph_dict[reaction[2]].nodes(data=True):
                pathA = nx.shortest_path(graph_dict[reaction[2]], source=atom[0], target=product_atoms[0])
                pathB = nx.shortest_path(graph_dict[reaction[2]], source=atom[0], target=product_atoms[1])
                if len(pathA) < len(pathB):
                    mon_A_indices.append(atom[0])
                elif len(pathB) < len(pathA):
                    mon_B_indices.append(atom[0])
                else:
                    print(f"Atom {atom[0]} is equidistant from both product atoms; assigning to neither.")
            mon_A_indices.sort()
            mon_B_indices.sort()
            print(f"Monomer A indices in product {reaction[2]}: {mon_A_indices}")
            print(f"Monomer B indices in product {reaction[2]}: {mon_B_indices}")

            f.write(f"atom_indices_DIM_A1 = {','.join(map(str, mon_A_indices))}\n")
            f.write(f"atom_indices_DIM_B1 = {','.join(map(str, mon_B_indices))}\n")
            f.write(f"DIM_total_atoms={len(graph_dict[reaction[2]].nodes())}\n")

        f.write(f"\n; Reaction criteria\n")
        criteria = kwargs.get('reaction_criteria', ['DistanceCutoff', 'ClosestPair'])
        f.write(f"reaction_criteria={','.join(criteria)}\n")
        post_reaction = kwargs.get('post_reaction_updates', ['RedistributeProductCharges'])
        f.write(f"post_reaction_updates={','.join(post_reaction)}\n")

        f.write("\n; Other parameters\n")
        f.write(f"distance_cutoff={distance_cutoff}\n")
        f.write(f"same_monomer_reaction=no\n")

def generate_graph_gromacs(gro_file, itp_file):
    molgraph = nx.Graph()

    status = 0
    with open(itp_file, 'r') as f:
        for line in f:
            if line.startswith('[ atoms ]'):
                status = 1
                next(f)  # skip the next line
                continue
            elif line.startswith('[ bonds ]'):
                status = 2
                next(f)  # skip the next line
                continue
            elif line.startswith('['):
                status = 0

            if line.startswith(';') or not line.strip():
                continue

            if status == 1:
                parts = line.split()
                if len(parts) >= 5:
                    atom_index = int(parts[0])
                    atom_type = parts[1]
                    molgraph.add_node(atom_index, atom_type=atom_type)
            elif status == 2:
                parts = line.split()
                if len(parts) >= 3:
                    atom1 = int(parts[0])
                    atom2 = int(parts[1])
                    molgraph.add_edge(atom1, atom2)

    # # draw the graph and save as PNG
    # try:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(8, 6))
    #     pos = nx.spring_layout(molgraph)
    #     labels = nx.get_node_attributes(molgraph, 'atom_type')
    #     nx.draw(molgraph, pos, with_labels=True, labels=labels, node_size=500, node_color='lightblue', font_size=10)
    #     plt.title(f'Molecular Graph from {gro_file} and {itp_file}')
    #     plt.savefig(gro_file.replace('.gro', '_graph.png'))
    #     plt.close()
    # except ImportError:
    #     print("matplotlib not installed, skipping graph visualization.")

    return molgraph