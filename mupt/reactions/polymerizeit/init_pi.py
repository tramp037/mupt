import polymerizeit as pi

import subprocess
import os
from datetime import datetime
from pathlib import Path

def generate_config_file(inputs, system):

    if 'dirname' not in inputs:
        print("No directory name provided in inputs. Using default directory name 'polymer'.")
        inputs['dirname'] = 'polymer'

    # write the init script
    with open("temp.sh", "w") as f:
        f.write(f"polymerizeit init << EOF\n")
        f.write(f"{inputs['dirname']}\n")
        f.write(f"\n")
        f.write(f"\n")
        f.write(f"{len(inputs['monomers'])}\n")
        for monomer in inputs['monomers']:
            f.write(f"{monomer['name']}\n")
            f.write(f"{3}\n")
            f.write(f"{monomer['smi']}\n")
        f.write(f"{len(inputs['repeat_units'])}\n")
        for repeat_unit in inputs['repeat_units']:
            f.write(f"{repeat_unit['name']}\n")
            f.write(f"{3}\n")
            f.write(f"{repeat_unit['smi']}\n")
        if 'protocol' not in inputs['reaction_engine']['inputs']:
            print("No protocol provided in reaction engine inputs. Using default protocol.")
            inputs['reaction_engine']['inputs']['protocol'] = 'default'
        if inputs['reaction_engine']['inputs']['protocol'] == 'default':
            f.write(f"{1}\n")
        else:
            print("Other protocols currently not supported through this interface. Using default protocol.")
            f.write(f"{1}\n")
        f.write(f"EOF\n")

    command = f"bash temp.sh"

    subprocess.run(command, shell=True, check=True)
    
    subprocess.run("rm temp.sh", shell=True, check=True)

    today = datetime.today().strftime('%Y-%m-%d')
    dir_name = f"{inputs['dirname']}_unknown_{today}"
    new_name = f"{inputs['dirname']}_{system['name']}"
    if os.path.exists(f"{new_name}"):
        archive_name = f"{new_name}_archive"
        newpath = rename_no_overwrite(new_name, archive_name)
        print(f"Directory {new_name} already exists. Renaming it to {newpath}.")

    os.rename(dir_name, new_name)

    inputs['directory'] = new_name
    return

def correct_config(inputs, system):

    return

def rename_no_overwrite(src: str | Path, dst: str | Path) -> Path:
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(src)

    parent = dst.parent
    stem = dst.name

    candidate = parent / stem
    i = 1
    while candidate.exists():
        candidate = parent / f"{stem}_{i}"
        i += 1

    src.rename(candidate)
    return candidate