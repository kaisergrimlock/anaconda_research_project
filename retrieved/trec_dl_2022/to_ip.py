from pathlib import Path
import shutil

root = Path(r"D:\Work\Research_Project\anaconda_research_project\retrieved\trec_dl_2022")

for folder in root.iterdir():
    if folder.is_dir() and folder.name.endswith("_instruct"):
        new_folder = folder.with_name(folder.name + "_ip")
        
        if new_folder.exists():
            print(f"Skipping, already exists: {new_folder}")
            continue
        
        shutil.copytree(folder, new_folder)
        print(f"Copied: {folder.name} -> {new_folder.name}")