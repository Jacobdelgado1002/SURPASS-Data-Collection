import os
import shutil
from pathlib import Path
import re


def main():
    # Base directory containing the tissue folders
    base_dir = Path("C:/Users/jdelga16/Documents/Research/dataset_sliced")
    
    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist.")
        return

    # Iterate diligently through tissue_# directories
    for tissue_dir in base_dir.iterdir():
        if tissue_dir.is_dir() and tissue_dir.name.startswith("tissue_"):
            print(f"Processing {tissue_dir.name}...")
            
            # Find subdirectories that look like timestamps
            # Pattern: YYYYMMDD-HHMMSS-micros? or just generic timestamp-like
            # We will assume anything not already matching "#_*" is a target.
            # But let's be safer and look for digits-starting names or specific length.
            # The user's example: "20251217-150034-281409"
            
            subdirs = []
            for item in tissue_dir.iterdir():
                if item.is_dir():
                    # Simple heuristic: starts with '202' (for years 2020-2029) or is purely numeric-dash
                    if item.name.startswith("202") and "-" in item.name:
                        subdirs.append(item)
            
            # Sort them to ensure deterministic ordering (chronological usually)
            subdirs.sort(key=lambda x: x.name)
            
            # Rename them
            for index, subdir in enumerate(subdirs, start=1):
                new_name = f"{index}_cholecystectomy"
                new_path = tissue_dir / new_name
                
                print(f"  Renaming {subdir.name} -> {new_name}")
                
                # Check collision
                if new_path.exists():
                    print(f"    Warning: Target {new_name} already exists. Skipping.")
                else:
                    try:
                        subdir.rename(new_path)
                    except Exception as e:
                        print(f"    Error renaming: {e}")

if __name__ == "__main__":
    main()
