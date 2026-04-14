from pathlib import Path

from FileRepository import DataRepository

def verify_labelled_mat_files(base_dir="biosignal_data"):
    base_path = Path(base_dir)
    repository = DataRepository(base_path=base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory '{base_dir}' not found. Make sure you run this script from the correct location.")
        return

    # Specifically target *_labelled.mat files inside 'edited' directories
    search_pattern = "**/edited/**/*_labelled.mat"
    target_files = sorted(list(base_path.glob(search_pattern)))

    if not target_files:
        print(f"No *_labelled.mat files found in any 'edited' directories under '{base_dir}'.")
        return

    print(f"[{'-'*10} Verifying {len(target_files)} Labelled Files {'-'*10}]\n")

    for file_path in target_files:
        try:
            # Load the .mat file
            mat_data = repository.load_mat(str(file_path))
            
            # Extract keys, ignoring the internal MATLAB metadata keys (which start with '__')
            keys = [key for key in mat_data.keys() if not key.startswith('__')]
            
            # Get the relative path for a cleaner printout
            rel_path = file_path.relative_to(base_path.parent)
            
            print(f"📄 File: {rel_path}")
            print(f"🔑 Keys: {keys}")
            print([d[:] for d in mat_data['MIN_MAX_ROBUST']])
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")
            print("-" * 60)

    print(f"\n✅ Scan complete. Checked {len(target_files)} files.")

if __name__ == "__main__":
    verify_labelled_mat_files()