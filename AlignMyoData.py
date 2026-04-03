import os
import scipy.io
import numpy as np

def align_myo_data(input_dir=r".\collected_data", subject="P3"):
    """
    Iterates through P1M1 to P1M9, subtracts the DC offset, 
    and saves as P1M1_edit.mat
    """
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return

    print(f"Starting alignment for {subject} in {input_dir}...\n")

    for m in range(1, 10):
        input_filename = f"{subject}M{m}.mat"
        output_filename = f"{subject}M{m}_edit.mat"
        
        input_path = os.path.join(input_dir+"/raw/", input_filename)
        output_path = os.path.join(input_dir+"/edit/", output_filename)

        if not os.path.exists(input_path):
            print(f"[-] Skipping {input_filename}: File not found.")
            continue

        # 1. Load the .mat file
        mat = scipy.io.loadmat(input_path)

        if 'EMGDATA' not in mat:
            print(f"[-] Skipping {input_filename}: 'EMGDATA' key not found.")
            continue

        raw_data = mat['EMGDATA'].astype(np.float32)

        aligned_data = raw_data - np.mean(raw_data, axis=1, keepdims=True)

        scipy.io.savemat(output_path, {'EMGDATA': aligned_data})
        print(f"[+] Successfully processed and saved: {output_filename}")

    print("\nAlignment complete!")

if __name__ == "__main__":
    align_myo_data()