# validation/inspect_ninapro.py
import scipy.io as sio
import numpy as np
import os
import sys

def inspect_mat_file(filepath):
    """Print variable names and shapes in a .mat file."""
    print(f"\n--- Inspecting: {filepath} ---")
    try:
        data = sio.loadmat(filepath)
        for key in data:
            if not key.startswith('__'):
                print(f"Variable: {key}, shape: {data[key].shape}, dtype: {data[key].dtype}")
    except NotImplementedError:
        # Likely a v7.3 file, try h5py
        import h5py
        with h5py.File(filepath, 'r') as f:
            print("HDF5 keys:")
            for key in f.keys():
                print(f"Dataset: {key}, shape: {f[key].shape}")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Adjust this to your actual path
        path = r"E:\NinaProDB7"

    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith('.mat'):
                inspect_mat_file(os.path.join(path, file))
    elif os.path.isfile(path) and path.endswith('.mat'):
        inspect_mat_file(path)
    else:
        print("Please provide a valid .mat file or directory.")