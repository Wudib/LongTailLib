import h5py
import numpy as np
import os
import glob
from pathlib import Path


def average_data(algorithm="", dataset="", goal="", times=10):
    # Local (personalized) accuracy
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times, key='rs_test_acc')
    max_accuracy = [arr.max() for arr in test_acc] if len(test_acc) > 0 else []
    print("std for best local accuracy:", np.std(max_accuracy) if len(max_accuracy) > 0 else 0.0)
    print("mean for best local accuracy:", np.mean(max_accuracy) if len(max_accuracy) > 0 else 0.0)

    # Global model accuracy (if available)
    try:
        global_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times, key='rs_global_acc')
        if len(global_acc) > 0:
            max_g = [arr.max() for arr in global_acc]
            print("std for best global accuracy:", np.std(max_g))
            print("mean for best global accuracy:", np.mean(max_g))
    except Exception:
        # If no global dataset exists, skip silently
        pass


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, key='rs_test_acc'):
    test_acc = []

    # Build candidate roots robustly (supports running from repo root or system/)
    system_root = Path(__file__).resolve().parents[1]
    roots = [
        system_root / "results",                 # c:/.../system/results
        Path(os.getcwd()).resolve() / "results", # cwd/results
        Path(os.getcwd()).resolve().parent / "results", # parent_of_cwd/results
    ]
    # Deduplicate and keep only existing directories
    roots = [str(p) for p in dict.fromkeys(roots) if p.is_dir()]

    # 1) Old naming scheme: dataset_algorithm_goal_i.h5
    for i in range(times):
        base_name = f"{dataset}_{algorithm}_{goal}_{i}.h5"
        found = False
        for r in roots:
            fp = os.path.join(r, base_name)
            if os.path.exists(fp):
                test_acc.append(np.array(read_data_then_delete(fp, delete=False, key=key)))
                found = True
                break  # use first match
        # if not found, just continue

    if len(test_acc) > 0:
        return test_acc

    # 2) New naming scheme: dataset_algorithm_timestamp.h5
    candidates = []
    tried_patterns = []

    for r in roots:
        pattern_root = os.path.join(r, f"{dataset}_{algorithm}_*.h5")
        tried_patterns.append(pattern_root)
        candidates.extend(glob.glob(pattern_root))

        pattern_recursive = os.path.join(r, "**", f"{dataset}_{algorithm}_*.h5")
        tried_patterns.append(pattern_recursive)
        candidates.extend(glob.glob(pattern_recursive, recursive=True))

    # Deduplicate and order by modification time
    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No result files found for dataset={dataset}, algorithm={algorithm}.\n"
            f"Searched patterns: " + " ; ".join(tried_patterns))

    selected = candidates[-times:] if times is not None and times > 0 else candidates

    for path in selected:
        test_acc.append(np.array(read_data_then_delete(path, delete=False, key=key)))

    return test_acc


def read_data_then_delete(file_name, delete=False, key='rs_test_acc'):
    # Accept absolute path or relative .h5 file path; otherwise search known roots
    if file_name.endswith('.h5') and os.path.isfile(file_name):
        file_path = file_name
    else:
        system_root = Path(__file__).resolve().parents[1]
        roots = [
            system_root / "results",
            Path(os.getcwd()).resolve() / "results",
            Path(os.getcwd()).resolve().parent / "results",
        ]
        roots = [str(p) for p in roots if p.is_dir()]
        base = f"{file_name}.h5" if not file_name.endswith('.h5') else file_name
        file_path = None
        for r in roots:
            candidate = os.path.join(r, base)
            if os.path.isfile(candidate):
                file_path = candidate
                break
        if file_path is None:
            raise FileNotFoundError(f"Result file not found: {file_name}")

    with h5py.File(file_path, 'r') as hf:
        if key not in hf.keys():
            # If requested key absent, raise to allow caller to handle
            raise KeyError(f"Dataset '{key}' not found in H5 file: {file_path}")
        rs_test_acc = np.array(hf.get(key))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc