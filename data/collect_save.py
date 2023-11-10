import os
import sys
from tkinter import Tcl
import numpy as np
from tqdm import tqdm


def fns(data_dir):
    fnsu = [
        fn
        for fn in os.listdir(data_dir)
        if fn.startswith(f"u") and fn.endswith(f".npy")
    ]
    fnsu = Tcl().call("lsort", "-dict", fnsu)
    fnsv = [
        fn
        for fn in os.listdir(data_dir)
        if fn.startswith(f"v") and fn.endswith(f".npy")
    ]
    fnsv = Tcl().call("lsort", "-dict", fnsv)
    fnsp = [
        fn
        for fn in os.listdir(data_dir)
        if fn.startswith(f"p") and fn.endswith(f".npy")
    ]
    fnsp = Tcl().call("lsort", "-dict", fnsp)
    return fnsu, fnsv, fnsp


def collect_data(data_dir):
    data_u, data_v, data_p = [], [], []
    diffs = []

    fnsu, fnsv, fnsp = fns(data_dir)
    for idx, (fnu, fnv, fnp) in tqdm(
        enumerate(zip(fnsu, fnsv, fnsp)), total=len(fnsp)
    ):
        u = np.load(os.path.join(data_dir, fnu)).squeeze()
        v = np.load(os.path.join(data_dir, fnv)).squeeze()
        p = np.load(os.path.join(data_dir, fnp)).squeeze()

        if idx > 0:
            diff = np.linalg.norm(u - data_u[-1])
            diffs.append(diff)

        data_u.append(u)
        data_v.append(v)
        data_p.append(p)

        os.remove(f"{data_dir}/{fnu}")
        os.remove(f"{data_dir}/{fnv}")
        os.remove(f"{data_dir}/{fnp}")

    # Calculate average difference
    avg_diff = np.mean(diffs)

    processed_data_u, processed_data_v, processed_data_p = (
        [data_u[0]],
        [data_v[0]],
        [data_p[0]],
    )
    for idx, diff in enumerate(diffs):
        if diff < 0.4 * avg_diff:
            # Too similar, drop the next data point
            print(f"Diff : {diff}, dropped idx {idx}")
            continue
        elif diff > 1.5 * avg_diff:
            # Too different, interpolate between data[idx] and data[idx+1]
            print(f"Diff : {diff}, interped idx {idx}")
            interpolated = 0.5 * (data_u[idx] + data_u[idx + 1])
            processed_data_u.append(interpolated)
            interpolated = 0.5 * (data_v[idx] + data_v[idx + 1])
            processed_data_v.append(interpolated)
            interpolated = 0.5 * (data_p[idx] + data_p[idx + 1])
            processed_data_p.append(interpolated)

        processed_data_u.append(data_u[idx + 1])
        processed_data_v.append(data_v[idx + 1])
        processed_data_p.append(data_p[idx + 1])

    uvp = np.stack(
        [
            np.array(processed_data_u),
            np.array(processed_data_v),
            np.array(processed_data_p),
        ],
        axis=0,
    )
    return np.transpose(uvp, (0, 3, 2, 1))


if __name__ == "__main__":
    case = sys.argv[1]
    data_dir = sp = f"{os.getcwd()}/{case}/unmasked"
    uvp = collect_data(data_dir)
    np.save(f"{data_dir}/uvp.npy", uvp)
