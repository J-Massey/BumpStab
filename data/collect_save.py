import os
import sys
from tkinter import Tcl
import numpy as np
from tqdm import tqdm


def fns(data_dir, root):
    fn = [
        fn
        for fn in os.listdir(data_dir)
        if fn.startswith(root) and fn.endswith(f".npy")
    ]
    fn = Tcl().call("lsort", "-dict", fn)
    return fn


def collect_data(fns, data_dir="./data"):
    resize_shape = np.load(f"{data_dir}/{fns[0]}")
    resize_shape = np.shape(resize_shape.squeeze())
    data = []
    for fn in tqdm(fns, desc="Loading data"):
        snap = np.load(f"{data_dir}/{fn}").squeeze()
        snap = np.resize(snap, resize_shape)
        data.append(snap)
        os.remove(f"{data_dir}/{fn}")
    return np.array(data).squeeze()


def remove_adjacent_allclose(arr):
    # Convert the array to a list for easier removal of elements
    arr_list = [arr[..., i] for i in range(arr.shape[-1])]
    i = 0
    while i < len(arr_list) - 1:
        if np.allclose(arr_list[i], arr_list[i + 1]):
            del arr_list[i + 1]
        else:
            i += 1
    # Convert the list back to a numpy array
    mopdified_arr = np.stack(arr_list, axis=-1)
    print(mopdified_arr.shape)
    return mopdified_arr


if __name__ == "__main__":
    case = sys.argv[1]
    data_dir = sp = f"{os.getcwd()}/{case}/data"
    root = "u"
    fn = fns(data_dir, root)
    u = collect_data(fn, data_dir)
    root = "v"
    fn = fns(data_dir, root)
    v = collect_data(fn, data_dir)
    root = "p"
    fn = fns(data_dir, root)
    p = collect_data(fn, data_dir)
    combined_data = np.stack([u, v, p], axis=0)
    combined_data = np.transpose(combined_data, (0, 3, 2, 1))
    combined_data = remove_adjacent_allclose(combined_data)
    np.save(f"{data_dir}/uvp.npy", combined_data)
