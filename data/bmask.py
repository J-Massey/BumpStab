
import os
from tkinter import Tcl
import numpy as np
from tqdm import tqdm


def bmask(dp, sp):
    fnsu, fnsv, fnsp, fnsb = fns(dp)
    for idx, (fnu, fnv, fnp, fnb) in tqdm(enumerate(zip(fnsu, fnsv, fnsp, fnsb))):
        u = np.load(os.path.join(dp, fnu))
        v = np.load(os.path.join(dp, fnv))
        p = np.load(os.path.join(dp, fnp))
        b = np.load(os.path.join(dp, fnb))
        bmask = np.where(b <= 1, False, True)
        u = np.where(bmask, u, 0)
        np.save(os.path.join(sp, f"u_{idx}"), u)
        v = np.where(bmask, v, 0)
        np.save(os.path.join(sp, f"v_{idx}"), v)
        p = np.where(bmask, p, 0)
        np.save(os.path.join(sp, f"p_{idx}"), p)
        # Now remove the files
        print(f"Removing {fnu}, {fnv}, {fnb}")
        try:
            os.remove(os.path.join("./data", fnu))
            os.remove(os.path.join("./data", fnv))
            os.remove(os.path.join("./data", fnp))
            os.remove(os.path.join("./data", fnb))
        except FileNotFoundError:
            pass


def fns(dp):
    fnsu = [
        fn
        for fn in os.listdir(dp)
        if fn.startswith("fluAv_u") and fn.endswith(f".npy")
    ]
    fnsu = Tcl().call("lsort", "-dict", fnsu)
    fnsv = [
        fn
        for fn in os.listdir(dp)
        if fn.startswith("fluAv_v") and fn.endswith(f".npy")
    ]
    fnsv = Tcl().call("lsort", "-dict", fnsv)
    fnsp = [
        fn
        for fn in os.listdir(dp)
        if fn.startswith("fluAv_p") and fn.endswith(f".npy")
    ]
    fnsp = Tcl().call("lsort", "-dict", fnsp)
    fnsb = [
        fn
        for fn in os.listdir(dp)
        if fn.startswith('bodAv') and fn.endswith(f".npy")
    ]
    fnsb = Tcl().call("lsort", "-dict", fnsb)
    return fnsu, fnsv, fnsp, fnsb

if __name__ == "__main__":
    dp = f"{os.getcwd()}/data/0.001/128/uvp"
    sp = f"{os.getcwd()}/data/0.001/128/data"
    bmask(dp, sp)