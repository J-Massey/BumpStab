import sys
import os
from tkinter import Tcl
import numpy as np
from tqdm import tqdm


def bmask(dp, sp):
    fnsu, fnsv, fnsp, fnsb = fns(dp)
    for idx, (fnu, fnv, fnp, fnb) in tqdm(enumerate(zip(fnsu, fnsv, fnsp, fnsb)), total=len(fnsp)):
        u = np.load(os.path.join(dp, fnu))
        v = np.load(os.path.join(dp, fnv))
        p = np.load(os.path.join(dp, fnp))
        b = np.load(os.path.join(dp, fnb))
        bmask = np.where(b <= 1, False, True)
        # u = np.where(bmask, u, 0)
        np.save(os.path.join(sp, f"u_{idx}"), u)
        # v = np.where(bmask, v, 0)
        np.save(os.path.join(sp, f"v_{idx}"), v)
        # p = np.where(bmask, p, 0)
        np.save(os.path.join(sp, f"p_{idx}"), p)
        # Now remove the files
        try:
            os.remove(os.path.join("./uvp", fnu))
            os.remove(os.path.join("./uvp", fnv))
            os.remove(os.path.join("./uvp", fnp))
            os.remove(os.path.join("./uvp", fnb))
        except FileNotFoundError:
            pass


def fns(dp):
    fnsu = [
        fn
        for fn in os.listdir(dp)
        if fn.startswith(f"{fln}_u") and fn.endswith(f".npy")
    ]
    fnsu = Tcl().call("lsort", "-dict", fnsu)
    fnsv = [
        fn
        for fn in os.listdir(dp)
        if fn.startswith(f"{fln}_v") and fn.endswith(f".npy")
    ]
    fnsv = Tcl().call("lsort", "-dict", fnsv)
    fnsp = [
        fn
        for fn in os.listdir(dp)
        if fn.startswith(f"{fln}_p") and fn.endswith(f".npy")
    ]
    fnsp = Tcl().call("lsort", "-dict", fnsp)
    fnsb = [
        fn
        for fn in os.listdir(dp)
        if fn.startswith(bln) and fn.endswith(f".npy")
    ]
    fnsb = Tcl().call("lsort", "-dict", fnsb)
    return fnsu, fnsv, fnsp, fnsb

if __name__ == "__main__":
    fln = "fluAv"; bln = "bodAv"
    case = sys.argv[1]
    os.system(f"mkdir -p {case}/unmasked")
    dp = f"{os.getcwd()}/{case}/uvp"
    sp = f"{os.getcwd()}/{case}/unmasked"
    bmask(dp, sp)