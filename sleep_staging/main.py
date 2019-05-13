import glob
import os
import os.path as osp
import pickle as pkl
import traceback as trc
import typing as ty
from collections import Counter

import h5py
import keras.callbacks as kcb
import numpy as np
import numpy.random as npr
import scipy.signal as ssg
import tqdm
from sklearn.model_selection import KFold

from vnv.np import sl_window

from . import eeg
from . import learners as lrn

# these are traces with glaring artefacts
BLACKLIST = [
    "0909045",
    "1011045",
    "1304115",
    "1305065",
    "1312035",
    "1312195",
    "1410165",
    "1211155",
]


ROOT_ROOT = ".."
MODEL_DIR = ROOT_ROOT + "/data/models/"
PLOTS_ROOT = ROOT_ROOT + "/results/plots/"


def norm_trace(x):
    z = x - np.mean(x, axis=-1)[:, np.newaxis]
    return z / np.std(x, axis=-1)[:, np.newaxis]


def noise_trace(x, s=0.02):
    out = npr.normal(0, s, x.shape)
    return x + out


def noise_spectro(x, s=0.2):
    n = npr.normal(0, s, x.shape)
    return np.log(np.exp(x) + n ** 2)


# TODO magic numbers
def get_spectrum_high(x):
    return np.log(
        ssg.spectrogram(x, fs=64, nperseg=128, noverlap=96, window="hanning")[
            -1
        ]
        + 1e-9
    )


# TODO: magic numbers
def get_spectrum_low(x):
    assert not (x.shape[-1] % 16)
    xl = ssg.resample(x, x.shape[-1] // 16, axis=-1)
    return np.log(
        ssg.spectrogram(xl, fs=4, nperseg=32, noverlap=24, window="hanning")[-1]
        + 1e-9
    )


# TODO (2019) hdf5 is a useless format. We can deal with straight up .npz files
# and save a lot of hassle.
def process_to_hdf5(
    fns: ty.Iterable[str], extra_transformations=None, overwrite=False
):
    """
    Proccesses a raw eeg file to a standard HDF5 layout.

    Arguments:
        fns: filenames to process
        extra_transformations: the values of this dictionary are taken to be
            transformation funcions of the trace, precomputing some derivative
            and storing it under the associated key in the hdf5 file. You
            may use this to precompute expensive feature transforms.
        overwrite: by default, this function will skip writing HDF5 files that
            exists already. This function forces an overwrite.
    """

    extra_transformations = extra_transformations or {}

    for fn in fns:
        print(f"Processing {fn} to hdf5 file...", end="")

        hn = "".join(fn.split(".")[:-1]) + ".hdf5"
        if not overwrite and osp.isfile(hn):
            print(f"File {hn} exists, not overwriting.")
            continue
        file_traces = eeg.get_edf_traces(fn)

        if file_traces is None:
            print(f"Skipping unreadable edf file {fn}.")
            continue

        name, tr, y, w = file_traces
        try:
            f = h5py.File(hn, "w")
            f.attrs["name"] = name
            print(f"Writing tr, shape {tr.shape}")
            f.create_dataset("trace", tr.shape, dtype=np.float32)[:] = tr[:]
            print(f"Writing y, shape {y.shape}")
            f.create_dataset("y", y.shape, dtype=np.float32)[:] = y[:]
            print(f"Writing w, shape {w.shape}")
            f.create_dataset("w", w.shape, dtype=np.float32)[:] = w[:]
            for k, v in extra_transformations.items():
                out = v(tr)
                print(f"write {k}, shape {out.shape}")
                f.create_dataset(k, out.shape, dtype=np.float32)[:] = out[:]

            f.flush()
            f.close()

        # TODO be more specific
        except Exception:
            print("HDF5 creation failed with error:")
            trc.print_exc()
            try:
                os.remove(hn)
            except OSError:
                pass

        # otherwise we get funny OOMs
        del name, tr, y, w, f


def load_hdf5(
    fn: str, do_traces=False, write_to=None
) -> ty.Union[
    ty.Tuple[str, np.ndarray, np.ndarray, np.ndarray],
    ty.Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Loads the traces from an HDF5 file given by `fn`.

    if `write_to` is given, the traces are loaded to pre_existing arrays. The
    keys of `write_to` should match the keys of the HDF5 file. Otherwise, new
    arrays are allocated for the output.
    """

    print(f"Loading file {fn}")

    write_to = write_to or {}

    f = h5py.File(fn, "r")

    y = write_to.get("y") or np.zeros(f["y"].shape, dtype=f["y"].dtype)
    w = write_to.get("y") or np.zeros(f["w"].shape, dtype=f["w"].dtype)
    f["y"].read_direct(y)
    f["w"].read_direct(w)

    name: str = f.attrs["name"]

    try:
        if not do_traces:
            xh = write_to.get("x_h") or np.zeros(
                f["x_h"].shape, dtype=np.float32
            )
            f["x_h"].read_direct(xh)
            xl = write_to.get("x_l") or np.zeros(
                f["x_l"].shape, dtype=np.float32
            )
            f["x_l"].read_direct(xl)
            return (name, y, xh, xl, w)
        else:
            trc = write_to.get(
                "trc", np.zeros(f["trc"].shape, dtype=np.float32)
            )
            f["trc"].read_direct(trc)
            return (name, y, trc, w)
    finally:
        f.close()


# FIXME the xover argument is confusing and should be removed. I don't
# think it even makes sense since the overlap is fixed by the preprocessing
# for now keep it at 4 and pretend it doesn't exist.
def fetch_dir(
    d: str,
    blacklist: ty.List[str],
    xover=4,
    seed=1337,
    lim: ty.Optional[int] = None,
    multiwindow=False,
    vec_len=5,
    vec_skip=1,
):
    """
    Loads the eeg files from a directory into a trainable dataset.

    Arguments;
        lim:
        xover:
            overlap factor of spectrograms (75% = 4, etc). super hacky
        multiwindow:
            whether to return arrays suitable for multiwindow training
        vec_len:
            if multiwindow, length of vector sequences
        vec_skip:
            if multiwindow, offset between successive sequences, in pages.
            vec_skip < vec_len -> overlap (often desirable, no extra mem)
            vec_skip > vec_len -> missing data (probably undesirable)
    """
    # so that the centre page is well-defined
    assert vec_len % 2, "can only train on odd-length page sequences"

    print(f"Fetching dir {d}")

    fns = sorted(glob.glob(osp.join(d, "*.hdf5")))
    fns = [fn for fn in fns if all([b not in fn for b in blacklist])]
    if lim:
        fns = fns[:lim]

    l = len(fns)

    npr.seed(seed)
    npr.shuffle(fns)

    # NOTE these are read from the first file and then asserted equal for all
    # subsequent files
    nc0 = nf_l0 = nf_h0 = nt_l0 = nt_h0 = None

    lens = []
    xls, xhs, ys, ws = [], [], [], []
    n_classes = np.zeros(6, dtype=np.float32)

    # this loop double-checks that all the files have uniformly shaped input
    # data
    for fn in fns:
        f = h5py.File(fn, "r")

        ly = len(f["y"])
        lw = len(f["w"])

        if ly != lw:
            print(f"WARNING: {fn}: y/w mismatch ({ly}/{lw}), skipping")
            continue

        l = lw
        lens.append(l)

        y = np.zeros((l, 6), dtype=np.float32)
        w = np.zeros(l, dtype=np.float32)

        f["y"].read_direct(y)
        f["w"].read_direct(w)
        ys.append(y)
        ws.append(w)

        n_classes += np.sum(y, axis=0)

        nc = f["x_l"].shape[0]
        nf_l = f["x_l"].shape[1]
        nf_h = f["x_h"].shape[1]
        # "raw" n_t values which counts bins in the next page
        nt_l = (f["x_l"].shape[2] + xover - 1) // l
        nt_h = (f["x_h"].shape[2] + xover - 1) // l

        if nc0 is None:
            nc0 = nc
            nf_l0 = nf_l
            nf_h0 = nf_h
            nt_l0 = nt_l
            nt_h0 = nt_h

        nt_full_l, nt_full_h = f["x_l"].shape[2], f["x_h"].shape[2]

        # all files must have uniform channels, freq bins, and valid lengths
        assert nc0 == nc
        assert nf_h == nf_h0
        assert nf_l == nf_l0
        assert f["x_h"].shape[1] == nf_h and f["x_l"].shape[1] == nf_l

        assert (xover - 1 + nt_full_l) % l == 0
        assert (xover - 1 + nt_full_h) % l == 0

        assert (nt_full_l + xover - 1) // l == nt_l
        assert (nt_full_h + xover - 1) // l == nt_h

    # assign weights
    class_w = sum(lens) / (5 * n_classes[:5])
    for y, w in zip(ys, ws):
        for i in range(5):
            wh = np.where(y[:, i] > 0)[0]
            w[wh] *= class_w[i]

    size_l = nt_l0 - xover + 1
    size_h = nt_h0 - xover + 1

    print("Loading tensors from files...")
    for fx, fn in tqdm.tqdm(enumerate(fns)):
        f = h5py.File(fn, "r")
        l = lens[fx]

        if multiwindow:
            ys[fx] = ys[fx][vec_len // 2 : -vec_len // 2 + 1 : vec_skip]

            w_vecs = sl_window(ws[fx], vec_len, vec_skip)
            w_final = np.zeros(w_vecs.shape[0], dtype=np.float32)

            for wx, wv in enumerate(w_vecs):
                # any vector with a zero weight gets zero weight
                if np.min(wv) < 1e-6:
                    w_final[wx] = 0.0
                else:
                    # this does make sense, right?
                    w_final[wx] = np.mean(wv)

            ws[fx] = w_final
            assert w.shape[0] == y.shape[0]

        y, w = ys[fx], ws[fx]

        dump_l = np.zeros(f["x_l"].shape, np.float32)
        dump_h = np.zeros(f["x_h"].shape, np.float32)
        f["x_l"].read_direct(dump_l)
        f["x_h"].read_direct(dump_h)

        if multiwindow:
            xh = np.zeros((l, nc, nf_h, size_h), dtype=np.float16)
            xl = np.zeros((l, nc, nf_l, size_l), dtype=np.float16)
            for win in range(l):
                xl[win] = dump_l[:, :, win * nt_l : win * nt_l + size_l]
                xh[win] = dump_h[:, :, win * nt_h : win * nt_h + size_h]

            xl = sl_window(xl, vec_len, vec_skip)
            xh = sl_window(xh, vec_len, vec_skip)

        else:
            xh = np.zeros((l, nc, nf_h, size_h), dtype=np.float16)
            xl = np.zeros((l, nc, nf_l, size_l), dtype=np.float16)
            for win in range(l):
                xl[win] = dump_l[:, :, win * nt_l : win * nt_l + size_l]
                xh[win] = dump_h[:, :, win * nt_h : win * nt_h + size_h]

        assert xl.shape[0] == xh.shape[0] == y.shape[0]

        xls.append(xl)
        xhs.append(xh)

    print("")
    return (nc, nf_l, size_l, nf_h, size_h), xls, xhs, ys, ws, fns


def get_concat_arrs(ixes, xls, xhs, ys, ws, mw=False):
    if mw:
        for w in ws:
            w[0] = 0.0
            w[-1] = 0.0

    xh = np.concatenate([xhs[t] for t in ixes])
    xl = np.concatenate([xls[t] for t in ixes])
    y = np.concatenate([ys[t] for t in ixes])
    w = np.concatenate([ws[t] for t in ixes])

    if mw:
        xl = sl_window(xl, 3, 1)
        xh = sl_window(xh, 3, 1)
        y = y[1:-1]
        w = w[1:-1]

    return xl, xh, y, w


# TODO move this to evaluate?
def score_top_k(yt: ty.List[int], yp: np.ndarray):
    """
    Computes the top-k score.

    Arguments:
        yt: true labels
        yp: (n_samples, n_stages) raw (unargmaxed!) predicted labels

    Returns:
        accuracy, recall, precision and f1 by rank
    """
    ns, nc = yp.shape
    support = Counter(yt)

    # double argsort returns element order
    pref = np.argsort(np.argsort(-yp, axis=1))
    rank = [0 for _ in yt]
    for yx, y in enumerate(yt):
        rank[yx] = pref[yx, y]

    # rank[yx] <= k <--> y_i is in the top k predictions

    # acc_at[k] = #(r_i <= k)/#(yt)
    acc_at = []
    for k in range(nc):
        acc = sum(1 / ns for r in rank if r <= k)
        acc_at.append(acc)

    # rec_at[cls][k] = #(y_i == cls ∧ r_i <= k)/#(y_i == cls)
    rec_at = []
    for cls in support.keys():
        cls_r = []
        for k in range(nc):
            rec = sum(
                1 / support[cls]
                for rx, r in enumerate(rank)
                if r <= k and yt[rx] == cls
            )
            cls_r.append(rec)
        rec_at.append(cls_r)

    # prc_at[cls][k] = #(y_i == cls ∧ r_i <= k)/#(pref_i[cls] <= k)
    prc_at = []
    for cls in support.keys():
        cls_p = []
        for k in range(nc):
            # prow[cls] is our preference for cls for that sample
            # hence, if prow[cls] <= k, class cls is in our top k
            total = sum(1 for prow in pref if prow[cls] <= k)
            prc = sum(
                1 / total
                for rx, r in enumerate(rank)
                if r <= k and yt[rx] == cls
            )
            cls_p.append(prc)
        prc_at.append(cls_p)

    f1_at = []
    for cls in support.keys():
        f1s = []
        for k in range(nc):
            p, r = prc_at[cls][k], rec_at[cls][k]
            try:
                f1 = 2 * p * r / (p + r)
            except ZeroDivisionError:
                f1 = 0
            f1s.append(f1)
        f1_at.append(f1s)

    return acc_at, prc_at, rec_at, f1_at


def main() -> None:

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model_name", type=str, help="name to give the model")
    parser.add_argument("data_dir", type=str, help="name to give the model")
    parser.add_argument(
        "--bs", type=int, default=128, help="batch size to use for training"
    )
    parser.add_argument(
        "--bag",
        type=int,
        default=None,
        help="number of bags to use. Bagging is disabled if not passed",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="maximum number of traces to load from the directory. "
        "Useful for debugging",
    )
    parser.add_argument(
        "--sw", action="store_true", help="run single-window training"
    )

    args = parser.parse_args()
    mw = not args.sw

    model_params, xls, xhs, ys, ws, fns = fetch_dir(
        args.data_dir, BLACKLIST, 4, lim=args.max_traces, multiwindow=False
    )

    print(model_params)

    fns = [osp.basename(fn) for fn in fns]

    pred_dir = osp.join(args.model_name, "predict")
    wgh_dir = osp.join(args.model_name, "w")

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(wgh_dir, exist_ok=True)

    kf = KFold(n_splits=5)
    splits = kf.split(ys)

    for sx, (train, test) in enumerate(splits):
        print(f"FOLD {sx}")
        sname = f"fold_{sx}"

        if args.bag is not None:
            sub_train = [
                npr.choice(train, size=len(train)) for _ in range(args.bag)
            ]
            bag_suffs = [f"_bag_{bx}" for bx in range(args.bag)]
        else:
            bag_suffs = [""]
            sub_train = [train]

            txh = np.concatenate([xhs[t] for t in train])
            txl = np.concatenate([xls[t] for t in train])
            ty = np.concatenate([ys[t] for t in train])
            tw = np.concatenate([ws[t] for t in train])

        vxl, vxh, vy, vw = get_concat_arrs(test, xls, xhs, ys, ws, mw=mw)

        for bx, bag_suff in enumerate(bag_suffs):

            print(f"BAG {bx}")
            bname = sname + bag_suff
            wfn = f"{wgh_dir}/{bname}.hdf5"

            txl, txh, ty, tw = get_concat_arrs(
                sub_train[bx], xls, xhs, ys, ws, mw=mw
            )

            # XXX was this anything special?
            # swwfn = "../models/sw2/predict/fold_0.hdf5"

            m = lrn.model_main(
                *model_params,
                noise=True,
                mw=mw,
                # XXX preload_weights=swwfn,
            )
            m.fit(
                [txl, txh],
                ty,
                batch_size=args.bs,
                epochs=100,
                validation_data=([vxl, vxh], vy, vw),
                callbacks=[
                    kcb.EarlyStopping(patience=2),
                    kcb.ReduceLROnPlateau(patience=1, factor=0.5),
                    kcb.ModelCheckpoint(wfn, save_best_only=True),
                ],
            )

            m.load_weights(wfn)
            for tx in test:
                yp = m.predict(
                    [sl_window(xls[tx], 3, 1), sl_window(xhs[tx], 3, 1)]
                )
                with open(
                    osp.join(pred_dir, bname + f"_pred_{fns[tx]}.p"), "wb"
                ) as f:
                    pkl.dump((ws[tx][1:-1], yp, ys[tx][1:-1]), f)
