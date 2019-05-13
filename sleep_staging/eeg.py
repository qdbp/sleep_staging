import glob
import os
import os.path as osp
import re
import sys
import traceback as trc
import typing as ty

import keras.utils.np_utils as kun
import numpy as np
import numpy.random as npr
import scipy.interpolate as sci
import scipy.signal as ssg

import pyedflib as pye

DEFAULT_PAGE = 30
DEFAULT_FS_OUT = 64

RMLG_DATE = re.compile(r"Recording Date:.*([0-9]{2}/[0-9]{2}/[0-9]{4})", re.M)
RMLG_RE = re.compile(
    r"^(\S+)\s+.*?([0-9]{2}:[0-9]{2}:[0-9]{2})" r".*SLEEP-.*?\s+([0-9]+)\s.*$",
    re.M,
)

RE_XML = re.compile(r"<SleepStage>([0-9])</SleepStage>", re.M)

# the channels which are considered "core" EEG information in the given order
# based on https://en.wikipedia.org/wiki/
# File:21_electrodes_of_International_10-20_system_for_EEG.svg
# CANONICAL_ORDER = ['f7-t3', 't3-t5', 'fp1-f3', 'f3-c3', 'c3-p3', 'p3-o1',
#                    'fp2-f4', 'f4-c4', 'c4-p4', 'p4-o2', 'f8-t4', 't4-t6',
#                    'c4-a1']

CANONICAL_ORDER = ["a1", "a2", "c3", "c4", "f3", "f4", "o1"]
NCAN = len(CANONICAL_ORDER)

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 5: "UNK"}
LABELS = [v for k, v in sorted(STAGE_NAMES.items())][:-1]

DCONF = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "5": 4,
    "9": 5,
    "W": 0,
    "R": 4,
    "?": 5,
    "U": 5,
}


def get_stage_signature(fname):
    counts = []
    stages = []
    last = None
    with open(fname, "r") as f:
        for row in f:
            l = row.rstrip("\n")
            if l != last:
                stages.append(l)
                counts.append(1)
                last = l
            else:
                counts[-1] += 1
    for item in zip(stages, counts):
        print(item)


def process_label_sequence(seq):
    labels = []
    skip_start = 0

    start_over = False
    for l in seq:
        if (l in ["U", "?", "9"]) and not start_over:
            skip_start += 1
        else:
            start_over = True
            try:
                labels.append(DCONF[l])
            except KeyError:
                raise ValueError(f"bad label {l} found, skipping file")

    return labels, skip_start


def read_xml(fname, dt=30):
    with open(fname, "r") as f:
        xml_seq = RE_XML.findall(f.read())
    return process_label_sequence(xml_seq)


def read_plain(fname):
    with open(fname, "r") as f:
        seq = f.read().split("\n")
    if seq[-1] == "":
        seq = seq[:-1]
    return process_label_sequence(seq)


def disambig_lf(fname):
    if fname.lower().endswith("xml"):
        return read_xml
    else:
        return read_plain


def summarize_edf_traces(efname: str) -> ty.List[str]:
    """
    Returns the list of channels found in edf file `efname`
    """
    try:
        r = pye.EdfReader(efname)
    except Exception:
        return []

    return sorted(set(l.decode("ascii").lower() for l in r.getSignalLabels()))


def get_edf_traces(
    efname: str, fs_out=64, label_samples=False, **kwargs
) -> ty.Optional[ty.Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Reads and EDF file to extract time traces.

    Traces are downasmpled to the given fs_out frequency.
    Traces which are at a lower sample rate than that are
    ignored.
    """

    print(f"Feading edf file {efname}")
    try:
        r = pye.EdfReader(efname)
    except OSError:
        trc.print_exc()
        print(f"Could not open {efname}, skipping")
        return None

    labs = set(l.decode("ascii").lower() for l in r.getSignalLabels())

    missing = [c for c in CANONICAL_ORDER if c not in labs]
    if any(missing):
        print(f"File {efname} is missing mandatory channels {missing}")
        return None

    label_fn_base = ".".join(efname.split(".")[:-1])
    if osp.isfile(label_fn_base + ".XML"):
        labels, skip_start = read_xml(label_fn_base + ".XML")
    elif osp.isfile(label_fn_base + ".edf.XML"):
        labels, skip_start = read_xml(label_fn_base + ".edf.XML")
    elif osp.isfile(label_fn_base + ".txt"):
        labels, skip_start = read_plain(label_fn_base + ".txt")
    elif osp.isfile(label_fn_base[:-2] + ".txt"):
        labels, skip_start = read_plain(label_fn_base[:-2] + ".txt")
    else:
        print(f"Sleep stage file {label_fn_base} not found, skipping")
        return None

    num_channels = r.signals_in_file
    traces = {}
    l = None

    for cx in range(num_channels):
        sig_header = r.getSignalHeader(cx)
        channel_name = sig_header["label"].decode("ascii").lower()

        if channel_name not in CANONICAL_ORDER:
            continue

        print(f"{efname}: read {channel_name: <7}: ", end="")
        trace = r.readSignal(cx)

        q = fs_out / sig_header["sample_rate"]
        num = int(len(trace) * q)

        # NOTE the padding makes resampling much faster
        lpad = (1 << int(np.ceil(np.log2(len(trace))))) - len(trace)
        lpad_rs = int(lpad * q)

        pad = np.zeros(lpad)
        ptrace = np.concatenate((trace, pad))

        print(
            f"resample fs {sig_header['sample_rate']} to {fs_out} "
            f"[{len(trace)} to {num} samples, "
            f"{len(ptrace)} to {num+lpad_rs} padded]) ",
            end="",
        )

        trunc_ix = -lpad_rs or None
        if num != len(trace):
            trace_rs = ssg.resample(ptrace, num + lpad_rs)[:trunc_ix]

        if l is None:
            l = len(trace_rs)
        else:
            l = min(l, len(trace_rs))

        traces[channel_name] = trace_rs
        print("done.")

    tr = np.zeros((NCAN, l))
    for lx, channel_name in enumerate(CANONICAL_ORDER):
        tr[lx, :] = norm_tr(np.asarray(traces[channel_name])[:l])

    try:
        out = label_pages(tr, labels, skip_start, **kwargs)
    except Exception:
        trc.print_exc()
        return None

    return (osp.basename(efname),) + out  # type: ignore


def label_pages(
    tr: np.ndarray,
    labels: ty.Iterable[int],
    skip_start: int,
    page_samples=DEFAULT_PAGE * DEFAULT_FS_OUT,
) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Generates a 1-hot label array for the trace `tr` given integer page labels
    `labels`.

    Arguments:
        tr: a 2-d array with dimensions (channels, samples)
        labels: the list of integer labels to use. Should correspond to the
            keys of `DCONF`.
        skip_start: will skip this many pages from the start of the trace.
        page_samples: consider a page to consist of this many samples

    Returns:
        a tuple of (cleaned trace, 1-hot labels, page weights)
    """

    extra_samples = tr.shape[1] % page_samples
    if extra_samples:
        print(f"trace has {extra_samples} dangling samples, truncating")

    end_ix = -extra_samples if extra_samples else None

    tr = tr[:, skip_start * page_samples : end_ix]
    ys = kun.to_categorical(labels, nb_classes=6)

    ws = np.ones(len(ys))
    # set we
    ws[np.where(ys[:, 5] > 0.5)[0]] = 0

    assert tr.shape[1] % page_samples == 0
    assert tr.shape[1] // page_samples == len(ys)
    assert len(ws) == len(ys)

    return tr, ys[:, :-1], ws


def list_edfs(d: str, blacklist=None, glb="*", ext="hdf5") -> ty.List[str]:
    fns = glob.glob(osp.join(d, glb + "." + ext))
    return sorted(
        [fn for fn in fns if not any([b in fn for b in blacklist or []])]
    )


def summarize_edf_dir(d: str, p=True):
    """
    Given a directory d, returns a visual summary of the trace content of all
    the EDF files in that directory.

    If p is True, prints the summary to terminal.
    """

    labss: ty.List[ty.List[str]] = []
    fns = list_edfs(d)
    bfns = [osp.basename(fn) for fn in fns]

    for fn in fns:
        labs = [l.upper() for l in summarize_edf_traces(fn)]
        labss.append(labs)

    all_labs: ty.Set[str] = set()
    for labs in labss:
        all_labs |= set(labs)
    union = sorted(union)

    maxlen = max([len(bfn) for bfn in bfns]) + 1
    header_lists = [
        [" " for i in range(maxlen + 2 * len(union))]
        for i in range(max([len(l) for l in union]))
    ]

    for i, lab in enumerate(union):
        for j, char in enumerate(lab):
            header_lists[j][2 * i + maxlen] = char

    header = "\n".join(["".join(row) for row in header])

    rows = "\n".join(
        [
            f"{bfns[i]: <{maxlen}}"
            + " ".join([("X" if u in ls else " ") for u in union])
            for i, ls in enumerate(labss)
        ]
    )

    universal = set()
    for lab in union:
        if all([lab in labs for labs in labss]):
            universal.add(lab)

    if p:
        print("\n".join([header, rows]))
        print(
            "\n labels found in all traces:\n\t"
            + "\n\t".join(sorted(universal))
        )
    return header, rows


def norm_tr(tr: np.ndarray, axis=0) -> np.ndarray:
    s = np.std(tr, axis=axis)
    assert not np.isclose(s, 0)
    return (tr - np.mean(tr, axis=axis)) / s
