from glob import glob
import os
import os.path as osp
import pickle
import re

import h5py
import matplotlib

matplotlib.use("Qt4Agg")
import matplotlib.patches as mpp
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib.ticker as mpt
import numpy as np
import sklearn.metrics as skm

import learners as lrn
import eeg
from main import fetch_dir, DATA_DIR, BLACKLIST


plt.style.use("ggplot")
# EEG_COLORS = {k: mpc.hex2color(v) for k, v in
#               {0: '#b5bd68',
#                1: '#e0e0e0',
#                2: '#c5c8c6',
#                3: '#969896',
#                4: '#81a2be',
#                5: '#de935f'}.items()
#               }
EEG_MARKERS = ["o", "s", "*", "^", "D"]
EEG_COLORS = [
    "grey",
    (204 / 255, 102 / 255, 102 / 255),
    (129 / 255, 162 / 255, 190 / 255),
    (181 / 255, 189 / 255, 104 / 255),
    (240 / 255, 198 / 255, 116 / 255),
]


def plot_stages(yp, page=eeg.DEFAULT_PAGE, fs=eeg.DEFAULT_FS_OUT):
    print("plotting sleep stages")
    spp = page * fs
    n = yp.shape[1]

    f = plt.figure()
    ax = f.add_subplot(111)

    divs = [(0, yp[0])]
    cur_y = yp[0]
    for ix, y in enumerate(yp[1:], 1):
        if not (np.argmax(y) == np.argmax(cur_y)):
            divs.append((spp * ix, cur_y))
            divs.append((spp * ix, y))
            cur_y = y

    divs.append((spp * len(yp) - 1, yp[-1]))
    divs = list(zip(*divs))

    for i in range(n):
        ax.fill_between(
            divs[0],
            0,
            np.asarray(divs[1])[:, i],
            transform=ax.get_xaxis_transform(),
            alpha=0.9,
            lw=0,
            color=EEG_COLORS[i],
        )

    patches = [
        mpp.Patch(color=EEG_COLORS[i], label=eeg.STAGE_NAMES[i])
        for i in range(n)
    ]
    ax.legend(handles=patches, loc=1)
    return f, ax


def generate_plots(fns, overwrite=True):
    for fn in fns:

        pfn = ".".join(fn.split(".")[:-1]) + ".png"
        if (not overwrite) and osp.isfile(pfn):
            print("{} exists, skipping".format(pfn))
            continue

        print("plotting {}".format(fn))

        f = h5py.File(fn, "r")
        tr = f["tr"]
        y = f["y"]
        name = f.attrs["name"]

        fig, ax = plot_stages(y)

        for i in range(tr.shape[0]):
            print("plotting trace {}".format(eeg.CANONICAL_ORDER[i].upper()))
            ax.plot(
                tr[i] + 10 * i,
                lw=0.1,
                label=eeg.CANONICAL_ORDER[i],
                color="k",
                alpha=0.75,
            )

        ax.set_title(name.split(".")[0])
        ax.get_yaxis().set_ticks([10 * i for i, _ in enumerate(tr)])
        ax.get_yaxis().set_ticklabels([x.upper() for x in eeg.CANONICAL_ORDER])
        ax.set_xlim((0, tr.shape[1]))

        fig.set_size_inches((1920 / 150, 1080 / 150))
        fig.savefig(pfn, dpi=150)

        f.close()
        plt.close(fig)

        del y, tr, f, fig, ax


def join_spectra(seqlen, ss):
    # ix, c, f, t
    sh = ss.shape
    outs = np.zeros((sh[0] + 1 - seqlen, sh[1], sh[2], sh[3] * seqlen))
    for i in range(sh[0] + 1 - seqlen):
        for j in range(seqlen):
            outs[i, :, :, sh[3] * j : sh[3] * (j + 1)] = ss[i + j]

    return outs


def evaluate_results(d, folds=3, bag=10):
    out = {}
    name_re = re.compile(r"_(.*)_R.hdf5.*")
    for f in range(folds):
        for b in range(bag):
            p = osp.join(d, "predict/fold_{}_bag_{}*pred*.p")
            for fn in glob(p):
                name = name_re.findall(fn)[0]
                if name not in out:
                    out[name] = {
                        "cm": None,
                        "prec": None,
                        "rec": None,
                        "acc": None,
                    }

                with open(fn, "rb") as f:
                    w, yp, yt = pload(f)


def present_bagging_results(resfn="validation.txt"):
    with open("bag_results.p", "rb") as f:
        ins = pkl.load(f)

    import sys

    outf = open(resfn, "w")
    # sys.stdout = outf

    outs = []
    outs_u = []
    for k in sorted(ins.keys()):
        v = ins[k]
        print("=" * 80)
        print("{:=^80}".format(" file {} ".format(k)))
        yt = v[0]
        yp = np.asarray(v[1:])

        # yp_ensemble
        ypc = np.argmax(yp, axis=2)
        # yp_aggregate
        ypa = np.argmax(np.sum(yp, axis=0), axis=1)

        corrects = [[0 for j in range(6)] for i in range(5)]
        totals = [[0 for j in range(6)] for i in range(5)]
        # for each window ...
        for ix, win in enumerate(ypc.T):
            bp = ypa[ix]
            agree = 0
            for cp in win:
                if bp == cp:
                    agree += 1
            totals[bp][agree - 1] += 1
            if bp == yt[ix]:
                corrects[bp][agree - 1] += 1

        print(totals)
        print(corrects)

        acc = skm.accuracy_score(yt, ypa)
        prec = skm.precision_score(yt, ypa, average=None)
        rec = skm.recall_score(yt, ypa, average=None)
        f1 = skm.f1_score(yt, ypa, average=None)
        cm = skm.confusion_matrix(yt, ypa)
        outs.append((acc, prec, rec, f1, cm, corrects, totals))

        unanimous = np.where(np.std(ypc, axis=0) == 0)[0]
        u_r = []
        raw_ws = []
        for i in range(5):
            where_lab = np.where(ypa == i)[0]
            u_r += [len(np.intersect1d(where_lab, unanimous)) / len(where_lab)]
            raw_ws += [len(np.intersect1d(where_lab, unanimous)) / len(ypa)]
            print(raw_ws)

        yt_u = yt[unanimous]
        yp_u = ypc[0][unanimous]
        acc_u = skm.accuracy_score(yt_u, yp_u)
        prec_u = skm.precision_score(yt_u, yp_u, average=None)
        rec_u = skm.recall_score(yt_u, yp_u, average=None)
        f1_u = skm.f1_score(yt_u, yp_u, average=None)
        cm_u = skm.confusion_matrix(yt_u, yp_u)
        outs_u.append(
            (acc_u, prec_u, rec_u, f1_u, cm_u, len(yt_u) / len(yt), u_r, raw_ws)
        )

    with open("bagging_outs.p", "wb") as f:
        pkl.dump(outs, f)

    with open("bagging_outs_u.p", "wb") as f:
        pkl.dump(outs_u, f)


def evaluate_unbagged(m, wfn, valfns, outfn="unbagged_outs.p"):

    m.load_weights(wfn)
    outs = []
    for valfn in valfns:
        valfn = osp.join(DATA_DIR_2, valfn) + ".hdf5"
        print("=" * 80)
        print("{:=^80}".format(valfn))
        n, y, sh, sl = load_hdf5(valfn)

        if seqlen > 1:
            sh = join_spectra(seqlen, sh)
            sl = join_spectra(seqlen, sl)
            y = y[seqlen - 1 :]
        print(sl.shape)
        print(sh.shape)

        y = np.argmax(y, axis=1)
        yp = np.argmax(m.predict({"x_h": sh, "x_l": sl})["y"], axis=1)

        acc = skm.accuracy_score(y, yp)
        prec = skm.precision_score(y, yp, average=None)
        rec = skm.recall_score(y, yp, average=None)
        f1 = skm.f1_score(y, yp, average=None)
        cm = skm.confusion_matrix(y, yp)
        print("accuracy: ", acc)
        print("report: ")
        print(skm.classification_report(y, yp))
        print("confusion matrix: ")
        print(cm)

        outs.append((acc, prec, rec, f1, cm))

    with open(outfn, "rb") as of:
        outs = pkl.load(of)


def mk_final_plots(
    outsfn,
    boutsfn,
    buoutsfn,
    plot_title="Untitled Plot",
    save=True,
    do_cm=False,
    do_scatter=False,
    do_scatter_u=False,
    do_perf_curve=False,
    do_accuracy=False,
):

    with open(outsfn, "rb") as f:
        outs = pkl.load(f)
    with open(boutsfn, "rb") as f:
        outs_b = pkl.load(f)
    with open(buoutsfn, "rb") as f:
        outs_bu = pkl.load(f)

    outss = [outs, outs_b, outs_bu]
    colors = [
        "grey",
        (204 / 255, 102 / 255, 102 / 255),
        (129 / 255, 162 / 255, 190 / 255),
        (181 / 255, 189 / 255, 104 / 255),
        (240 / 255, 198 / 255, 116 / 255),
    ]
    labels = ["Wake", "REM", "N1", "N2", "N3"]

    # confmats
    if do_cm:
        out = outss[1]
        acm = np.sum([i[4] for i in out], axis=0)
        acm = acm / np.sum(acm, axis=1)[:, np.newaxis]

        plt.figure()
        plt.matshow(acm, cmap="Reds", vmin=0, vmax=1)
        plt.gca().set_title("Confusion matrix for the bagged classifier")
        plt.gca().set_xlabel("Predicted stage")
        plt.gca().set_ylabel("True stage")
        plt.gca().set_xticklabels([None, "Wake", "REM", "N1", "N2", "N3"])
        plt.gca().set_yticklabels([None, "Wake", "REM", "N1", "N2", "N3"])

        plt.show()

    # accuracy
    if do_accuracy:
        correctss = []
        totalss = []
        for pat in outss[1]:
            correctss.append(pat[5])
            totalss.append(pat[6])

        correctss = np.asarray(correctss)
        totalss = np.asarray(totalss)

        st = np.sum(totalss, axis=0) + 1
        sc = np.sum(correctss, axis=0) + 0.2

        f, (ax_u, ax_l) = plt.subplots(2, sharex=True)
        ax_u.set_title(
            "Correctness vs. number of classifiers in agreement with majority"
        )
        for i in range(5):
            ax_u.plot(
                [j + 2 for j in range(5)],
                sc[i, 1:] / st[i, 1:],
                label=labels[i],
                color=colors[i],
                lw=3,
                marker=markers[i],
            )
            ax_l.bar(
                [j + 2 + 0.1 * (i - 2) for j in range(5)],
                st[i, 1:],
                color=colors[i],
                label="{}".format(labels[i]),
                lw=0,
                width=0.1,
            )
        ax_l.set_xlabel("Number of classifiers in agreement")
        ax_l.set_ylabel("Total number of instances")
        ax_u.set_ylabel("Proportion correct")
        ax_u.legend(bbox_to_anchor=(0, 1), loc=2, prop={"size": 10})
        plt.show()

    # scatters
    if do_scatter:
        titles = ["Unbagged classifier", "Bagged classifier"]
        f, (ax, ax_b) = plt.subplots(1, 2, sharey=True)
        axes = [ax, ax_b]
        for ix, ak in enumerate(axes):
            # confusion matrix for outs
            out = outss[ix]
            f1s = []
            for pat in out:
                f1 = pat[3]
                f1s.append(f1)
                ak.scatter(list(range(5)), f1, s=20, marker=".", color="k")

            f1s = np.median(np.asarray(f1s), axis=0)
            ak.scatter(
                list(range(5)),
                f1s,
                s=50,
                flierprops=dict(marker="x"),
                color="r",
                zorder=1000,
            )

            ak.set_title(titles[ix])
            # ak.set_xlabel('Sleep stage')
            if ix == 0:
                ak.set_ylabel("F1 score, X is median")
            ax.set_ylim(0, 1)
            ak.set_xlim(-0.5, 4.5)
            ak.set_xticklabels(["", "Wake", "REM", "N1", "N2", "N3", ""])
        plt.show()

    if do_scatter_u:
        # no_oracle
        out_u = outss[2]
        f, (ax_u, ax_o) = plt.subplots(1, 2, sharey=True)
        f1s = []
        f1s_u = []
        for pat in out_u:
            f1 = pat[3]
            f1s.append(f1)
            f1s_ux = []
            for i in range(5):
                ax_u.plot(
                    [i - 0.25, i - 0.25 + 0.5 * pat[6][i]],
                    [f1[i], f1[i]],
                    color="k",
                )
                ax_u.plot(
                    [i - 0.25, i + 0.25],
                    [f1[i], f1[i]],
                    ls="none",
                    marker="|",
                    color="k",
                )
                oracle_res = pat[6][i] * f1[i] + (1 - pat[6][i])
                f1s_ux.append(oracle_res)
                ax_o.plot(
                    [i - 0.25, i - 0.25 + 0.5 * pat[6][i]],
                    [oracle_res, oracle_res],
                    color="k",
                )
                ax_o.plot(
                    [i - 0.25, i + 0.25],
                    [oracle_res, oracle_res],
                    ls="none",
                    marker="|",
                    color="k",
                )

            f1s_u.append(f1s_ux)

        f1s = np.median(np.asarray(f1s), axis=0)
        f1s_u = np.median(np.asarray(f1s_u), axis=0)
        ax_u.scatter(
            list(range(5)),
            f1s,
            s=50,
            flierprops=dict(marker="x"),
            color="r",
            zorder=1000,
        )
        ax_o.scatter(
            list(range(5)),
            f1s_u,
            s=50,
            flierprops=dict(marker="x"),
            color="r",
            zorder=1000,
        )
        for ix, ak in enumerate([ax_u, ax_o]):
            ak.set_title(
                "Unanimous classifier"
                if ix == 0
                else "Unanimous + oracle classifier"
            )
            #            ak.set_xlabel('Sleep stage')
            if ix == 0:
                ak.set_ylabel("F1 score, X is median")
            ak.set_ylim(0, 1)
            ak.set_xlim(-0.5, 4.5)
            ak.set_xticklabels(["", "Wake", "REM", "N1", "N2", "N3", ""])

        plt.show()

    if do_perf_curve:
        f = plt.figure()
        out_b = outss[1]
        out_u = outss[2]
        f1s = []
        f1s_u = []
        ur = []
        for pat in out_u:
            f1s_u.append(pat[3])
            ur.append(pat[6])

        med_ur = np.median(np.asarray(ur), axis=0)
        med_f1s_u = np.median(np.asarray(f1s_u), axis=0)

        for pat in out_b:
            f1s.append(pat[3])

        mdef_f1s = np.median(np.asarray(f1s), axis=0)

        for i in range(5):
            plt.plot(
                [0, 1 - med_ur[i], 1],
                [mdef_f1s[i], med_f1s_u[i] * med_ur[i] + (1 - med_ur[i]), 1],
                marker=markers[i],
                label=labels[i],
                color=colors[i],
                lw=3,
            )

        plt.title(
            "Bagging performance vs. proportion of oracle labels by stage"
        )
        plt.xlabel("Median proportion of labels done by oracle")
        plt.ylabel("Median F1 stage accuracy")
        plt.legend(
            bbox_to_anchor=(0.98, 0.02),
            bbox_transform=plt.gca().transAxes,
            loc=4,
        )

        plt.show()


def mk_f1_plot(
    title,
    axlabel,
    labels,
    vals,
    res=(1920, 1080),
    mdpi=72,
    do_grid=True,
    do_label_pts=True,
):
    tlabs = ["Wake", "REM", "N1", "N2", "N3"]
    f, aks = plt.subplots(1, 5, sharey=True)
    f.set_size_inches(res[0] / mdpi, res[1] / mdpi)

    f.suptitle(title)
    aks[0].set_ylabel(axlabel)

    labels.append("MED")
    vals.append([np.median([v[j] for v in vals]) for j in range(len(vals[0]))])

    for ix, ak in enumerate(aks):

        xtl = ak.get_xticklabels()
        xtl[0].set_visible(False)
        ak.set_xlim(-2, 2)
        ak.set_ylim(0, 1)

        ak.set_title(tlabs[ix])
        ak.xaxis.set_ticks_position("none")
        ak.set_xticklabels([])
        # ghetto gridlines
        if do_grid:
            for i in range(1, 50):
                if i % 10:
                    ak.plot(
                        [-2, 2],
                        [i / 50, i / 50],
                        lw=0.5,
                        color="#a0a0a0",
                        ls=":",
                        zorder=-10,
                    )

            for i in range(1, 10):
                ak.plot(
                    [-2, 2],
                    [i / 10, i / 10],
                    lw=0.75,
                    color="#a0a0a0",
                    zorder=-10,
                )

        taken = [[0 for i in range(20)] for j in range(4)]

        for lx, l in enumerate(labels):
            ak.scatter(
                0,
                vals[lx][ix],
                marker=("." if lx + 1 < len(labels) else "x"),
                color=("k" if lx + 1 < len(labels) else "red"),
                s=(20 if lx + 1 < len(labels) else 100),
            )
            if do_label_pts:
                pref_y = int(20 * (vals[lx][ix]))
                if pref_y < 0:
                    pref_y = 0
                elif pref_y > 19:
                    pref_y = 19
                pref_x = 0

                ssign = 1
                sintv = 1
                while taken[pref_x][pref_y]:
                    pref_x += 1
                    if pref_x > 3:
                        pref_x = 0
                        pref_y += ssign * sintv
                        ssign *= -1
                        sintv += 1
                    if pref_y < 0:
                        pref_y = 0
                    elif pref_y > 19:
                        pref_y = 19

                taken[pref_x][pref_y] = 1
                ak.annotate(
                    "${:>02s}$".format(str(labels[lx])),
                    xy=(0, vals[lx][ix]),
                    xycoords="data",
                    xytext=(
                        0.7 * (-pref_x - 1 if pref_x < 2 else pref_x - 1),
                        pref_y / 20,
                    ),
                    textcoords="data",
                    arrowprops=dict(
                        facecolor="#808080",
                        edgecolor="none",
                        shrink=0.10,
                        headwidth=0,
                        frac=0,
                        width=0.8,
                    ),
                    horizontalalignment="right",
                    verticalalignment="top",
                )

    return f


def get_training_stats(n_train=60):
    # model_params, xls, xhs, ys, ws, fns =\
    #     fetch_dir(DATA_DIR, BLACKLIST, 4)

    # ns, ys, trcs = zip(*fetch_dirs([DATA_DIR_2], [BLACKLIST_2],
    #                                glbs=['*'], do_traces=True))

    ys = []
    print(len(sorted(glob(osp.join(DATA_DIR, "*.hdf5")))))
    for fn in sorted(glob(osp.join(DATA_DIR, "*.hdf5"))):
        bad = False
        for b in BLACKLIST:
            bad = bad or b in fn
        if bad:
            continue

        f = h5py.File(fn, "r")
        ys.append(f["y"])
        print(fn)

    print(len(ys))

    arr = np.zeros((len(ys), ys[0].shape[1]))
    for px, pat in enumerate(ys):
        pat = pat[:]
        pat = np.concatenate([np.array(pat), np.eye(6)])
        arr[px, :] = np.bincount(np.argmax(pat, axis=1)) - 1

    sm = np.sum(arr, axis=0)
    print(sm / np.sum(sm))

    arr /= np.sum(arr, axis=1)[:, np.newaxis]
    ixes = np.argsort(arr[:, 0])
    print(ixes)

    cs = np.cumsum(arr[ixes], axis=1)

    colors = [
        "grey",
        (204 / 255, 102 / 255, 102 / 255),
        (129 / 255, 162 / 255, 190 / 255),
        (181 / 255, 189 / 255, 104 / 255),
        (240 / 255, 198 / 255, 116 / 255),
    ]
    labels = [eeg.STAGE_NAMES[i] for i in range(6)]
    for i in range(5):
        plt.bar(
            range(len(arr)),
            arr[ixes, i],
            width=1,
            linewidth=0,
            label=labels[i],
            bottom=None if i == 0 else cs[:, i - 1],
            color=colors[i],
        )

    plt.ylim(0, 1)
    plt.xlim(0 - 0.5, len(arr) - 0.5)
    plt.legend()
    plt.title("Distribution of sleep stages among training patients.")
    plt.ylabel("Proportion of patient trace in stage")
    plt.xlabel("Patient number")
    plt.gcf().set_size_inches(10, 5)
    plt.savefig("stage_distribution.pdf", dpi=200)
    plt.show()


def get_raw_file_preds(fn):
    with open(fn, "rb") as f:
        data = pickle.load(f)

    y_p = data[1][:, :5]
    y_t = data[2][:, :5]
    w = data[0]

    return y_p, y_t, w


def get_preds(fns):
    y_ps = []
    y_ts = []
    ws = []
    for fn in fns:
        y_p, y_t, w = get_raw_file_preds(fn)
        y_ps.append(np.argmax(y_p, axis=1))
        y_ts.append(np.argmax(y_t, axis=1))
        ws.append(w)

    return y_ps, y_ts, ws


def get_preds_bagged(fns, bag, vote=True):
    y_tss = []
    y_pss = []
    wss = []
    for bx in range(bag):
        y_tss.append([])
        y_pss.append([])
        wss.append([])
        sub_fns = sorted([fn for fn in fns if f"bag_{bx}" in fn])
        for fx, fn in enumerate(sub_fns):
            y_p, y_t, w = get_raw_file_preds(fn)
            y_tss[bx].append(y_t)
            y_pss[bx].append(y_p)
            wss[bx].append(w)

            assert y_t.shape == y_tss[0][fx].shape
            assert y_p.shape == y_pss[0][fx].shape
            assert w.shape == wss[0][fx].shape

    if vote:
        y_ps = [np.argmax(yp, axis=1) for yp in np.mean(y_pss, axis=0)]
    else:
        y_ps = y_pss
    y_ts = [np.argmax(yt, axis=1) for yt in y_tss[0]]
    ws = wss[0]

    return y_ps, y_ts, ws


def get_file_scores(score_f, fns, bag=None, **kwargs):

    if bag:
        y_ps, y_ts, ws = get_preds_bagged(fns, bag)
    else:
        y_ps, y_ts, ws = get_preds(fns)

    return np.array(
        [
            score_f(
                yt,
                yp,
                # sample_weight=w,
                **kwargs,
            )
            for yp, yt, w in zip(y_ps, y_ts, ws)
        ]
    )


def get_fns(name):
    return sorted(glob(f"../models/{name}/predict/*.p"))


def hardmax(vec):
    x = np.argmax(vec)
    vec[:] = 0.0
    vec[x] = 1.0
    return vec


def plot_unan_oracle(at_k=0):
    fns = get_fns("mw4_bagged")
    y_pss, y_ts, ws = get_preds_bagged(fns, 10, vote=False)

    f, (ax_u, ax_o) = plt.subplots(1, 2, sharey=True)
    ax_u.set_title(r"unanimous")
    ax_o.set_title(r"oracle")
    ax_u.set_ylabel(r"F1 score, X is median")

    scores = np.zeros((len(y_ts), 5))
    oracle_scores = np.zeros((len(y_ts), 5))

    for ix, yt in enumerate(y_ts):
        ypb = np.array([yps[ix] for yps in y_pss])
        yp_votes = np.apply_along_axis(hardmax, -1, ypb).sum(axis=0)
        yp_cat = np.argmax(yp_votes, axis=1)

        oracle_scores
        unan_ixes = np.where(np.max(yp_votes, axis=1) >= 10 - at_k)[0]

        yt_unan = yt[unan_ixes]
        yp_unan = np.argmax(yp_votes, axis=1)[unan_ixes]

        score = skm.f1_score(yt_unan, yp_unan, average=None)
        scores[ix, :] = score

        for stage in range(5):
            n_unan = len(np.where(yp_unan == stage)[0])
            n_yp = len(np.where(yp_cat == stage)[0])

            p_unan = n_unan / n_yp

            ax_u.plot(
                [stage - 0.25, stage + 0.25],
                [score[stage], score[stage]],
                marker="|",
                color="k",
                ls="none",
            )
            ax_u.plot(
                [stage - 0.25, stage - 0.25 + 0.5 * p_unan],
                [score[stage], score[stage]],
                color="k",
            )

            oracle_score = p_unan * score[stage] + (1 - p_unan)
            oracle_scores[ix, stage] = oracle_score

            ax_o.plot(
                [stage - 0.25, stage + 0.25],
                [oracle_score, oracle_score],
                marker="|",
                color="k",
                ls="none",
            )
            ax_o.plot(
                [stage - 0.25, stage - 0.25 + 0.5 * p_unan],
                [oracle_score, oracle_score],
                color="k",
            )

    med_scores = np.median(scores, axis=0)
    med_oscores = np.median(oracle_scores, axis=0)
    ax_u.set_xticklabels([""] + eeg.LABELS)
    ax_o.set_xticklabels([""] + eeg.LABELS)
    ax_u.plot([0, 1, 2, 3, 4], med_scores, marker="x", color="red", ls="none")
    ax_o.plot([0, 1, 2, 3, 4], med_oscores, marker="x", color="red", ls="none")
    plt.gcf().set_size_inches((10, 5))
    plt.ylim((0, 1))
    plt.savefig("unweighted_unan_oracle.pdf", dpi=200)


def plot_scatter():
    score_f = skm.f1_score

    plt.subplot(1, 4, 1)
    fns = get_fns("sw2")
    plt.boxplot(
        np.array(get_file_scores(score_f, fns, average=None)),
        labels=eeg.LABELS,
        flierprops=dict(marker="x"),
    )
    plt.ylim((0, 1))
    plt.title("Single Page, Unbagged")
    plt.ylabel("F1 Score")

    plt.subplot(1, 4, 2)
    fns = get_fns("mw2")
    plt.boxplot(
        np.array(get_file_scores(score_f, fns, average=None)),
        labels=eeg.LABELS,
        flierprops=dict(marker="x"),
    )
    plt.ylim((0, 1))
    plt.title("Multi Page, Unbagged")
    plt.ylabel("F1 Score")

    plt.subplot(1, 4, 3)
    plt.ylim((0, 1))
    fns = get_fns("sw2_bagged")
    plt.boxplot(
        np.array(get_file_scores(score_f, fns, bag=10, average=None)),
        labels=eeg.LABELS,
        flierprops=dict(marker="x"),
    )
    plt.title("Single Page, Bagged")
    plt.ylabel("F1 Score")

    plt.subplot(1, 4, 4)
    plt.ylim((0, 1))
    fns = get_fns("mw4_bagged")
    plt.boxplot(
        np.array(get_file_scores(score_f, fns, bag=10, average=None)),
        labels=eeg.LABELS,
        flierprops=dict(marker="x"),
    )
    plt.title("Multi Page, Bagged")
    plt.ylabel("F1 Score")

    plt.gcf().set_size_inches((15, 7.5))
    plt.tight_layout()
    plt.savefig("base_scatter.pdf", dpi=200)
    plt.show()


def count_agree(row):
    y_t = row[0]
    y_p = row[1:]
    return np.count_nonzero(y_t == y_p)


def plot_accuracy():

    fns = get_fns("mw4_bagged")
    y_pss, y_ts, ws = get_preds_bagged(fns, 10, vote=False)

    y_t = np.concatenate(y_ts, axis=0)

    y_vote = np.concatenate(
        [np.argmax(yp, axis=1) for yp in np.mean(y_pss, axis=0)], axis=0
    )

    y_indiv = np.zeros((10,) + y_vote.shape)
    for bx in range(10):
        bag_pred = np.concatenate(
            [np.argmax(ypb, axis=-1) for ypb in y_pss[bx]], axis=0
        )
        y_indiv[bx, :] = bag_pred

    correct_map = y_vote == y_t
    agree_map = np.apply_along_axis(
        count_agree, 0, np.concatenate([y_vote[None, :], y_indiv])
    )

    agree_accs = np.zeros((5, 9))
    agree_counts = np.zeros((5, 9))

    for stage in range(5):
        stage_ixes = np.where(y_vote == stage)[0]
        n_stage = len(stage_ixes)
        for agreement in range(2, 11):
            ixes = np.where(
                np.logical_and(y_vote == stage, agree_map == agreement)
            )[0]
            agree_counts[stage, agreement - 2] = len(ixes) / n_stage
            agree_accs[stage, agreement - 2] = skm.accuracy_score(
                y_t[ixes], y_vote[ixes]
            )

    f, (ax_u, ax_l) = plt.subplots(2, sharex=True)
    ax_u.set_title(
        "Accuracy vs. number of classifiers in agreement with majority"
    )
    ax_l.set_title(
        "Distribution of number of classifiers in agreement with final vote"
    )
    for stage in range(5):
        ax_u.plot(
            [*range(2, 11)],
            agree_accs[stage, :],
            label=eeg.LABELS[stage],
            color=EEG_COLORS[stage],
            lw=3,
            marker=EEG_MARKERS[stage],
        )
        ax_l.bar(
            [j + 0.1 * (stage) for j in range(2, 11)],
            agree_counts[stage, :],
            color=EEG_COLORS[stage],
            label=f"{eeg.LABELS[stage]}",
            lw=0,
            width=0.1,
        )
    ax_l.set_xlabel("Number of classifiers in agreement")
    ax_l.set_ylabel("Fractional number of instances")
    ax_u.set_ylabel("Accuracy")
    ax_u.legend(bbox_to_anchor=(0, 1), loc=2, prop={"size": 10})

    plt.gcf().set_size_inches((10, 6))
    plt.tight_layout()
    plt.savefig("agreement.pdf", dpi=200)

    accs = np.zeros((12, 5))
    accs[11, :] = 1
    p_accords = np.zeros((12, 5))
    p_accords[11, :] = 0

    for cutoff in range(2, 11):
        for stage in range(5):
            agree_ixes = np.where(
                np.logical_and(y_vote == stage, agree_map >= cutoff)
            )[0]
            p_accord = len(agree_ixes) / len(np.where(y_vote == stage)[0])
            p_accords[cutoff, stage] = p_accord

            acc = skm.accuracy_score(y_t[agree_ixes], y_vote[agree_ixes])
            acc = p_accord * acc + (1 - p_accord)

            accs[cutoff, stage] = acc

    print(accs)
    print(p_accords)
    plt.figure()
    for stage in range(5):
        plt.plot(
            accs[2:, stage],
            1 - p_accords[2:, stage],
            marker=EEG_MARKERS[stage],
            label=eeg.LABELS[stage],
            color=EEG_COLORS[stage],
        )

    plt.title("Percentage of labels by oracle vs. accuracy")
    plt.ylabel("Mean proportion of labels done by oracle")
    plt.xlabel("Mean accuracy by stage")
    plt.legend(
        bbox_to_anchor=(0.98, 0.02), bbox_transform=plt.gca().transAxes, loc=4
    )

    plt.gcf().set_size_inches((10, 6))
    plt.tight_layout()
    plt.savefig("perf2.pdf", dpi=200)


if __name__ == "__main__":
    fns = get_fns("mw3_bagged")
    scores = get_file_scores(skm.f1_score, fns, bag=10, average=None)
    print(np.median(scores, axis=0))
    # plot_accuracy()
    # plot_unan_oracle()
    # plot_scatter()
