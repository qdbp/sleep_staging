import numpy as np

from sleep_staging.main import score_top_k


def test_score_top_k():
    yt = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    yp = np.asarray(
        [
            [0.6, 0.3, 0.1],  # right at 0
            [0.3, 0.6, 0.1],  # right at 1
            [0.6, 0.3, 0.1],  # right at 0
            [0.1, 0.3, 0.6],  # right at 1
            [0.3, 0.6, 0.1],  # right at 0
            [0.3, 0.1, 0.6],  # right at 2
            [0.3, 0.6, 0.3],  # right at 0
            [0.2, 0.1, 0.6],  # right at 0
            [0.2, 0.1, 0.6],  # right at 0
            [0.3, 0.6, 0.1],  # right at 2
            [0.6, 0.1, 0.3],  # right at 1
            [0.1, 0.3, 0.6],  # right at 0
        ]
    )
    import matplotlib.pyplot as plt

    print(yt)
    print(list(np.argmax(yp, axis=1)))

    out = score_top_k(yt, yp)
    print(out)

    for cx, c in enumerate(["red", "blue", "green"]):
        for kx, ls in enumerate([":", "--", "-"]):
            plt.plot(
                out[kx + 1][cx],
                color=c,
                ls=ls,
                label=("class {}".format(cx) if kx == 2 else ""),
            )
    plt.legend()
    plt.show()

    # FIXME asserts???
