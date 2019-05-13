import typing as ty

import keras.backend as K
import keras.layers as klc
import keras.layers.convolutional as kcv
import keras.layers.noise as kno
import keras.models as krm
from keras.layers.merge import concatenate

K.set_image_data_format("channels_first")


def model_main(
    nc: int,
    nf_l: int,
    nt_l: int,
    nf_h: int,
    nt_h: int,
    *,
    mw=False,
    preload_weights: str = None,
    opti="adam",
    noise=False,
    noise_amp=0.2,
) -> krm.Model:

    if mw:
        pre_shape: ty.Tuple[int, ...] = (3,)
    else:
        pre_shape = ()

    i_x_l = klc.Input(shape=pre_shape + (nc, nf_l, nt_l), name="input.lo")
    i_x_h = klc.Input(shape=pre_shape + (nc, nf_h, nt_h), name="input.hi")

    conv_stack_h = [
        kcv.Conv2D(8, (3, 3), name="conv_h.0.0", activation="elu"),
        kcv.Conv2D(8, (3, 3), name="conv_h.0.1", activation="elu"),
        kcv.MaxPooling2D((2, 2)),
        kcv.Conv2D(8, (3, 3), name="conv_h.1.0", activation="elu"),
        kcv.Conv2D(8, (3, 3), name="conv_h.1.1", activation="elu"),
        kcv.MaxPooling2D((2, 2)),
        klc.Flatten(name="conv_h.flatten"),
    ]

    conv_stack_l = [
        kcv.Conv2D(8, (3, 3), name="conv_l.0.0", activation="elu"),
        kcv.Conv2D(8, (3, 3), name="conv_l.0.1", activation="elu"),
        kcv.MaxPooling2D((2, 2)),
        klc.Flatten(name="conv_l.flatten"),
    ]

    if mw:
        conv_stack_h = [
            klc.Permute((1, 4, 2, 3)),
            klc.Reshape((-1, nc, nf_h), input_shape=(3, nc, nf_h, nt_h)),
            klc.Permute((2, 1, 3)),
        ] + conv_stack_h

        conv_stack_l = [
            klc.Permute((1, 4, 2, 3)),
            klc.Reshape((-1, nc, nf_l), input_shape=(3, nc, nf_l, nt_l)),
            klc.Permute((2, 1, 3)),
        ] + conv_stack_l

    if noise:
        conv_stack_h = [
            kno.GaussianNoise(noise_amp, name="inoise_h")
        ] + conv_stack_h
        conv_stack_l = [
            kno.GaussianNoise(noise_amp, name="inoise_l")
        ] + conv_stack_l

    conv_l = i_x_l
    for layer in conv_stack_l:
        conv_l = layer(conv_l)

    conv_h = i_x_h
    for layer in conv_stack_h:
        conv_h = layer(conv_h)

    dn_suff = ".mw" if mw else ""
    merged_conv = concatenate([conv_l, conv_h])
    dense_stack = [
        klc.Dense(24, name="dense.0" + dn_suff, activation="elu"),
        klc.Dropout(0.5),
        klc.Dense(24, name="dense.1" + dn_suff, activation="elu"),
    ]

    y = merged_conv
    for layer in dense_stack:
        y = layer(y)

    y = klc.Dense(6, name="y" + dn_suff, activation="softmax")(y)
    m = krm.Model(inputs=[i_x_l, i_x_h], outputs=y)
    m.compile(optimizer=opti, loss="categorical_crossentropy")

    if preload_weights and mw:
        m.load_weights(preload_weights, by_name=True)

    return m
