from modules.permutation_alignment import permutation_alignment7
from modules.losses import (
    LehmerMeanDSFLoss,
    PowerMeanDSFLoss,
    phase_invariant_cosine_squared_distance,
)
from typing import Union
from modules.utils import complex_abs, nextpow2
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import kapre
import numpy as np
from tqdm import tqdm
from modules.permutation_alignment import permutation_alignment7 as palign


class DirectionalSparseFiltering(tfk.Model):
    def __init__(
        self,
        x: Union[np.ndarray, tf.Tensor],
        n_src: int,
        fs: int,
        n_fft: int = None,
        hop_length: int = None,
        window: str = "hann_window",
        waveform_data_format: str = "channels_last",
        precision: int = 64,
        eps: float = 1e-12,
        inline_decoupling: bool = True,
        wiener_filter: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        n_samples, n_chan = x.shape

        # initialize STFT
        if n_fft is None:
            n_fft = nextpow2(2048 * fs / 16000)
            hop_length = n_fft // 4

        if type(hop_length) is float:
            assert 0 < hop_length <= 1
            hop_length = int(hop_length * n_fft)
        else:
            # assert type(hop_length) is int
            assert 0 < hop_length <= n_fft

        (
            self.stft_op,
            self.istft_op,
        ) = kapre.composed.get_perfectly_reconstructing_stft_istft(
            stft_input_shape=(n_samples, n_chan),
            istft_input_shape=None,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            forward_window_name=window,
            waveform_data_format=waveform_data_format,
            stft_data_format="channels_last",
            stft_name="stft",
            istft_name="istft",
        )

        self.wiener_filter = wiener_filter
        if self.wiener_filter:
            (
                self.stft_finetune_op,
                self.istft_finetune_op,
            ) = kapre.composed.get_perfectly_reconstructing_stft_istft(
                stft_input_shape=(n_samples, n_chan),
                istft_input_shape=None,
                n_fft=n_fft // 2,
                win_length=n_fft // 2,
                hop_length=hop_length // 2,
                forward_window_name=window,
                waveform_data_format="channels_last",
                stft_data_format="channels_last",
                stft_name="stft",
                istft_name="istft",
            )

        self.trim_begin = n_fft - hop_length
        self.n_freq = n_fft // 2 + 1

        # initialize variables
        assert precision in [32, 64]

        if precision == 32:
            self.real_dtype = tf.float32
            self.complex_dtype = tf.complex64
        elif precision == 64:
            self.real_dtype = tf.float64
            self.complex_dtype = tf.complex128

        self.eps = eps
        self.n_src = n_src
        self.n_chan = n_chan
        self.n_samples = n_samples
        self.inline_decoupling = inline_decoupling

        self.init_variables(x)

    def init_variables(self, x):
        x = tf.convert_to_tensor(x, dtype=self.real_dtype)
        self.X = self.stft(x)
        self.Xwhite, self.Q, self.Qinv = self.whitening(self.X)
        self.X_bar = self.column_norm(self.Xwhite)

    def stft(self, x):

        x = x[None, :, :]  # (1, n_samples, n_chan)

        X = self.stft_op(x)  # (1, n_frames, n_freq, n_chan)
        X = tf.squeeze(X)  # (n_frames, n_freq, n_chan)
        X = tf.transpose(X, (1, 0, 2))  # (n_freq, n_frames, n_chan)

        return X

    def istft(self, X, src_axis=3, frame_axis=1, freq_axis=0, chan_axis=2):

        # X.shape = (n_freq, n_frames, n_chan, n_src)
        X = tf.transpose(
            X, (src_axis, frame_axis, freq_axis, chan_axis)
        )  # (n_src, n_frames, n_freq, n_chan)

        x = self.istft_op(X)  # (n_src, n_samples*, n_chan)
        x = x[:, self.trim_begin : self.trim_begin + self.n_samples, :]

        return x

    def column_norm(self, X):

        norm = tf.norm(X, axis=-1, keepdims=True) + self.eps
        X_bar = X / norm

        return X_bar

    def whitening(self, X, eps=1e-16):
        # (n_freq, n_frames, n_chan)

        _, n_frames, _ = X.shape

        X = tf.transpose(X, (0, 2, 1))

        X = X - tf.reduce_mean(X, axis=2, keepdims=True)

        # (n_freq, n_chan, n_frames) @ (n_freq, n_frames, n_chan)
        #   --> (n_freq, n_chan, n_chan)
        covX = tf.matmul(X, X, adjoint_b=True) / (n_frames - 1.0)

        D, U, _ = tf.linalg.svd(covX)

        Drt = tf.sqrt(tf.maximum(D, eps))

        Dsqrt = tf.linalg.diag(tf.cast(tf.math.real(Drt), U.dtype))
        Disqrt = tf.linalg.diag(tf.cast(tf.math.real(1.0 / Drt), U.dtype))

        Q = tf.matmul(tf.matmul(U, Dsqrt), U, adjoint_b=True)
        Qinv = tf.matmul(tf.matmul(U, Disqrt), U, adjoint_b=True)

        Xwhite = tf.transpose(tf.matmul(Q, X), (0, 2, 1))  # (n_freq, n_frames, n_chan)

        return Xwhite, Q, Qinv

    def __fit__(
        self,
        epoch: int = 1000,
        optimizer: tfk.optimizers.Optimizer = tfk.optimizers.SGD(
            learning_rate=0.1, momentum=0.99, nesterov=True
        ),
        verbose: int = 1,
        abs_tol: float = 1e-9,
        rel_tol: float = 1e-8,
    ):

        prev_loss = -1.0

        with tqdm(range(epoch)) as t:
            for i in t:

                with tf.GradientTape() as tape:
                    loss = self.loss_function(self.X_bar)

                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                if tf.abs(prev_loss - loss) < abs_tol * self.n_freq:
                    print("Absolute tolerance reached.")
                    break

                if tf.abs(tf.abs(prev_loss - loss) / prev_loss) < rel_tol:
                    print("Relative tolerance reached.")
                    break

                prev_loss = loss

                t.set_postfix({"loss": loss.numpy()})

    def fit(self, *args, **kwargs):
        if hasattr(self, "loss_function"):
            self.__fit__(*args, **kwargs)
        else:
            raise NotImplementedError

    def extract(self, beta=12.5, max_iter=100, perm_tol=1e-8, proc_limit=1):

        if self.inline_decoupling:
            H = self.loss_function.inline_decoupling_op(self.loss_function.H)
        else:
            H = self.loss_function.H

        Hnorm = H / tf.norm(H, axis=1, keepdims=True)

        csim = tf.math.real(
            complex_abs(
                tf.reduce_sum(
                    Hnorm[:, None, :, :] * tf.math.conj(self.X_bar[:, :, :, None]),
                    axis=2,
                )
            )
        )  # (n_freq, n_frame, n_src)

        mask = tf.nn.softmax(-beta * csim, axis=-1)
        mask, _ = palign(mask, max_iter=max_iter, tol=perm_tol, proc_limit=proc_limit)

        Y = (
            self.X[:, :, :, None] * mask[:, :, None, :]
        )  # (n_freq, n_frame, n_chan, n_src)
        y = self.istft(Y)

        if self.wiener_filter:
            Xf = self.stft_finetune_op(y)  # (n_src, n_frames, n_freq, n_chan)
            Xn = Xf[..., 0] * tf.math.conj(Xf[..., 0])
            Xnc = tf.reduce_sum(Xn, axis=0, keepdims=True)

            G = Xn / Xnc

            Y = G[..., None] * Xf
            y = self.istft_finetune_op(Y)[
                :, self.trim_begin : self.trim_begin + self.n_samples, :
            ]

        return y


class LehmerMeanDSF(DirectionalSparseFiltering):
    def __init__(
        self,
        x: Union[np.ndarray, tf.Tensor],
        n_src: int,
        fs: int,
        r: float = 0.5,
        alpha: float = 1.0,
        inline_decoupling: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(x, n_src, fs, *args, **kwargs)

        self.loss_function = LehmerMeanDSFLoss(
            self.n_freq,
            self.n_chan,
            self.n_src,
            r=r,
            alpha=alpha,
            distance_func=phase_invariant_cosine_squared_distance,
            time_pooling_func=tf.reduce_mean,
            freq_pooling_func=tf.reduce_sum,
            inline_decoupling=inline_decoupling,
        )


class PowerMeanDSF(DirectionalSparseFiltering):
    def __init__(
        self,
        x: Union[np.ndarray, tf.Tensor],
        n_src: int,
        fs: int,
        p: float = -0.5,
        inline_decoupling: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(x, n_src, fs, *args, **kwargs)

        self.loss_function = PowerMeanDSFLoss(
            self.n_freq,
            self.n_chan,
            self.n_src,
            p=p,
            distance_func=phase_invariant_cosine_squared_distance,
            time_pooling_func=tf.reduce_mean,
            freq_pooling_func=tf.reduce_sum,
            inline_decoupling=inline_decoupling,
        )
