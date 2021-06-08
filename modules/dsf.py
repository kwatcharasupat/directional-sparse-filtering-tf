from modules.losses import (
    LehmerMeanDSFLoss,
    phase_invariant_cosine_squared_distance,
)
from typing import Callable, Union
from modules.utils import nextpow2
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import kapre
import numpy as np
from tqdm import tqdm


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
        epoch: int = 10000,
        optimizer: tfk.optimizers.Optimizer = tfk.optimizers.Adam(),
        verbose: int = 1,
        abs_tol: float = 1e-8,
        rel_tol: float = 1e-5,
    ):

        # if verbose > 0:
        #     _tqdm = tqdm
        # else:
        #     _tqdm = lambda var: var

        prev_loss = -1.0

        with tqdm(range(epoch)) as t:
            for i in t:

                with tf.GradientTape() as tape:
                    loss = self.loss_function(self.X_bar)

                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                if tf.abs(prev_loss - loss) < abs_tol:
                    break

                if tf.abs(tf.abs(prev_loss - loss) / prev_loss) < rel_tol:
                    break

                prev_loss = loss

                t.set_postfix({"loss": loss.numpy()})

    def fit(self, *args, **kwargs):
        if hasattr(self, "loss_function"):
            self.__fit__(*args, **kwargs)
        else:
            raise NotImplementedError

    def extract(self):

        # X_bar.shape = (n_freq, n_frame, n_chan)
        # H.shape = (n_freq, n_frame, n_chan, n_src)

        pass


class LehmerMeanDSF(DirectionalSparseFiltering):
    def __init__(
        self,
        x: Union[np.ndarray, tf.Tensor],
        n_src: int,
        fs: int,
        r: float = 0.5,
        alpha: float = 1.0,
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
        )


class PowerMeanDSF(DirectionalSparseFiltering):
    def __init__(
        self,
        x: Union[np.ndarray, tf.Tensor],
        n_src: int,
        fs: int,
        p: float = -0.5,
        *args,
        **kwargs,
    ):
        super().__init__(x, n_src, fs, *args, **kwargs)

        self.loss_function = LehmerMeanDSFLoss(
            self.n_freq,
            self.n_chan,
            self.n_src,
            p=p,
            distance_func=phase_invariant_cosine_squared_distance,
            time_pooling_func=tf.reduce_mean,
            freq_pooling_func=tf.reduce_sum,
        )
