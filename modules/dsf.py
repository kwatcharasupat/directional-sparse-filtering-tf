from tensorflow.python.platform.tf_logging import warning
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
        hop_length: Union[float, int] = None,
        window: str = "hann_window",
        waveform_data_format: str = "channels_last",
        precision: int = 32,
        inline_decoupling: bool = False,
        wiener_filter: bool = False,
        use_real_proxies: bool = False,
    ):
        """
        Base Class for Directional Sparse Filtering (DSF)

        This class serves as a "system" wrapper for the entire DSF pipeline rather than a specific implementation.
        The "trainable variables" are contained in the specific loss module rather than in this class.

        Parameters
        ----------
        x : Union[np.ndarray, tf.Tensor], shape=(n_samples, n_chan)
            input mixture
        n_src : int
            number of sources
        fs : int
            sampling rate in Hertz
        n_fft : int, optional
            FFT length, by default None
            If None, calculated from `fs` using `n_fft = nextpow2(2048*fs/16000)`
        hop_length : int, optional
            Hop size, by default None
            If integer, treated as number of samples. If float, treated as a fraction of `n_fft`.
            If None, default to `n_fft//4` if `n_fft` is also None. Otherwise, default kapre behaviour.
        window : str, optional
            Window function, by default "hann_window"
            See `kapre.backend.get_window()` for available windows
        waveform_data_format : str, optional
            The audio data format of waveform batch, by default "channels_last"
            See https://kapre.readthedocs.io/en/latest/composed.html
        precision : int, optional
            Precision in bit, by default 64
        inline_decoupling : bool, optional
            Whether to use inline decoupling, by default False
        wiener_filter : bool, optional
            Whether to use Wiener filter to postprocess the extracted signal, by default False
        """
        super().__init__()

        # initialize variables
        assert precision in [
            32,
            64,
        ], "Only 32-bit and 64-bit precision are currently supported"

        if precision == 32:
            tf.keras.backend.set_floatx("float32")
            self.real_dtype = tf.float32
            self.complex_dtype = tf.complex64
        elif precision == 64:
            # raise NotImplementedError
            tf.keras.backend.set_floatx("float64")
            self.real_dtype = tf.float64
            self.complex_dtype = tf.complex128
        else:
            raise NotImplementedError

        if inline_decoupling:
            warning(
                "Inline decoupling currently has inconsistent behaviour on Tensorflow. We recommend that this is turned off for now."
            )

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

        self.n_src = n_src
        self.n_chan = n_chan
        self.n_samples = n_samples

        self.use_real_proxies = use_real_proxies
        # if self.use_real_proxies:
        #     assert (
        #         not inline_decoupling
        #     ), "inline decoupling is not available in real proxies mode"
        self.inline_decoupling = inline_decoupling

        self.init_variables(x)
        self.n_frames = self.X.shape[1]

    def init_variables(self, x):

        x = tf.convert_to_tensor(x, dtype=self.real_dtype)
        self.X = self.stft(x)

        with tf.device("/CPU:0"):
            self.Xwhite, self.Q, self.Qinv = self.whitening(self.X)
            self.X_bar = self.column_norm(self.Xwhite)

        self.X_bar_ = self.X_bar

        if self.use_real_proxies:
            self.X_bar = tf.stack(
                [tf.math.real(self.X_bar), tf.math.imag(self.X_bar)], axis=-1
            )

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

    def column_norm(self, X, eps=1e-8):
        X_bar, _ = tf.linalg.normalize(X, axis=-1)
        
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

        Drt = tf.sqrt(D)  # tf.sqrt(tf.maximum(D, eps))

        Dsqrt = tf.linalg.diag(tf.cast(tf.math.real(Drt), U.dtype))
        Disqrt = tf.linalg.diag(tf.cast(tf.math.real(1.0 / Drt), U.dtype))

        Q = tf.matmul(tf.matmul(U, Disqrt), U, adjoint_b=True)
        Qinv = tf.matmul(tf.matmul(U, Dsqrt), U, adjoint_b=True)

        Xwhite = tf.transpose(tf.matmul(Q, X), (0, 2, 1))  # (n_freq, n_frames, n_chan)

        return Xwhite, Q, Qinv

    def __fit__(
        self,
        epoch: int = 1000,
        optimizer: tfk.optimizers.Optimizer = tfk.optimizers.SGD(
            learning_rate=1.0, momentum=0.9, nesterov=True
        ),
        verbose: int = 1,
        abs_tol: float = 1e-8,
        rel_tol: float = 1e-5,
    ):

        prev_loss = None

        with tqdm(range(epoch)) as t:
            for i in t:

                with tf.GradientTape() as tape:
                    loss = self.loss_function(self.X_bar)

                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                if (
                    prev_loss is not None
                    and tf.abs(prev_loss - loss) < abs_tol * self.n_freq
                ):
                    print("Absolute tolerance reached.")
                    break

                if (
                    prev_loss is not None
                    and tf.abs(tf.abs(prev_loss - loss) / prev_loss) < rel_tol
                ):
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

        with tf.device("/CPU:0"):

            if self.inline_decoupling:
                self.loss_function.inline_decoupling_op()

            H = self.loss_function.H

            if self.use_real_proxies:
                H = tf.complex(H[..., 0], H[..., 1])

            Hnorm, _ = tf.linalg.normalize(H, axis=1)

            csim = tf.math.real(
                complex_abs(
                    tf.reduce_sum(
                        Hnorm[:, None, :, :] * tf.math.conj(self.X_bar_[:, :, :, None]),
                        axis=2,
                    )
                )
            )  # (n_freq, n_frame, n_src)

            mask = tf.nn.softmax(-beta * csim, axis=-1)
            mask, _ = palign(
                mask, max_iter=max_iter, tol=perm_tol, proc_limit=proc_limit
            )

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
                    :, self.trim_begin // 2 : self.trim_begin // 2 + self.n_samples, :
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
        *args,
        **kwargs,
    ):
        """
        Convenience class for DSF via Lehmer Mean as implemented in

            K. Watcharasupat, A. H. T. Nguyen, C. -H. Ooi and A. W. H. Khong,
            "Directional Sparse Filtering Using Weighted Lehmer Mean for Blind
            Separation of Unbalanced Speech Mixtures," ICASSP 2021 - 2021 IEEE
            International Conference on Acoustics, Speech and Signal Processing
            (ICASSP), 2021, pp. 4485-4489, doi: 10.1109/ICASSP39728.2021.9414336.

        Additional Parameters
        ---------------------
        r : float, optional
            Exponent for Lehmer mean, by default 0.5
        alpha : float, optional
            Weight smoothing parameter, by default 1.0

        See `dsf.DirectionalSparseFiltering` for other parameters
        """
        super().__init__(x, n_src, fs, *args, **kwargs)

        self.loss_function = LehmerMeanDSFLoss(
            self.n_freq,
            self.n_chan,
            self.n_src,
            r=r,
            alpha=alpha,
            n_frames=self.n_frames,
            distance_func=phase_invariant_cosine_squared_distance,
            time_pooling_func=tf.reduce_mean,
            freq_pooling_func=tf.reduce_sum,
            inline_decoupling=self.inline_decoupling,
            real_dtype=self.real_dtype,
            complex_dtype=self.complex_dtype,
            use_real_proxies=self.use_real_proxies,
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
        """
        Convenience class for DSF via Power Mean as implemented in

            A. H. T. Nguyen, V. G. Reju and A. W. H. Khong,
            "Directional Sparse Filtering for Blind Estimation of
            Under-Determined Complex-Valued Mixing Matrices," in
            IEEE Transactions on Signal Processing, vol. 68, pp.
            1990-2003, 2020, doi: 10.1109/TSP.2020.2979550.

            A. H. T. Nguyen, V. G. Reju, A. W. H. Khong and I. Y. Soon,
            "Learning complex-valued latent filters with absolute cosine
            similarity," 2017 IEEE International Conference on Acoustics,
            Speech and Signal Processing (ICASSP), 2017, pp. 2412-2416,
            doi: 10.1109/ICASSP.2017.7952589.

        Additional Parameters
        ---------------------
        p : float, optional
            Exponent for power mean, by default -0.5

        See `dsf.DirectionalSparseFiltering` for other parameters
        """
        super().__init__(x, n_src, fs, *args, **kwargs)

        self.loss_function = PowerMeanDSFLoss(
            self.n_freq,
            self.n_chan,
            self.n_src,
            p=p,
            distance_func=phase_invariant_cosine_squared_distance,
            time_pooling_func=tf.reduce_mean,
            freq_pooling_func=tf.reduce_sum,
            inline_decoupling=self.inline_decoupling,
            real_dtype=self.real_dtype,
            complex_dtype=self.complex_dtype,
            use_real_proxies=self.use_real_proxies,
        )
