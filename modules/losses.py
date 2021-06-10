from modules.utils import complex_abs
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


def phase_invariant_cosine_squared_distance(u, v, axis, proxy=False):

    if not proxy:
        return 1.0 - tf.clip_by_value(
            tf.math.real(
                tf.reduce_sum(tf.math.conj(u) * v, axis=axis)
                * tf.reduce_sum(u * tf.math.conj(v), axis=axis)
            ),
            0.0,
            1.0,
        )
    else:

        # u.shape = (n_freq, 1, n_chan, n_src, 2)
        # v.shape = (n_freq, n_frame, n_chan, 1, 2)

        # (a - bj) * (c + dj)
        #   = (a*c + b*d) + j(a*d - b*c)

        a = u[..., 0]
        b = u[..., 1]
        c = v[..., 0]
        d = v[..., 1]

        re = tf.reduce_sum(a * c + b * d, axis=axis)
        im = tf.reduce_sum(a * d - b * c, axis=axis)

        dot = tf.square(re) + tf.square(im)

        return 1.0 - tf.clip_by_value(dot, 0.0, 1.0)


class PowerMean(tfk.Model):
    def __init__(self, p=-0.5, *args, **kwargs):
        super().__init__()

        self.p = p

    def call(self, dist, axis=-1, eps=1e-6):

        dist_pow = tf.pow(tf.maximum(dist, eps), self.p)
        dist_pow_mean = tf.where(
            tf.reduce_min(dist, axis=axis) < eps,
            0.0,
            tf.reduce_mean(dist_pow, axis=axis),
        )
        dist_pm = tf.where(
            dist_pow_mean == 0.0, 0.0, tf.pow(dist_pow_mean, 1.0 / self.p)
        )

        return dist_pm


class LehmerMean(tfk.Model):
    def __init__(
        self,
        n_freq,
        n_src,
        r=0.5,
        alpha=1,
        w_init=1.0,
        real_dtype=tf.float32,
        complex_dtype=tf.complex64,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        assert alpha >= 0.0

        init_w = w_init * tf.ones(shape=(n_freq, n_src), dtype=real_dtype)
        self.w = tf.Variable(initial_value=init_w, trainable=True, dtype=real_dtype)

        self.r = r
        self.alpha = alpha

    def call(self, dist, axis=-1, eps=1e-12):

        if axis != -1:
            raise NotImplementedError

        w = tf.maximum(self.w + self.alpha, self.alpha)

        dr1 = tf.reduce_mean(
            tf.pow(tf.maximum(dist, eps), self.r - 1) * w[:, None, :], axis=-1
        )
        dr = tf.reduce_mean(
            tf.pow(tf.maximum(dist, eps), self.r) * w[:, None, :], axis=-1
        )

        return dr / dr1


class DSFLoss(tfk.Model):
    def __init__(
        self,
        n_freq,
        n_chan,
        n_src,
        src_pooling_func,
        src_func_kwargs={},
        distance_func=phase_invariant_cosine_squared_distance,
        time_pooling_func=tf.reduce_mean,
        freq_pooling_func=tf.reduce_mean,
        inline_decoupling=True,
        real_dtype=tf.float32,
        complex_dtype=tf.complex64,
        use_real_proxies=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_freq = n_freq
        self.n_src = n_src
        self.n_chan = n_chan

        self.src_pooling_func = src_pooling_func(
            n_freq=n_freq, n_src=n_src, **src_func_kwargs
        )

        self.distance_func = distance_func
        self.time_pooling_func = time_pooling_func
        self.freq_pooling_func = freq_pooling_func

        self.real_dtype = real_dtype
        self.complex_dtype = complex_dtype

        self.use_real_proxies = use_real_proxies
        # if self.use_real_proxies:
        #     assert (
        #         not inline_decoupling
        #     ), "inline decoupling is not available in real proxies mode"
        self.inline_decoupling = inline_decoupling

        if not self.use_real_proxies:
            init_mixing_matrix = tf.complex(
                tf.random.normal(
                    shape=(self.n_freq, self.n_chan, self.n_src),
                    dtype=self.real_dtype,
                ),
                tf.random.normal(
                    shape=(self.n_freq, self.n_chan, self.n_src),
                    dtype=self.real_dtype,
                ),
            )

            self.H = tf.Variable(initial_value=init_mixing_matrix, trainable=True)

        else:
            init_mixing_matrix = tf.random.normal(
                shape=(self.n_freq, self.n_chan, self.n_src, 2),
                dtype=self.real_dtype,
            )

            self.H = tf.Variable(initial_value=init_mixing_matrix, trainable=True)

        self.inline_decoupling_op()

    def inline_projection_op(self):

        H = self.H

        if self.use_real_proxies:
            H = tf.complex(H[..., 0], H[..., 1])

        _, u, v = tf.linalg.svd(H, full_matrices=False)
        H = tf.matmul(u, v, adjoint_b=True)
        # H = tf.matmul(tf.linalg.sqrtm(tf.matmul(H, H, adjoint_b=True)), H)

        if self.use_real_proxies:
            H = tf.stack([tf.math.real(H), tf.math.imag(H)], axis=-1)

        self.H = tf.Variable(H, trainable=True)

    def inline_decoupling_op(self):
        self.inline_projection_op()

    def proxy_normalize(self, H):

        norm = tf.sqrt(tf.reduce_sum(tf.square(self.H), axis=[-1, 1], keepdims=True))

        Hnorm = H / norm
        return Hnorm

    def call(self, Xbar):

        # Xbar.shape = (n_freq, n_frames, n_chan)
        # H.shape = (n_freq, n_chan, n_src)

        if self.inline_decoupling:
            self.inline_decoupling_op()

        H = self.H

        if not self.use_real_proxies:
            Hnorm, _ = tf.linalg.normalize(H, axis=1)
            srcframe_dist = self.distance_func(
                Hnorm[:, None, :, :], Xbar[:, :, :, None], axis=2
            )  # (n_freq, n_frames, n_src)
        else:
            Hnorm = self.proxy_normalize(H)
            srcframe_dist = self.distance_func(
                Hnorm[:, None, :, :, :], Xbar[:, :, :, None, :], axis=2, proxy=True
            )  # (n_freq, n_frames, n_src)

        frame_dist = self.src_pooling_func(
            srcframe_dist, axis=-1
        )  # (n_freq, n_frames, )
        freq_loss = self.time_pooling_func(frame_dist, axis=-1)  # (n_freq, )
        loss = self.freq_pooling_func(freq_loss, axis=-1)
        return loss


class PowerMeanDSFLoss(DSFLoss):
    def __init__(
        self,
        n_freq,
        n_chan,
        n_src,
        p=-0.5,
        distance_func=phase_invariant_cosine_squared_distance,
        time_pooling_func=tf.reduce_mean,
        freq_pooling_func=tf.reduce_sum,
        inline_decoupling=True,
        use_real_proxies=False,
        *args,
        **kwargs
    ):

        super().__init__(
            n_freq,
            n_chan,
            n_src,
            src_pooling_func=PowerMean,
            src_func_kwargs={"p": p, **kwargs},
            distance_func=distance_func,
            time_pooling_func=time_pooling_func,
            freq_pooling_func=freq_pooling_func,
            inline_decoupling=inline_decoupling,
            use_real_proxies=use_real_proxies,
            *args,
            **kwargs
        )


class LehmerMeanDSFLoss(DSFLoss):
    def __init__(
        self,
        n_freq,
        n_chan,
        n_src,
        r=0.5,
        alpha=1.0,
        n_frames=None,
        distance_func=phase_invariant_cosine_squared_distance,
        time_pooling_func=tf.reduce_mean,
        freq_pooling_func=tf.reduce_mean,
        inline_decoupling=True,
        use_real_proxies=False,
        *args,
        **kwargs
    ):
        super().__init__(
            n_freq,
            n_chan,
            n_src,
            src_pooling_func=LehmerMean,
            src_func_kwargs={
                "r": r,
                "alpha": alpha,
                "w_init": n_frames + (n_src - 1) * alpha,
                **kwargs,
            },
            distance_func=distance_func,
            time_pooling_func=time_pooling_func,
            freq_pooling_func=freq_pooling_func,
            inline_decoupling=inline_decoupling,
            use_real_proxies=use_real_proxies,
            *args,
            **kwargs
        )
