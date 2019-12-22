from collections import Callable

from keras import backend as K
from keras.layers import Layer


class CurveFit(Layer):

    def __init__(self, parameters: int, function: Callable, **kwargs):
        super().__init__(**kwargs)
        self.parameters = parameters
        self.function = function

    def build(self, input_shape):
        """
        parameters: pass the number of parameters of the function you try to fit
        function: pass the function you want to fit i.e. `lambda x, args: x * sum(args)`
        """

        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.parameters, ),
                                      initializer='uniform',
                                      trainable=True)

        # Be sure to call this at the end
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # here we gonna invoke out function and return the result.
        # the loss function will do whatever is needed to fit this function as good as possible
        return self.function(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape


class LinearRegressionLayer(CurveFit):

    def __init__(self):
        super().__init__(2, LinearRegressionLayer.fit)

    @staticmethod
    def fit(x, args):
        # y = k * x + d
        return args[0] * K.arange(0, x.shape[1], 1, dtype=x.dtype) + args[1]


class LPPLLayer(CurveFit):

    def __init__(self):
        super().__init__(-1, LPPLLayer.lppl)

    @staticmethod
    def lppl(x, args):
        is_bubble = True
        N = K.constant(x.shape[-1], dtype=x.dtype)
        t = K.arange(0, N, 1, dtype=x.dtype)
        tc = args[0]
        m = args[1]
        w = args[2]

        # check if OLS regression slope is >= 0 -> bubble detection or < 0 -> anti-bubble detection
        # and calculate the lppl with the given parameters
        dt = (tc - t) if is_bubble else (t - tc)
        a, b, c1, c2 = LPPLLayer.matrix_equation(x, dt, m, w, N)
        return a + K.pow(dt, m) * (b + ((c1 * K.cos(w * K.log(dt))) + (c2 * K.sin(w * K.log(dt)))))

    @staticmethod
    def matrix_equation(x, dt, m, w, N):
        dtEm = K.pow(dt, m)
        logdt = K.log(dt)

        fi = dtEm
        gi = dtEm * K.cos(w * logdt)
        hi = dtEm * K.sin(w * logdt)

        fi_pow_2 = K.sum(fi * fi)
        gi_pow_2 = K.sum(gi * gi)
        hi_pow_2 = K.sum(hi * hi)

        figi = K.sum(fi * gi)
        fihi = K.sum(fi * hi)
        gihi = K.sum(gi * hi)

        yi = K.log(x)
        yifi = K.sum(yi * fi)
        yigi = K.sum(yi * gi)
        yihi = K.sum(yi * hi)

        fi = K.sum(fi)
        gi = K.sum(gi)
        hi = K.sum(hi)
        yi = K.sum(yi)

        A = K.stack([
            K.stack([N, fi, gi, hi], axis=1),
            K.stack([fi, fi_pow_2, figi, fihi], axis=1),
            K.stack([gi, figi, gi_pow_2, gihi], axis=1),
            K.stack([hi, fihi, gihi, hi_pow_2], axis=1)
        ])

        b = K.stack([yi, yifi, yigi, yihi])

        # do a classic x = (A'A)⁻¹A' b
        return K.inverse(K.transpose(A) @ A) @ K.transpose(A) @ b
