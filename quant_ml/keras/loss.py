from keras import backend as _K
from keras.losses import categorical_crossentropy

from quant_ml.util import LazyInit


def differentiable_argmax(nr_of_categories, beta=1e10, dtype='float32'):
    # need to lazy initialized as it needs to be in the same keras session
    y_range = LazyInit(lambda: _K.arange(0, nr_of_categories, dtype=dtype))
    return lambda y: _K.sum(_K.softmax(y * beta) * y_range(), axis=-1)


def tailed_categorical_crossentropy(nr_of_categories, alpha=0.1, beta=1e10, dtype='float32'):
    """
    assuming that we have discretized something like returns where we have less observations in the tails.
    If we want to train a neural net to place returns into the expected bucket we want to penalize if the
    prediction is too close to the mean. we rather want to be pessimistic and force the predictor to
    encounter the tails.

    :param nr_of_categories: number of categories aka length of the one hot encoded vectors
    :param alpha: describes the steepness of the parabola
    :param beta: used for the differentiable_argmax
    :return: returns a keras loss function
    """

    # custom loss function is cross entropy with penalized tail errors
    def loss_function(y_true, y_pred):
        argmax = differentiable_argmax(nr_of_categories, beta, dtype=dtype)
        penalty = alpha * (argmax(y_pred) - argmax(y_true)) ** 2
        loss = categorical_crossentropy(y_true, y_pred)
        return loss + penalty

    return loss_function
