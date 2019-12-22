from keras import backend as K


def differentiable_argmax(nr_of_categories, beta=1e10, dtype='float32'):
    y_range = K.arange(0, nr_of_categories, dtype=dtype)
    return lambda y: K.sum(K.softmax(y * beta) * y_range, axis=-1)


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
    argmax = differentiable_argmax(nr_of_categories, beta, dtype=dtype)
    return lambda y_true, y_pred: alpha * (argmax(y_pred) - argmax(y_true)) ** 2
