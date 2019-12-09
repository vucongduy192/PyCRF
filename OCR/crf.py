import numpy as np

def node_potentials(features, state_params):
    """
    Example np.dot
        features is word, each element is char
        features = np.array([
          [1, 2, 3, 4, 1, 2, 3, 4],
          [5, 6, 7, 8, 5, 6, 7, 8]
        ])

        state_params = np.array(
          [[1, 2, 3, 4, 5, 6, 7, 8], [5, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]
        )
        each feature (each char) * state_params
        np.dot(features, np.transpose(state_params))
        result [[100 104] [244 264]]
    :param features:
    :param state_params:
    :return: a (w, k) numpy array of node potentials,
    where w is the word length and
          k is the size of the alphabet.
    """
    return np.dot(features, np.transpose(state_params))


def beliefs(theta, features):
    pass

def likelihood(theta, data, alphabet):
    if len(theta) != 2:
        k = len(alphabet)      # number of possible character labels
        n = len(data[0][1][0]) # length of feature vector
        mid = k * n

        state_params = np.reshape(theta[:mid], (k, n))
        trans_params = np.reshape(theta[mid:], (k, k))
        theta = [state_params, trans_params]

    p = []
    for label, features in data:
        beta = beliefs(theta, features)

def likelihood_prime():
    pass