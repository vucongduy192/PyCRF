import argparse
import numpy as np

from OCR.dataset import read_data
from OCR.crf import node_potentials

from scipy.optimize import fmin_l_bfgs_b


def train(data, alphabet, maxiter, log):
    """
    Returns the learned [state_params, trans_params] list,
    where each parameter table is a numpy array.
    """
    # Initialize state and transition parameter tables with zeros
    state_params = np.ndarray.flatten(np.zeros((len(alphabet), len(data[0][1][0]))))
    trans_params = np.ndarray.flatten(np.zeros((len(alphabet), len(alphabet))))
    theta = np.concatenate([state_params, trans_params])

    print(theta.shape)
    k = len(alphabet)  # number of possible character labels
    n = len(data[0][1][0])  # length of feature vector
    mid = k * n

    state_params = np.reshape(theta[:mid], (k, n))
    trans_params = np.reshape(theta[mid:], (k, k))
    theta = [state_params, trans_params]

    for label, features in data:
        print(features)
        phi = node_potentials(features, state_params)
        print(phi)
        break
    cliques = [(node, None) for node in phi[:-2]] + [(phi[-2], phi[-1])]
    print(cliques)
    for n1, n2 in cliques:
        psi = trans_params + n1[:, np.newaxis]
        if n2 is not None:
            psi += n2
        print('---------------------')
        print(psi)

    # theta, fmin, _ = fmin_l_bfgs_b(likelihood, theta, fprime=likelihood_prime,
    #                                args=(data, alphabet), maxiter=maxiter, disp=log)


def main(train_pattern, model_dir, alphabet, maxiter, log):
    alphabet = list(alphabet)

    # Read training data
    data = read_data(train_pattern)
    print(data[0])

    if log > 0:
        print('Successfully read', len(data), 'data cases')

    # Train the model
    model = train(data, alphabet, maxiter, log)


if __name__ == '__main__':
    """ python3 ocr_train.py -train data/ -model model/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Regex pattern of training files', dest='train', required=True)
    parser.add_argument('-model', help='Directory of model files', dest='model', required=True)
    parser.add_argument('-alphabet', help='String of all possible character labels', dest='alphabet',
                        default='etainoshrd')
    parser.add_argument('-maxiter', help='Maximum iteration for L-BFGS optimization', dest='maxiter', default=1000,
                        type=int)
    parser.add_argument('-log', help='Print log to stdout if 1', dest='log', default=1, type=int)
    args = parser.parse_args()

    main(args.train, args.model, args.alphabet, args.maxiter, args.log)
