import numpy as np


def kalman_transit_covariance(S, A, R):
    """
    :param S: Current covariance matrix
    :param A: Either transition matrix or jacobian matrix
    :param R: Current noise covariance matrix
    """
    state_size = S.shape[0]
    assert S.shape == (state_size, state_size)
    assert A.shape == (state_size, state_size)
    assert R.shape == (state_size, state_size)
    new_S = np.dot(np.dot(A, S), A.T) + R
    return new_S


def kalman_process_observation(mu, S, observation, C, Q):
    """
    Performs processing of an observation coming from the model: z = C * x + noise
    :param mu: Current mean
    :param S: Current covariance matrix
    :param observation: Vector z
    :param C: Observation matrix
    :param Q: Noise covariance matrix (with zero mean)
    """
    state_size = mu.shape[0]
    observation_size = observation.shape[0]
    assert S.shape == (state_size, state_size)
    assert observation_size == C.shape[0]
    assert observation_size == Q.shape[0]
    H = np.linalg.inv(np.dot(np.dot(C, S), C.T) + Q)
    K = np.dot(np.dot(S, C.T), H)
    new_mu = mu + np.dot(K, observation - np.dot(C, mu))
    new_S = np.dot(np.eye(state_size) - np.dot(K, C), S)
    # Избавляемся от маленьких чисел. Из-за них могут быть мнимые числа в собственных значениях
    new_S[np.abs(new_S) < 1e-16] = 0
    return new_mu, new_S
