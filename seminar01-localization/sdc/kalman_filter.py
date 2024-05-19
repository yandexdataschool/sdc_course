import typing as T
import numpy as np


def kalman_transit_covariance(
        S: np.ndarray, A: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    :param S: Current covariance matrix
    :param A: Either transition matrix or jacobian matrix
    :param R: Current noise covariance matrix

    :returns: Returns covariance matrix after transition
    :rtype: np.ndarray
    """
    state_size = S.shape[0]
    assert S.shape == (state_size, state_size),\
        f'S has shape {S.shape} but expected {(state_size, state_size)}'
    assert A.shape == (state_size, state_size),\
        f'A has shape {A.shape} but expected {(state_size, state_size)}'
    assert R.shape == (state_size, state_size),\
        f'R has shape {R.shape} but expected {(state_size, state_size)}'
    assert A.dtype in [np.float32, np.float64]
    assert S.dtype == A.dtype, f'Mixing floating point values precisions: {S.dtype} and {A.dtype}'
    assert S.dtype == R.dtype, f'Mixing floating point values precisions: {S.dtype} and {R.dtype}'
    next_S = np.dot(np.dot(A, S), A.T) + R
    return next_S


def kalman_process_observation(
        mu, S, observation, C, Q) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Performs processing of an observation coming from the model: z = C * x + noise
    :param mu: Current mean
    :param S: Current covariance matrix
    :param observation: Vector z
    :param C: Observation matrix
    :param Q: Noise covariance matrix (with zero mean)

    :returns: (mean, cov)
    """
    state_size = mu.shape[0]
    observation_size = observation.shape[0]
    assert S.shape == (state_size, state_size)
    assert observation_size == C.shape[0]
    assert observation_size == Q.shape[0]
    H = np.linalg.inv(np.dot(np.dot(C, S), C.T) + Q)
    K = np.dot(np.dot(S, C.T), H)
    next_mu = mu + np.dot(K, observation - np.dot(C, mu))
    next_S = np.dot(np.eye(state_size) - np.dot(K, C), S)
    # Избавляемся от маленьких чисел. Из-за них могут быть мнимые числа в собственных значениях
    next_S[np.abs(next_S) < 1e-16] = 0
    return next_mu, next_S
