import numpy as np

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#
# License: MIT License


def get_random_projections(n_projections, d, seed=None):
    r"""
    Generates n_projections samples from the uniform on the unit sphere of dimension d-1: :math:`\mathcal{U}(\mathcal{S}^{d-1})`
    Parameters
    ----------
    n_projections : int
        number of samples requested
    d : int
        dimension of the space
    seed: int or RandomState, optional
        Seed used for numpy random number generator
    Returns
    -------
    out: ndarray, shape (n_projections, d)
        The uniform unit vectors on the sphere
    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_random_projections(n_projections, d)
    >>> np.allclose(np.sum(np.square(projs), 1), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True
    """

    if not isinstance(seed, np.random.RandomState):
        random_state = np.random.RandomState(seed)
    else:
        random_state = seed

    projections = random_state.normal(0., 1., [n_projections, d])
    norm = np.linalg.norm(projections, ord=2, axis=1, keepdims=True)
    projections = projections / norm
    return projections