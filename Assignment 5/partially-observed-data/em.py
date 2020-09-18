import numpy as np

"""
Expectation-Maximization for a 3 node graph, formed as a v-structure. 
X -> Z <- Y
All 3 variables are binary, {0, 1}, and each has a table (numpy array) that represents the probabilities (the parameters theta).
Since X and Y have no parents, they only need 2 parameters, [p(X=0), p(Y=1)].
Z however require 8 parameters, since we need one for each combination of (X, Y, Z), i.e. 2*2*2=8.
These are represented in a 3d array pz=[[[p(z=0|x=0, y=0) p(z=1|x=0, y=0)], [...]], [[...], [...]]].
The ordering is such that to acces p(z|x, y), you take element pz[x, y, z].
Thus, each of p[x, y, :] should sum to 1 (p(z=0|x, y) + p(z=1|x, y) = 1). 
Marginals can be computed by summing over x or y.
There are also help functions to generate data and print parameters.
"""

def expectation_maximization(x, y, z, e_step, m_step, num_iter):
    """ Performs Expectation-Maximization algorithm and checks correctness.
    It only works for a V-structure, i.e. X -> Z <- Y
    It does not allow missing z, but both or either of x and y can be missing.
    Args:
        x, y, z (np.array): Input data where a None in x or y is interpreted as missing data. 
        e_step (function): A function that takes current parameter estimates qx, qy, qz and data x, y, z 
            and outputs expected sufficient statistics Mx, My, Mz.
        m_step (function): A function that takes current expected sufficient statistics
            and outputs new parameter estimates qx, qy, qz.
        num_iter (int): The number of iterations to run EM.
    Return:
        qx, qy, qx (np.array): Final parameter estimates after num_iter iterations of e_step and m_step.
    """
    n = len(x)
    qx, qy, qz = initialize_parameters()
    for i in range(num_iter):
        Mx, My, Mz = e_step(qx, qy, qz, x, y, z)
        # Assert valid statatistics
        assert np.isclose(np.sum(Mx), n), f"Mx = {Mx} should sum to {n}"
        assert np.isclose(np.sum(My), n), f"My = {My} should sum to {n}"
        assert np.isclose(np.sum(Mz[0]), Mx[0]), f"Mz[0] = {Mz[0]} should sum to Mx[0] = {Mx[0]}"
        assert np.isclose(np.sum(Mz[1]), Mx[1]), f"Mz[1] = {Mz[1]} should sum to Mx[1] = {Mx[1]}"
        assert np.isclose(np.sum(Mz[:, 0]), My[0]), f"Mz[:, 0] = {Mz[:, 0]} should sum to My[0] = {My[0]}"
        assert np.isclose(np.sum(Mz[:, 1]), My[1]), f"Mz[:, 1] = {Mz[:, 1]} should sum to My[1] = {My[1]}"

        qx, qy, qz = m_step(Mx, My, Mz)
        # Assert valid parameters
        assert (qx >= 0).all(), f"qx = {qx} need to be non-negative"
        assert (qy >= 0).all(), f"qy = {qy} need to be non-negative"
        assert (qz >= 0).all(), f"qz = {qz} need to be non-negative"
        assert np.isclose(np.sum(qx), 1), f"qx = {qx} need to sum to one"
        assert np.isclose(np.sum(qy), 1), f"qy = {qy} need to sum to one"
        assert np.isclose(np.sum(qz, axis=2), 1).all(), f"Each row of qz = {qz} needs to sum to one"
    return qx, qy, qz


def generate_data(px, py, pz, n, *, partially_observed=False, never_coobserved=False):
    """ Generates data given table-CPDs of a V-stucture X -> Z <- Y
    It can generate complete or partially observed data
    Args:
        px, py, px (np.array): Parameters to generate data with.
        n (int): Number of data points.
        partially_observed (bool): If True, half of x and y will be missing (set to None)
        never_coobserved (bool): If True, y is missing if and only if x is observed, 
            so that no data points contains both x and y. 
            If False, y is missing independently of whether x is missing.
            Has no effect if partially_observed is False.
    Return:
        x, y, z (np.array): Generated data where a None in x or y is interpreted as missing data. 
    """
    assert px.shape == (2,), f"px = {px} should have shape (2,)"
    assert py.shape == (2,), f"py = {py} should have shape (2,)"
    assert pz.shape == (2, 2, 2), f"pz = {pz} should have shape (2, 2, 2)"
    x = np.argmax(np.random.multinomial(1, px, n), axis=1)
    y = np.argmax(np.random.multinomial(1, py, n), axis=1)
    z = np.argmax([np.random.multinomial(1, p) for p in pz[x, y]], axis=1)
    if partially_observed:
        x = x.astype(object)
        y = y.astype(object)
        x[np.unique(np.random.choice(n, int(n/2)))] = None
        if never_coobserved:
            y[np.not_equal(x, None)] = None
        else:
            y[np.unique(np.random.choice(n, int(n/2)))] = None
    return x, y, z


def initialize_parameters(random=False):
    """ Initializes parameters for the EM algorithm
    Args:
        random (bool): If True, the parameters are set to random values (in range [0, 1] that sum to 1).
            If False, all probabilities are 0.5 (binary variables).
    Returns:
        qx, qy, qx (np.array): Initial parameters.
    """
    if random:
        qx = np.random.rand(2)
        qy = np.random.rand(2)
        qz = np.random.rand(2, 2, 2) 
    else:
        qx = np.ones(2)
        qy = np.ones(2)
        qz = np.ones((2,2,2))
    qx = qx / np.sum(qx)
    qy = qy / np.sum(qy)
    qz = qz / np.sum(qz, axis=2, keepdims=1)
    return qx, qy, qz


def print_tables(px, py, pz):
    """ Prints probability tables in a nice way. 
    Args:
        px, py, pz (np.array): Parameters, true or learnt.
    """
    assert px.shape == (2,), f"px = {px} should have shape (2,)"
    assert py.shape == (2,), f"py = {py} should have shape (2,)"
    assert pz.shape == (2, 2, 2), f"pz = {pz} should have shape (2, 2, 2)"
    print(f"p(x) = {px}")
    print(f"p(y) = {py}")
    print(f"p(z|x=0, y=0) = {pz[0, 0]}")
    print(f"p(z|x=0, y=1) = {pz[0, 1]}")
    print(f"p(z|x=1, y=0) = {pz[1, 0]}")
    print(f"p(z|x=1, y=1) = {pz[1, 1]}")


def print_marginals(px, py, pz):
    """ Prints marginal probabilities of z given x or y, i.e. one variable is summed out.
    Args:
        px, py, pz (np.array): Parameters, true or learnt.
    """
    assert px.shape == (2,), f"px = {px} should have shape (2,)"
    assert py.shape == (2,), f"py = {py} should have shape (2,)"
    assert pz.shape == (2, 2, 2), f"pz = {pz} should have shape (2, 2, 2)"
    print(f"p(z|x=0) = {pz[0, 0] * py[0] + pz[0, 1] * py[1]}")
    print(f"p(z|x=1) = {pz[1, 0] * py[0] + pz[1, 1] * py[1]}")
    print(f"p(z|y=0) = {pz[0, 0] * px[0] + pz[1, 0] * px[1]}")
    print(f"p(z|y=1) = {pz[0, 1] * px[0] + pz[1, 1] * px[1]}")
