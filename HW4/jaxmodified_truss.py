import jax.numpy as jnp
from math import sin, cos, sqrt, pi

def truss(A):
    """Computes mass and stress for the 10-bar truss problem

    Parameters
    ----------
    A : ndarray of length nbar
        cross-sectional areas of each bar
        see image in book for number order if needed

    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length nbar
        stress in each bar

    """

    # --- specific truss setup -----
    P = 1e5  # applied loads
    Ls = 360.0  # length of sides
    Ld = sqrt(360**2 * 2)  # length of diagonals

    start = [5, 3, 6, 4, 4, 2, 5, 6, 3, 4]
    finish = [3, 1, 4, 2, 3, 1, 4, 3, 2, 1]
    phi = jnp.array([0, 0, 0, 0, 90, 90, -45, 45, -45, 45])*pi/180
    L = jnp.array([Ls, Ls, Ls, Ls, Ls, Ls, Ld, Ld, Ld, Ld])

    nbar = len(A)  # number of bars
    E = 1e7*jnp.ones(nbar)  # modulus of elasticity
    rho = 0.1*jnp.ones(nbar)  # material density

    Fx = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Fy = jnp.array([0.0, -P, 0.0, -P, 0.0, 0.0])
    rigid = [False, False, False, False, True, True]
    # ------------------

    n = len(Fx)  # number of nodes
    DOF = 2  # number of degrees of freedom

    # mass
    mass = jnp.sum(rho*A*L)

    # stiffness and stress matrices
    K = jnp.zeros((DOF*n, DOF*n))
    S = jnp.zeros((nbar, DOF*n))

    for i in range(nbar):  # loop through each bar

        # compute submatrix for each element
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])

        # insert submatrix into global matrix
        idx = node2idx([start[i], finish[i]], DOF)  # pass in the starting and ending node number for this element
        K = K.at[jnp.ix_(idx, idx)].add(Ksub) #+= Ksub
        S = S.at[i, idx].set(Ssub) #= Ssub

    # applied loads
    F = jnp.zeros((n*DOF, 1))

    for i in range(n):
        idx = node2idx([i+1], DOF)  # add 1 b.c. made indexing 1-based for convenience
        F = F.at[idx[0]].set(Fx[i]) #= Fx[i]
        F = F.at[idx[1]].set(Fy[i]) #= Fy[i]


    # boundary condition
    idx = [i+1 for i, val in enumerate(rigid) if val] # add 1 b.c. made indexing 1-based for convenience
    remove = node2idx(idx, DOF)

    K = jnp.delete(K, remove, axis=0)
    K = jnp.delete(K, remove, axis=1)
    F = jnp.delete(F, remove, axis=0)
    S = jnp.delete(S, remove, axis=1)

    # solve for deflections
    d = jnp.linalg.solve(K, F)

    # compute stress
    stress = jnp.dot(S, d).reshape(nbar)
    return mass, stress



def bar(E, A, L, phi):
    """Computes the stiffness and stress matrix for one element

    Parameters
    ----------
    E : float
        modulus of elasticity
    A : float
        cross-sectional area
    L : float
        length of element
    phi : float
        orientation of element

    Outputs
    -------
    K : 4 x 4 ndarray
        stiffness matrix
    S : 1 x 4 ndarray
        stress matrix

    """

    # rename
    c = cos(phi)
    s = sin(phi)

    # stiffness matrix
    k0 = jnp.array([[c**2, c*s], [c*s, s**2]])
    k1 = jnp.hstack([k0, -k0])
    K = E*A/L*jnp.vstack([k1, -k1])

    # stress matrix
    S = E/L*jnp.array([-c, -s, c, s])

    return K, S



def node2idx(node, DOF):
    """Computes the appropriate indices in the global matrix for
    the corresponding node numbers.  You pass in the number of the node
    (either as a scalar or an array of locations), and the degrees of
    freedom per node and it returns the corresponding indices in
    the global matrices

    """

    idx = jnp.array([], dtype=int)

    for i in range(len(node)):

        n = node[i]
        start = DOF*(n-1)
        finish = DOF*n

        idx = jnp.concatenate((idx, jnp.arange(start, finish, dtype=int)))

    return idx