import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from scipy.optimize import check_grad


def load_movielens(filename, minidata=False):
    """
    Cette fonction lit le fichier filename de la base de donnees
    Movielens, par exemple 
    filename = '~/datasets/ml-100k/u.data'
    Elle retourne 
    R : une matrice utilisateur-item contenant les scores
    mask : une matrice valant 1 si il y a un score et 0 sinon
    """

    data = np.loadtxt(filename, dtype=int)

    R = sparse.coo_matrix((data[:, 2], (data[:, 0]-1, data[:, 1]-1)),
                          dtype=float)
    R = R.toarray()  # not optimized for big data

    # code la fonction 1_K
    mask = sparse.coo_matrix((np.ones(data[:, 2].shape),
                              (data[:, 0]-1, data[:, 1]-1)), dtype=bool)
    mask = mask.toarray()  # not optimized for big data

    if minidata is True:
        R = R[0:100, 0:200].copy()
        mask = mask[0:100, 0:200].copy()

    return R, mask


def objective(P, Q0, R, mask, rho):
    """
    La fonction objectif du probleme simplifie.
    Prend en entree 
    P : la variable matricielle de taille C x I
    Q0 : une matrice de taille U x C
    R : une matrice de taille U x I
    mask : une matrice 0-1 de taille U x I
    rho : un reel positif ou nul

    Sorties :
    val : la valeur de la fonction
    grad_P : le gradient par rapport a P
    """

    tmp = (R - Q0.dot(P)) * mask

    val = np.sum(tmp ** 2)/2. + rho/2. * (np.sum(Q0 ** 2) + np.sum(P ** 2))

    grad_P = -Q0.transpose().dot(tmp) + rho*P

    return val, grad_P


def total_objective(P, Q, R, mask, rho):
    """
    La fonction objectif du probleme complet.
    Prend en entree 
    P : la variable matricielle de taille C x I
    Q : la variable matricielle de taille U x C
    R : une matrice de taille U x I
    mask : une matrice 0-1 de taille U x I
    rho : un reel positif ou nul

    Sorties :
    val : la valeur de la fonction
    grad_P : le gradient par rapport a P
    grad_Q : le gradient par rapport a Q
    """

    tmp = (R - Q.dot(P)) * mask

    val = np.sum(tmp ** 2)/2. + rho/2. * (np.sum(Q ** 2) + np.sum(P ** 2))

    grad_P = 0  # todo

    grad_Q = 0  # todo

    return val, grad_P, grad_Q


def total_objective_vectorized(PQvec, R, mask, rho):
    """
    Vectorisation de la fonction precedente de maniere a ne pas
    recoder la fonction gradient
    """

    # reconstruction de P et Q
    n_items = R.shape[1]
    n_users = R.shape[0]
    F = PQvec.shape[0] / (n_items + n_users)
    Pvec = PQvec[0:n_items*F]
    Qvec = PQvec[n_items*F:]
    P = np.np.reshape(Pvec, (F, n_items))
    Q = np.np.reshape(Qvec, (n_users, F))

    val, grad_P, grad_Q = total_objective(P, Q, R, mask, rho)
    return val, np.concatenate([grad_P.np.ravel(), grad_Q.np.ravel()])


def gradient(g, P0, gamma, epsilon):
    """
    Minimise g par la méthode du gradient a pas constant.
    Prend en entree
    g : fonction a minimiser, Pk -> g(Pk), grad_g(Pk)
    P0 : point de depart
    gamma : pas constant
    epsilon : critere d'arret, norme de Frobenius du gradient de g en Pk
              inferieur ou egal a epsilon
    
    Sorties :
    Pk : minimiseur
    """
    
    grad_g = g(P0)[1]
    while np.sum(grad_g**2) > epsilon:
        Pk = Pk - gamma*grad_G
        grad_G = g(Pk)[1]
    
    return Pk


def gradient_linear(g, P0, epsilon):
    """
    Minimise g par la methode du gradient avec recherche lineaire.
    Prend en entree
    g : fonction a minimiser, Pk -> g(Pk), grad_g(Pk)
    P0 : point de depart
    epsilon : critere d'arret, norme de Frobenius du gradient de g en Pk inferieur ou egal a epsilon

    Sorties :
    Pk : minimiseur
    """

    a, b, beta = 1/2, 1/2, 10**(-4) # Armijo's linear search parameters

    grad_g = g(P0)[1]
    while np.sum(grad_g**2) > epsilon:
        # Armijo's linear search
        l = 0
        while g(Pk-b*(aggl)*grad_g) > g(Pk) - beta*b*(a**l)*np.sum(grad_g**2):
            l = l + 1

        Pk = Pk - b*(a**l)*grad_g
        b = 2*b*(a**l)
        grad_g = g(Pk)[1]

    return Pk


def gradient_conjugate(g, P0, epsilon):
    """
    Minimise g par la methode du gradient conjugue : Fletcher et Reeves.
    Prend en entree
    g : fonction a minimiser, Pk -> g(Pk), grad_g(Pk)
    P0 : point de depart
    
    Sorties :
    Pk : minimiseur
    """
    
    grad_g = g(P0)[1]
    d = -grad_g
    while np.sum(grad_g**2) > epsilon:
        s = 0#TODO sk min g(Pk + s*d)
        Pk = Pk + s*d
        prev_grad = grad_g
        grad_g = g(Pk)[1]
        b = np.sum(grad_g**2) / np.sum(prev_grad**2)
        d = -grad_g + b*d

        return Pk


def test():
    R, mask = load_movielens('ml-100k/u.data', True)
    Q0, s, P0 = svds(R, k=4)
    rho = 0.3
    print(check_grad(
          lambda p: objective(np.reshape(p, np.shape(P0)), Q0, R, mask, rho)[0],
          lambda p: np.ravel(
          objective(np.reshape(p, np.shape(P0)), Q0, R, mask, rho)[1]),
          np.ravel(P0)))


