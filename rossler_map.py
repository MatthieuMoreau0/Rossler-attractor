import numpy as np
from scipy.integrate import solve_ivp


class RosslerMap:
    """
    Rossler attractor
    With a=0.2, b=0.2, and c=5.7
    """

    def __init__(_, a=0.2, b=0.2, c=5.7, delta_t=1e-3):
        _.a, _.b, _.c = a, b, c
        _.delta_t = delta_t

    def v_eq(_, t=None, v=None):
        '''
        From a vector of position v for a particule, returns its speed.
        '''
        x, y, z = v[0], v[1], v[2]
        dot_x = -y - z
        dot_y = x + _.a*y
        dot_z = _.b + z*(x-_.c)
        return np.array([dot_x, dot_y, dot_z])

    def jacobian(_, v):
        '''
        Returns J such that dot_X = J*X (* is matrix multiplication)
        '''
        x, z = v[0], v[2]
        res = np.array([[       0,      -1,       -1],
                       [        1,     _.a,        0],
                       [        z,       0,   x-_.c]])
        return res

    def full_traj(_, nb_steps, init_pos):
        '''
        Generates a trajectory from init_pos (?)
        '''
        t = np.linspace(0, nb_steps * _.delta_t, nb_steps)
        f = solve_ivp(_.v_eq, [0, nb_steps * _.delta_t], init_pos, method='RK45', t_eval=t)
        positions = np.moveaxis(f.y, -1, 0)
        speeds = np.zeros(positions.shape)
        for i in range(nb_steps):
            speeds[i] = _.v_eq(v=positions[i])
        jacobians = np.zeros((positions.shape[0], 3,3))
        for i in range(positions.shape[0]):
            jacobians[i] = _.jacobian(positions[i]) * _.delta_t + np.eye(3) # Converting to discrete jacobian
        return positions,speeds,jacobians,t
    
    def equilibrium(_):
        '''
        Returns the equilibrium position of the system (closed form)
        '''
        x0 = (_.c-np.sqrt(_.c**2-4*_.a*_.b))/2
        y0 = (-_.c+np.sqrt(_.c**2-4*_.a*_.b))/(2*_.a)
        z0 = (_.c-np.sqrt(_.c**2-4*_.a*_.b))/(2*_.a)
        return np.array([x0,y0,z0])
