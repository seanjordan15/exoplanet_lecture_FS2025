import numpy as np
from PyAstronomy import pyasl
import matplotlib.pyplot as plt


class TwoBodyProblem:
    """ Class to calculate a two-body-problem in an interatial reference frame
    as defined for radial velocity exoplanet detections.

    Parameters
    ----------
    m_p : float
        The mass of the planet in Jupiter masses.
    m_s : float
        The mass of the star in Jupiter masses.
    period : float
        The period of the orbit in days.
    eccentricity : float
        The eccentricity of the orbit.
    inclination : float
        The inclination of the orbit in degrees.
    v0 : numpy.ndarray
        The velocity of the barycenter in m/s.
    t: numpy.ndarray
        The time in seconds.
    r_s: numpy.ndarray
        The position of the star in m.
    v_s: numpy.ndarray
        The velocity of the star in m/s.
    r_p: numpy.ndarray
        The position of the planet in m.
    v_p: numpy.ndarray
        The velocity of the planet in m/s.
    r_s_au: numpy.ndarray
        The position of the star in AU.
    r_p_au: numpy.ndarray
        The position of the planet in AU.
    barycenter_au: numpy.ndarray
        The position of the barycenter in AU.


    """
    def __init__(self,
                 m_p: float,
                 m_s: float,
                 period: float,
                 eccentricity: float,
                 inclination: float,
                 v0: np.ndarray = np.zeros(3)):
        """
        Initialize the class with the parameters of the two-body-problem.

        Parameters
        ----------
        m_p : float
            The mass of the planet in Jupiter masses.
        m_s : float
            The mass of the star in Jupiter masses.
        period : float
            The period of the orbit in days.
        eccentricity : float
            The eccentricity of the orbit.
        inclination : float
            The inclination of the orbit in degrees.
        v0 : numpy.ndarray
            The velocity of the barycenter in m/s.
        """
        self.m_p = m_p
        self.m_s = m_s
        self.period = period
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.v0 = v0

        self.t = None
        self.r_s = None
        self.v_s = None
        self.r_p = None
        self.v_p = None
        self.barycenter = None

        self.r_s_au = None
        self.r_p_au = None
        self.barycenter_au = None

    def get_orbit(self, t_min, t_max, num_points=10000):
        """
        Calculate the orbit of the two-body system.

        Parameters
        ----------
        t_min : float
            The start time of the integration in days.
        t_max : float
            The end time of the integration in days.
        num_points : int
            The number of points to evaluate the orbits for.

        Returns
        -------
        t : numpy.ndarray
            The time in seconds.
        r_s : numpy.ndarray
            The position of the star in m.
        v_s : numpy.ndarray
            The velocity of the star in m/s.
        r_p : numpy.ndarray
            The position of the planet in m.
        v_p : numpy.ndarray
            The velocity of the planet in m/s.
        """

        bo = pyasl.BinaryOrbit(m2m1=self.m_p / self.m_s,
                               mtot=(self.m_p + self.m_s)/1047.87,
                               per=self.period,
                               e=self.eccentricity,
                               tau=0.,
                               Omega=90.,
                               w=0.,
                               i=self.inclination-90.)

        ke1 = bo.getKeplerEllipse_primary()
        ke2 = bo.getKeplerEllipse_secondary()

        # Input time in seconds
        self.t = np.linspace(t_min * 24 * 60 * 60,
                             t_max * 24 * 60 * 60,
                             num=num_points)

        self.r_s, self.r_p = bo.xyzPos(self.t)
        self.v_s, self.v_p = bo.xyzVel(self.t)

        self.v_s += self.v0
        self.v_p += self.v0
        self.r_s += self.v0[np.newaxis, :] * self.t[:, np.newaxis]
        self.r_p += self.v0[np.newaxis, :] * self.t[:, np.newaxis]

        self.barycenter = ((self.m_s * self.r_s + self.m_p * self.r_p)
                            / (self.m_s + self.m_p))

        return self.t, self.r_s, self.v_s, self.r_p, self.v_p

    def get_rv(self):
        """Calculate the radial velocity of the star along the x-axis
        and the radial velocity of the planet along the x-axis.

        Returns
        -------
        t : numpy.ndarray
            The time in days.
        rv_s : numpy.ndarray
            The radial velocity of the star in m/s.
        rv_p : numpy.ndarray
            The radial velocity of the planet in m/s.
        """
        return self.t / 60 / 60 / 24, self.v_s[:, 0], self.v_p[:, 0]

    def plot_orbit(self):
        """
        Plot the orbit of the two-body system.
        """
        round_to = 10
        self.r_s_au = self.r_s / 1.496e11
        self.r_p_au = self.r_p / 1.496e11
        self.barycenter_au = self.barycenter / 1.496e11

        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(131, projection='3d')
        ax.plot(self.r_s_au[:, 0], self.r_s_au[:, 1], self.r_s_au[:, 2],
                label='star')
        ax.plot(self.r_p_au[:, 0], self.r_p_au[:, 1], self.r_p_au[:, 2],
                label='exoplanet')
        ax.plot(self.barycenter_au[:, 0],
                self.barycenter_au[:, 1],
                self.barycenter_au[:, 2],
                label='C/G')
        ax.legend()

        ax = fig.add_subplot(132, projection='3d')
        r_s_rel_cg = np.round(self.r_s_au - self.barycenter_au,
                              decimals=round_to)
        r_p_rel_cg = np.round(self.r_p_au - self.barycenter_au,
                              decimals=round_to)
        ax.plot(r_s_rel_cg[:, 0], r_s_rel_cg[:, 1], r_s_rel_cg[:, 2],
                label='star')
        ax.plot(r_p_rel_cg[:, 0], r_p_rel_cg[:, 1], r_p_rel_cg[:, 2],
                label='exoplanet')
        ax.plot(0, 0, 0, 'ro', label='C/G')
        ax.legend()

        ax = fig.add_subplot(133, projection='3d')
        r_p_rel_r_s = np.round(self.r_p_au - self.r_s_au,
                               decimals=round_to)
        cg_rel_r_s = np.round(self.barycenter_au - self.r_s_au,
                              decimals=round_to)
        ax.plot(r_p_rel_r_s[:, 0], r_p_rel_r_s[:, 1], r_p_rel_r_s[:, 2],
                label='exoplanet')
        ax.plot(cg_rel_r_s[:, 0], cg_rel_r_s[:, 1], cg_rel_r_s[:, 2],
                label='C/G')
        ax.plot(0, 0, 0, 'ro', label='star')
        ax.legend()

        plt.show()


# class TwoBodyProblem:
#     """Class to solve the two-body problem.
#     Adapted from
#     https://orbital-mechanics.space/the-n-body-problem/two-body-inertial-numerical-solution.html
#
#     Parameters
#     ----------
#     pos_p : numpy.ndarray
#         The initial position of the planet in km.
#     pos_s : numpy.ndarray
#         The initial position of the star in km.
#     vel_p : numpy.ndarray
#         The initial velocity of the planet in km/s.
#     vel_s : numpy.ndarray
#         The initial velocity of the star in km/s.
#     v_0 : numpy.ndarray
#         The velocity of the barycenter in km/s.
#     m_p : float
#         The mass of the planet in kg.
#     m_s : float
#         The mass of the star in kg.
#
#     Attributes
#     ----------
#     G : float
#         The gravitational constant in km^3 / kg / s^2.
#     m_p : float
#         The mass of the planet in kg.
#     m_s : float
#         The mass of the star in kg.
#     y_0 : numpy.ndarray
#         The initial state vector of the two-body system.
#     r_s : numpy.ndarray
#         The position of the star in km.
#     r_p : numpy.ndarray
#         The position of the planet in km.
#     v_s : numpy.ndarray
#         The velocity of the star in km/s.
#     v_p : numpy.ndarray
#         The velocity of the planet in km/s.
#     barycenter : numpy.ndarray
#         The position of the barycenter in km.
#     """
#
#     def __init__(self,
#                  pos_p: np.ndarray,
#                  pos_s: np.ndarray,
#                  vel_p: np.ndarray,
#                  vel_s: np.ndarray,
#                  m_p: float,
#                  m_s: float,
#                  v_0: np.ndarray=np.zeros(3),
#                  G=6.67430e-20):
#         self.G = G
#         self.m_p = m_p
#         self.m_s = m_s
#
#         self.y_0 = np.hstack((pos_s, pos_p, vel_s + v_0, vel_p + v_0))
#
#     def absolute_motion(self, t, y):
#         """Calculate the motion of a two-body system in an inertial reference
#         frame.
#
#         The state vector ``y`` should be in the order:
#
#         1. Coordinates of $m_s$
#         2. Coordinates of $m_p$
#         3. Velocity components of $m_s$
#         4. Velocity components of $m_p$
#         """
#         # Get the six coordinates for m_s and m_p from the state vector
#         r_1 = y[:3]
#         r_2 = y[3:6]
#
#         # Fill the derivative vector with zeros
#         ydot = np.zeros_like(y)
#
#         # Set the first 6 elements of the derivative equal to the last
#         # 6 elements of the state vector, which are the velocities
#         ydot[:6] = y[6:]
#
#         # Calculate the acceleration terms and fill them in to the rest
#         # of the derivative array
#         r = np.sqrt(np.sum(np.square(r_2 - r_1)))
#         ddot = self.G * (r_2 - r_1) / r ** 3
#         ddotR_1 = self.m_p * ddot
#         ddotR_2 = -self.m_s * ddot
#
#         ydot[6:9] = ddotR_1
#         ydot[9:] = ddotR_2
#         return ydot
#
#     def solve_orbit(self, t_min, t_max, num_points=10000):
#         """Solve the orbit of the two-body system.
#
#         Parameters
#         ----------
#         t_min : float
#             The start time of the integration in days.
#         t_max : float
#             The end time of the integration in days.
#         num_points : int
#             The number of points to solve for.
#
#         Returns
#         -------
#         r_s : numpy.ndarray
#             The position of the star in km.
#         r_p : numpy.ndarray
#             The position of the planet in km.
#         v_s : numpy.ndarray
#             The velocity of the star in km/s.
#         v_p : numpy.ndarray
#             The velocity of the planet in km/s.
#         barycenter : numpy.ndarray
#             The position of the barycenter in km.
#         """
#
#         t_min *= 24 * 60 * 60
#         t_max *= 24 * 60 * 60
#
#         t_points = np.linspace(t_min, t_max, num_points)
#
#         sol = solve_ivp(fun=self.absolute_motion,
#                         t_span=[t_min, t_max],
#                         y0=self.y_0,
#                         t_eval=t_points)
#
#         y = sol.y.T
#         self.r_s = y[:, :3]  # km
#         self.r_p = y[:, 3:6]  # km
#         self.v_s = y[:, 6:9]  # km/s
#         self.v_p = y[:, 9:]  # km/s
#         self.barycenter = ((self.m_s * self.r_s + self.m_p * self.r_p)
#                            / (self.m_s + self.m_p))
#
#     def plot_orbit(self):
#         round_to = 10
#         self.r_s_au = self.r_s / 1.496e8
#         self.r_p_au = self.r_p / 1.496e8
#         self.barycenter_au = self.barycenter / 1.496e8
#
#         fig = plt.figure(figsize=(15, 5))
#         ax = fig.add_subplot(131, projection='3d')
#         ax.plot(self.r_s_au[:, 0], self.r_s_au[:, 1], self.r_s_au[:, 2],
#                 label='star')
#         ax.plot(self.r_p_au[:, 0], self.r_p_au[:, 1], self.r_p_au[:, 2],
#                 label='exoplanet')
#         ax.plot(self.barycenter_au[:, 0],
#                 self.barycenter_au[:, 1],
#                 self.barycenter_au[:, 2],
#                 label='C/G')
#         ax.legend()
#
#         ax = fig.add_subplot(132, projection='3d')
#         r_s_rel_cg = np.round(self.r_s_au - self.barycenter_au,
#                               decimals=round_to)
#         r_p_rel_cg = np.round(self.r_p_au - self.barycenter_au,
#                               decimals=round_to)
#         ax.plot(r_s_rel_cg[:, 0], r_s_rel_cg[:, 1], r_s_rel_cg[:, 2],
#                 label='star')
#         ax.plot(r_p_rel_cg[:, 0], r_p_rel_cg[:, 1], r_p_rel_cg[:, 2],
#                 label='exoplanet')
#         ax.plot(0, 0, 0, 'ro', label='C/G')
#         ax.legend()
#
#         ax = fig.add_subplot(133, projection='3d')
#         r_p_rel_r_s = np.round(self.r_p_au - self.r_s_au,
#                                decimals=round_to)
#         cg_rel_r_s = np.round(self.barycenter_au - self.r_s_au,
#                               decimals=round_to)
#         ax.plot(r_p_rel_r_s[:, 0], r_p_rel_r_s[:, 1], r_p_rel_r_s[:, 2],
#                 label='exoplanet')
#         ax.plot(cg_rel_r_s[:, 0], cg_rel_r_s[:, 1], cg_rel_r_s[:, 2],
#                 label='C/G')
#         ax.plot(0, 0, 0, 'ro', label='star')
#         ax.legend()
#
#         plt.show()


def get_exoplanet_orbit(semimajor: float,
                        period: float,
                        eccentricity: float,
                        inclination: float):
    """Calculate the position and velocity of an exoplanet.

    Parameters
    ----------
    semimajor : float
        The semi-major axis of the orbit in AU.
    period : float
        The period of the orbit in days.
    eccentricity : float
        The eccentricity of the orbit.
    inclination : float
        The inclination of the orbit in degrees.

    Returns
    -------
    pos : numpy.ndarray
        The 3D-position-vector of the planet in km.
    vel : numpy.ndarray
        The 3D-velocity-vecotr of the planet in km/s.
    """

    ke = pyasl.KeplerEllipse(a=semimajor * 1.496e8,
                             per=period * 24 * 60 * 60,
                             e=eccentricity,
                             Omega=90.,
                             i=inclination,
                             w=0.0,
                             tau=0)
    pos = ke.xyzPos(0)
    vel = ke.xyzVel(0)

    return pos, vel
