"""
 CMod Ex2: Particle3D, a class to describe 3D particles
"""
import numpy as np

class Particle3D(object):
    """
    Class to describe 3D particles.

    Properties:
    position(array) - position in x, y, z
    velocity(array) - velocity in x, y, z
    mass(float) - particle mass

    Methods:
    * formatted output
    * kinetic energy
    * first-order velocity update
    * first- and second order position updates
    * relative vector seperation between two particles
    * creation of particle instance from a file
    """

    def __init__(self, pos, vel, mass):
        """
        Initialise a Particle3D instance
        
        :param pos: position as array
        :param vel: velocity as array
        :param mass: mass as float
        :param label: particles label
        """
        self.position = pos
        self.velocity = vel
        self.mass = mass
    

    def __str__(self):
        """
        Define output format.
        For particle p=([2.0, 0.5, 1.0], [1.0, 0, 2.0], 5.0) this will print as
        "x = 2.0, y = 0.5, z = 1.0, vx = 1.0, vy = 0, vz = 2.0, m = 5.0"
        """
        return "x = " + str(self.position[0]) + " y = " + str(self.position[1]) + " z =  " + str(self.position[2]) + " vx = " + str(self.velocity[0]) + " vy = " + str(self.velocity[1]) + " vz = " + str(self.velocity[2]) + " m = " + str(self.mass)

    def kinetic_energy(self):
        """
        Return kinetic energy as
        1/2*mass*np.linalg.norm(vel)**2
        """
        return 0.5*self.mass*np.linalg.norm(self.velocity)**2
        

    # Time integration methods
    def leap_velocity(self, dt, force):
        """
        First-order velocity update,
        v(t+dt) = v(t) + dt*F(t)

        :param dt: timestep as float
        :param force: force on particle as an array
        """
        self.velocity += dt*force/self.mass

    def leap_pos1st(self, dt):
        """
        First-order position update,
        x(t+dt) = x(t) + dt*v(t)

        :param dt: timestep as float
        """
        self.position += dt*self.velocity


    def leap_pos2nd(self, dt, force):
        """
        Second-order position update,
        x(t+dt) = x(t) + dt*v(t) + 1/2*dt^2*F(t)

        :param dt: timestep as float
        :param force: current force as an array
        """
        self.position += (dt*self.velocity + 0.5*dt**2*force/self.mass)
    
    
    @staticmethod
    def delta_r(p1, p2):
        """
        Returns distance between two particles
        
        :param p1: particle 1
        :param p2: particle 2
        """
        return np.linalg.norm(p1.position - p2.position)
    
    @staticmethod
    def create(file):
        """
        Takes file handle as argument and creates Particle3D instance
    
        :param file: input file
        """
        import_file = open(file, "r")
        data = import_file.readlines()
        return Particle3D(np.array([float(data[1]), float(data[3]), float(data[5])]), np.array([float(data[7]), float(data[9]), float(data[11])]), float(data[13]))
