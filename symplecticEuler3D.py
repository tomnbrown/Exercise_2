"""
CMod Ex2: symplectic Euler time integration of
two particles moving in a Morse potential.

Produces plots of the vector seperation of the particles
and the energy of the system, both as function of time. Also
saves both to file.

V(x) = D*((1-exp[-a(delta_r-r)])^2-1), where
a, D and r are system dependent and imported from a user defined file
and passed to the functions that
calculate force and potential energy.
"""

import sys
import math
import numpy as np
import random as rdm
import matplotlib.pyplot as pyplot
from Particle3D import Particle3D


def force_dw(p1, p2, a, D, r):
    """
    Method to return the force on a particle
    in a Morse potential.
    Force is given by
    F(x) = -dV/dx = 2*a*D*(1-exp[-a(delta_r-r)])exp[-a(delta_r-r)]
    
    :param p1: Particle3D instance 1
    :param p2: Particle3D instance 2
    :param a: parameter alpha from Morse potential
    :param D: parameter D from potential
    :param r: parameter r from potential
    :return: force acting on particle as Numpy array
    """
    force = 2*a*D*(1-np.exp(-a*(Particle3D.delta_r(p1, p2)-r)))*np.exp(-a*(Particle3D.delta_r(p1, p2)-r))*(p2.position-p1.position)/np.linalg.norm(p2.position-p1.position)
    return force


def pot_energy_dw(p1, p2, a, D, r):
    """
    Method to return potential energy 
    of particle in Morse potential
    V(x) = D*((1-exp[-a(delta_r-r)])^2-1)

    :param p1: Particle3D instance 1
    :param p2: Particle3D instance 2
    :param a: parameter alpha from Morse potential
    :param D: parameter D from potential
    :param r: parameter r from potential
    :return: potential energy of particle system as float
    """
    potential = D*((1-np.exp(-a*(Particle3D.delta_r(p1, p2)-r)))**2-1)
    return potential


# Begin main code
def main():
    # Read name of output file from command line
    if len(sys.argv)!=3:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + " <output file>")
        quit()
    else:
        outfile_name = sys.argv[1]

    # Open output file
    outfile = open(outfile_name, "w")
    
    #Open input file with user defined parameters
    import_file = open(sys.argv[2], "r")
    data = import_file.readlines()
    import_file.close()

    # Set up simulation parameters
    dt = 0.001
    numstep = 20000
    time = 0.0
    a = float(data[15])
    D = float(data[17])
    r = float(data[19])

    # Set up particle initial conditions:
    #Reads in data from file
    p1 = Particle3D.create(sys.argv[2])
    p2 = Particle3D(np.array([-r/2, 0, 0]), np.array([-0.1, 0, 0]), 16.00)

    # Write out initial conditions
    energy = p1.kinetic_energy() + p2.kinetic_energy()+ pot_energy_dw(p1, p2, a, D, r)
    outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time,Particle3D.delta_r(p1,p2),energy))

    # Get initial force
    force = force_dw(p1, p2, a, D, r)
    
    # Initialise data lists for plotting later
    time_list = [time]
    pos_list = [Particle3D.delta_r(p1,p2)]
    energy_list = [energy]

    # Start the time integration loop
    for i in range(numstep):
        # Update particle position
        p1.leap_pos1st(dt)
        p2.leap_pos1st(dt)
        # Update force
        force_new = force_dw(p1, p2, a, D, r)
        # Update particle velocity using new force
        p1.leap_velocity(dt, force_new)
        p2.leap_velocity(dt, -force_new)
        
        # Re-define force value
        force = force_new

        # Increase time
        time += dt
        
        # Output particle information
        energy = p1.kinetic_energy() + p2.kinetic_energy() + pot_energy_dw(p1, p2, a, D, r)
        outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time,Particle3D.delta_r(p1, p2),energy))

        # Append information to data lists
        time_list.append(time)
        pos_list.append(Particle3D.delta_r(p1, p2))
        energy_list.append(energy)
    

    # Post-simulation:
    # Close output file
    outfile.close()

    # Plot particle trajectory to screen
    pyplot.title('Symplectic Euler: position vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Position')
    pyplot.plot(time_list, pos_list)
    pyplot.show()

    # Plot particle energy to screen
    pyplot.title('Symplectic Euler: total energy vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Energy')
    pyplot.plot(time_list, energy_list)
    pyplot.show()


# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()

