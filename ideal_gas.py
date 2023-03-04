"""
    ------------------------------
    ---- Ideal Gas Simulation ----
    ------------------------------

    The simplest definition of Ideal Gas states that it is a gas that follows 
    the model of elastic collisions between particles and assumes that there 
    are no intermolecular forces between its particles.

    In this code, we perform simulation of an ideal gas at temperature T 
    composed of N particles constrained to move in a 2-dimensional box of 
    length L.

    author: @idiegoalvarado
    github.com/iDiegoAlvarado/

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation


"""
    Define systems' parameters.
"""

L = 1.0                   # Lenght of the box       [m]
T = 5.0e12                # System's temperature    [K]

N = 150                   # Number of particles    
m = 1.0e-8                # Mass of the particles   [kg]
r = 1.0e-2                # Radius of the particles [m] 

R   = 8.3145              # Universal gas constant  [J/mol K]
k_B = 1.3806e-23          # Boltzmann constant      [J/K]
U   = 2/3 * N * k_B * T   # Internal energy         [J] 

TIME = 1000               # Simulation time         [s]
dt   = 0.03               # Time step               [s]





class Particle:


    def __init__(self, pos, vel, radius, mass):

        """
            Initiallisation of particle parameters


            ::mass:: of the particles (identicle for all of them).

            ::radius:: of the particles, in order for the particles to
                        collide, the size of each particle must be of the
                        same order as L / sqrt(N).

            ::pos:: initial position of the particle given as a
                    2-dimensional array. 

            ::vel:: initial velocity of the particle given as a 
                    2-dimensional array.
                    
        """

        self.radius = radius
        self.mass   = mass
        self.pos    = pos
        self.vel    = vel



    def interaction(self, op):

        """
            Interaction between two particles when collision
            occurs.
        """

        p_pos, op_pos  = self.pos, op.pos
        p_vel, op_vel  = self.vel, op.vel
        rad2 = self.radius + op.radius

        dist = p_pos - op_pos
        dist_norm = np.linalg.norm(dist)

        if dist_norm <= rad2: 

            self.vel = p_vel  - np.dot(p_vel - op_vel,  dist) * ( dist) / dist_norm ** 2    
            op.vel   = op_vel - np.dot(op_vel - p_vel, -dist) * (-dist) / dist_norm ** 2

            """
                When a collision occurs, the particles strictly separate at the 
                next timestamp in dirrection of the colide vec(r1 - r2).
            """
            clip = rad2 - dist_norm
            norm = dist / dist_norm
            clip_vec = norm * clip

            self.pos += clip_vec

       
    
    
    def motion(self, particles):
        
        """
            The motion of the particles is triggered.
        """

        for p in particles:
            if self == p:
                continue
            self.interaction(p)

        """
            Checks for wall collision. If the particle penetrates the wall, then
            it is forcely moved to be within the limits of the box.
        """        

        if self.pos[0] < 0 + self.radius:
            self.vel[0]  = -self.vel[0]
            self.pos[0] += 2*(0 + self.radius - self.pos[0])

        elif self.pos[0] > L - self.radius:
            self.vel[0]  = -self.vel[0]
            self.pos[0] += 2*(L - self.radius - self.pos[0])
        
        if self.pos[1] < 0 + self.radius:
            self.vel[1]  = -self.vel[1]
            self.pos[1] += 2*(0 + self.radius - self.pos[1])

        elif self.pos[1] > L - self.radius:
            self.vel[1]  = -self.vel[1]
            self.pos[1] += 2*(L - self.radius - self.pos[1])
        
        # Updates position
        self.pos = self.pos + self.vel * dt




def max_boltz(V):
    
    """
        Computes the Maxwell-Boltzmann distribution.
    """

    fv = m * np.exp(-m * V ** 2 / (2 * T * k_B)) / (2 * np.pi * T * k_B) * 2 * np.pi * V

    return fv




def init_param():

    """
        Defines each particle parameters.

        
        Using the random.dirichlet() module, the internal energy of the
        of the system is randomly distributed for all of the N particles.
        This will define the initial velovity for each particle.

        The initial positions are defined in a perfect squared grid.
        
    """

    # array of energies
    u  = np.random.uniform(size=N)
    u *= U / sum(u)

    # squared grid for initial position
    X = np.linspace(L*0.1, L*0.9, int(np.ceil(np.sqrt(N))))
    grid = [np.array([x, y]) for x in X for y in X]

    # create a list where particles' parameters will be stored.
    particles = []

    for i in range(N):
        
        rx = np.random.uniform()
        ry = (1 - rx)
        r_norm = np.linalg.norm((np.array([rx, ry])))
        
        vel_x = np.sqrt(2 * u[i] / m) * rx/r_norm * (-1) ** np.random.randint(1,3)
        vel_y = np.sqrt(2 * u[i] / m) * ry/r_norm * (-1) ** np.random.randint(1,3)

        pos = grid[i]
        vel = np.array([vel_x, vel_y])
        
        particles.append(Particle(pos, vel, r, m))

    return particles




def sys_motion(particles):

    """
        Computes and stores all system motion.

        The possitions and velocities for each particle in each time step
        are stored in two lists:

        pos_coll = [
            [p1.pos, p2.pos, p3.pos, ... ],    t = 1
            [p1.pos, p2.pos, p3.pos, ... ],    t = 2
            [p1.pos, p2.pos, p3.pos, ... ],    t = 3
            ... ] 

        vel_coll = [
            [p1.vel, p2.vel, p3.vel, ... ],    t = 1
            [p1.vel, p2.vel, p3.vel, ... ],    t = 2
            [p1.vel, p2.vel, p3.vel, ... ],    t = 3
            ... ] 
    """

    # lists for storing lists of particles' possition and velocity
    pos_coll = []
    vel_coll = []

    for _ in range(TIME):

        # lists for storing particles particles' possition and velocity at time t
        pos_dt = []
        vel_dt = []

        for p in particles:
            p.motion(particles)
            pos_dt.append(p.pos.copy())
            mod_vel = np.linalg.norm(p.vel.copy())
            vel_dt.append(mod_vel)
        
        pos_coll.append(pos_dt)
        vel_coll.append(vel_dt)
    
    return pos_coll, vel_coll




def simulation(pos_collection, vel_collection):

    """
        The system's dynamics and speed histograms are animated.
    """

    v_mod = [np.linalg.norm(vel) for vel in vel_collection[0]]
    v_max = max(v_mod)

    V     = np.linspace(0, v_max*2.5, 120)
    fv    = max_boltz(V)

    pos_up = list(zip(*pos_collection))
    v_hist = vel_collection  

    
    # initialise figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].add_patch(Rectangle((0, 0), L, L, ls='--', fill=False))
    ax[0].set_xlim([0 - L/10, L + L/10])
    ax[0].set_ylim([0 - L/10, L + L/10])
    ax[0].set_aspect('equal', 'box')
    ax[0].set_title('Ideal gas simulation')
    ax[0].text(L*0.65, L*1.04, f'# of particles = {N}', fontsize=8)

    ax[1].set_ylim([0, max(fv)*1.5])
    ax[1].plot(V, fv, lw = 2, label = 'Maxwell-Boltzmann distribution')
    ax[1].set_title('Frequency Distribution of Speeds')
    ax[1].set_xlabel('Speed ($v$)')
    ax[1].set_ylabel('Density Frequency ($f(v)$)')


    circles = [Circle((0, 0), radius=r, color='black', fill = False) for _ in range(len(pos_up))] 
    for circle in circles:
        ax[0].add_patch(circle)

    BINS = np.linspace(0, v_max*2.5, 15)
    n, _ = np.histogram(v_hist[0], BINS, density=True)
    _, _, bar_container = ax[1].hist(v_hist[0], BINS, lw=1, ec="black", 
                                     fc="orange", label='Simulated data')
    ax[1].legend(fontsize=8)


    # Update figure
    def update(i):

        for circle, xy in zip(circles, pos_up):
            circle.center = xy[i]

        n, _ = np.histogram(v_hist[i], BINS, density=True)
        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)
        
        return circles, bar_container.patches

    anim = FuncAnimation(fig, update, frames=TIME, blit=False, interval=100)
    
    # to save animation select save = True 
    save = False
    if save:
        anim.save('idealgas_test_4.mp4', fps=60, dpi=400)
    
    plt.show()




def main():

    """
        Executes all code
    """

    initial_params = init_param()
    
    sys_mo = sys_motion(initial_params)
    sys_mo_pos = sys_mo[0]
    sys_mo_vel = sys_mo[1]

    simulation(sys_mo_pos, sys_mo_vel)


main()
