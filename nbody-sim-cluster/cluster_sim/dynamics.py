#Contains all the dynamics-related functions for the stellar cluster simulation

#Imports
import numpy as np
from numba import jit
from .constants import G, epsilon

#Accelerations and jerks for the N-body system
@jit(nopython=True)
def getaccel(x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, masses):
  num_bodies = len(masses)
  ax = np.zeros(num_bodies)
  ay = np.zeros(num_bodies)
  az = np.zeros(num_bodies)
  axdot = np.zeros(num_bodies)
  aydot = np.zeros(num_bodies)
  azdot = np.zeros(num_bodies)

  #Set up loop
  for i in range(0, num_bodies):
    for j in range(0, num_bodies):
      if i != j:
        dx = x_pos[j] - x_pos[i]
        dy = y_pos[j] - y_pos[i]
        dz = z_pos[j] - z_pos[i]
        dvx = x_vel[j] - x_vel[i]
        dvy = y_vel[j] - y_vel[i]
        dvz = z_vel[j] - z_vel[i]
        dot = (dx*dvx + dy*dvy + dz*dvz)
        dist = np.sqrt(dx*dx + dy*dy + dz*dz + epsilon*epsilon)
        acc_denom = (1.0 / dist**3)
        jerk_denom = (1.0 / dist**5)

        #Acceleration of body j on body i
        ax[i] += G * masses[j] * dx * acc_denom
        ay[i] += G * masses[j] * dy * acc_denom
        az[i] += G * masses[j] * dz * acc_denom

        axdot[i] += G * masses[j] * (dvx * acc_denom - 3.0 * dot * dx * jerk_denom)
        aydot[i] += G * masses[j] * (dvy * acc_denom - 3.0 * dot * dy * jerk_denom)
        azdot[i] += G * masses[j] * (dvz * acc_denom - 3.0 * dot * dz * jerk_denom)

  return ax, ay, az, axdot, aydot, azdot

#Energy calculation for the N-body system
@jit(nopython=True)
def energy(x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, masses):
  num_bodies = len(masses)

  #First calculate kinetic energy by accounting for the center of mass motion
  comx = np.sum(masses*x_vel)/ np.sum(masses)
  comy = np.sum(masses*y_vel)/ np.sum(masses)
  comz = np.sum(masses*z_vel)/ np.sum(masses)

  vx = x_vel - comx
  vy = y_vel - comy
  vz = z_vel - comz

  speed= (vx*vx + vy*vy + vz*vz)
  ke = np.sum(0.5 * masses * speed)

  #Now calculate gravitational potential energy (GPE)
  gpe = 0
  for i in range(0,num_bodies):
    for j in range(i+1,num_bodies): #Avoids double counting
      dx = x_pos[j] - x_pos[i]
      dy = y_pos[j] - y_pos[i]
      dz = z_pos[j] - z_pos[i]
      dist = np.sqrt(dx*dx + dy*dy + dz*dz + epsilon*epsilon)
      denom = 1/dist

      gpe -= G * masses[j]*masses[i] * denom

  te = gpe + ke
  return ke, gpe, te