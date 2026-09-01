#Contains the implementation of the Hermite integrator for the N-body simulation

#Imports
import numpy as np
from .dynamics import getaccel, energy
from .orbital import kroupa
from .save import save_snapshot

#4th order Hermite integrator implementation

# Hermite scheme
def Hermite(x,y,z,vx,vy,vz,masses,dt,tmax,nu,n_order,is_star,snapshot,data,
            snapshot_folder="data/raw/snapshots", progress_interval=None):
  if n_order < 1:
    raise ValueError("n_order must be at least 1")

  time = 0.0
  loop = 0
  sixth = 1.0 / 6.0
  if dt <= 0 or tmax <= 0 or data <= 0:
    raise ValueError("dt, tmax, and data must be positive")
  if snapshot is not None and snapshot <= 0:
    raise ValueError("snapshot must be positive when snapshots are enabled")
  if progress_interval is not None and progress_interval <= 0:
    raise ValueError("progress_interval must be positive when progress is enabled")
  next_snapshot_time = snapshot if snapshot is not None else np.inf
  next_data_time = data
  snapshot_counter = 1
  next_progress_time = progress_interval if progress_interval is not None else np.inf

  xl=[]
  yl=[]
  zl=[]
  vxl=[]
  vyl=[]
  vzl=[]
  kel=[]
  gpel=[]
  tel=[]
  timel=[]
  eccenl=[]
  angl=[]
  pl=[]
  al=[]
  ebl=[]

  ax,ay,az,axdot,aydot,azdot = getaccel(x,y,z,vx,vy,vz,masses)  # Initial accel and jerk calculations.

  while time < tmax:
    if loop == 0:
      pass
    else:
      a1 = np.sqrt(ax1*ax1 + ay1*ay1 + az1*az1)
      a1dot = np.sqrt(axdot1*axdot1 + aydot1*aydot1 + azdot1*azdot1)
      ac3 = np.sqrt(axc3*axc3 + ayc3*ayc3 + azc3*azc3)
      acx2_ext = axc2 + dt*axc3
      acy2_ext = ayc2 + dt*ayc3
      acz2_ext = azc2 + dt*azc3
      ac2_ext = np.sqrt(acx2_ext*acx2_ext + acy2_ext*acy2_ext + acz2_ext*acz2_ext)
      dt_aarseth = np.sqrt( nu * (a1*ac2_ext + a1dot*a1dot) / (a1dot*ac3 + ac2_ext*ac2_ext) )
      dt_try = min(dt_aarseth)
      dt = np.clip(dt_try, 0.01 * dt, 2.0 * dt)

    # Avoid advancing beyond the requested end time.
    dt = min(dt, tmax - time)

    #Get accel and jerk for this timestep
    if loop > 0:
      ax,ay,az,axdot,aydot,azdot = getaccel(x,y,z,vx,vy,vz,masses)

    # Using current position and velocity, predict new position and velocity.
    xp = x + dt*vx + 0.5*dt*dt*ax + sixth*dt*dt*dt*axdot
    yp = y + dt*vy + 0.5*dt*dt*ay + sixth*dt*dt*dt*aydot
    zp = z + dt*vz + 0.5*dt*dt*az + sixth*dt*dt*dt*azdot
    vxp = vx + dt*ax + 0.5*dt*dt*axdot
    vyp = vy + dt*ay + 0.5*dt*dt*aydot
    vzp = vz + dt*az + 0.5*dt*dt*azdot

    #Correction loop
    for n_iter in range(0, n_order):
      if n_iter == 0:
        ax1,ay1,az1,axdot1,aydot1,azdot1 = getaccel(xp,yp,zp,vxp,vyp,vzp,masses) #If first loop then use predicted pos and vel
      else:
        ax1,ay1,az1,axdot1,aydot1,azdot1 = getaccel(xc,yc,zc,vxc,vyc,vzc,masses) #On subsequent loops use corrected pos and vel to further correct and add onto predicted values

      #Now work out 2nd and 3rd order corrections based on ax and ax1
      axc2 = ( -6.0*(ax - ax1) - dt*(4.0*axdot + 2.0*axdot1) ) / (dt**2)
      ayc2 = ( -6.0*(ay - ay1) - dt*(4.0*aydot + 2.0*aydot1) ) / (dt**2)
      azc2 = ( -6.0*(az - az1) - dt*(4.0*azdot + 2.0*azdot1) ) / (dt**2)

      axc3 = ( 12.0*(ax - ax1) + 6.0*dt*(axdot + axdot1) ) / (dt**3)
      ayc3 = ( 12.0*(ay - ay1) + 6.0*dt*(aydot + aydot1) ) / (dt**3)
      azc3 = ( 12.0*(az - az1) + 6.0*dt*(azdot + azdot1) ) / (dt**3)

      # finally, correct the predicted step using these corrected accels
      xc = xp + (dt**4)*axc2/24. + (dt**5)*axc3/120.
      yc = yp + (dt**4)*ayc2/24. + (dt**5)*ayc3/120.
      zc = zp + (dt**4)*azc2/24. + (dt**5)*azc3/120.

      vxc = vxp + (dt**3)*axc2/6. + (dt**4)*axc3/24.
      vyc = vyp + (dt**3)*ayc2/6. + (dt**4)*ayc3/24.
      vzc = vzp + (dt**3)*azc2/6. + (dt**4)*azc3/24.

    # Update positions and velocities.
    x = xc
    y = yc
    z = zc
    vx = vxc
    vy = vyc
    vz = vzc

    time += dt
    loop += 1

    if time >= next_progress_time:
      print(
        f"Integration progress: {100.0 * time / tmax:.1f}% "
        f"({time / 31557600.0:.1f} simulated years), "
        f"timestep={dt:.3g} s",
        flush=True,
      )
      while next_progress_time <= time:
        next_progress_time += progress_interval

    # Calculate energies and orbital properties, then append them to the output lists.
    if time >= next_data_time or time >= tmax:
        ke,gpe,te = energy(x,y,z,vx,vy,vz,masses)
        a,ecc,ang,p,eb = kroupa(x,y,z,vx,vy,vz,masses,is_star)
        xl.append(x)
        yl.append(y)
        zl.append(z)
        vxl.append(vx)
        vyl.append(vy)
        vzl.append(vz)
        kel.append(ke)
        gpel.append(gpe)
        tel.append(te)
        eccenl.append(ecc)
        angl.append(ang)
        pl.append(p)
        al.append(a)
        ebl.append(eb)
        timel.append(time)
        while next_data_time <= time:
          next_data_time += data

        if snapshot is not None and time >= next_snapshot_time:
          save_snapshot(
            snapshot_counter, timel, xl, yl, zl, vxl, vyl, vzl,
            kel, gpel, tel, eccenl, angl, pl, len(masses), snapshot_folder
          )
          snapshot_counter += 1
          while next_snapshot_time <= time:
            next_snapshot_time += snapshot

  print(time)
  print(loop)

  return xl, yl, zl, vxl, vyl, vzl, kel, gpel, tel, eccenl, angl, pl, al, ebl, timel
