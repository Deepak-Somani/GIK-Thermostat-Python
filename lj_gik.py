#==========================Molecular Dynamics Simulation of 2D LJ-Gas with application of GIK Thermostat============================#
#========================================Created by Deepak Somani============================================#

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 500 # Number of particles
t_max = 600 # Maximum time units for the simulation
dt = 1 # One unit time for step propagation

r_min = -100000.0
r_max = 100000.0
# Simulation Box

epsilon = 1.0
sigma = 1.0
# LJ parameterss

m = 1.0 # mass of particle

d_max = 500.0
d_min = 0.00001

x_pos = np.linspace(-1000,1000,N)#10*np.random.random(N) # Initialization
y_pos = np.linspace(-1000,1000,N)#10*np.random.random(N) # Initialization
x_vel = np.zeros(N) #10*np.random.random(N) # Initialization
y_vel = np.zeros(N) #10*np.random.random(N) # Initialization

x_pos_new = np.zeros(N)
y_pos_new = np.zeros(N)
x_vel_new = np.zeros(N)
y_vel_new = np.zeros(N)

t = np.linspace(0,t_max,int(t_max/dt))
KE = np.zeros(len(t))

force_x = np.zeros(N)
force_y = np.zeros(N)

#================================NVE run=======================================#

for count in range(0,int(t_max/(3*dt)),1):

    vel_sum_sq = 0
    force_x_new = np.zeros(N)
    force_y_new = np.zeros(N)
    
    for i in range(0,N,1):
        for j in range(i+1,N,1):
            
            dist_x = abs(x_pos[i] - x_pos[j])
            if dist_x>d_max: dist_x = d_max
            elif dist_x<d_min: dist_x = d_min
            
            dist_y = abs(y_pos[i] - y_pos[j])
            if dist_y>d_max: dist_y = d_max
            elif dist_y<d_min: dist_y = d_min
            
            fx = 4*epsilon*((12*(sigma**12)/(dist_x)**13)-(6*(sigma**6)/(dist_x)**7))
            fy = 4*epsilon*((12*(sigma**12)/(dist_y)**13)-(6*(sigma**6)/(dist_y)**7))
            sign_x = (x_pos[i] - x_pos[j])/dist_x
            sign_y = (y_pos[i] - y_pos[j])/dist_y
            
            force_x_new[i] += -sign_x*fx
            force_y_new[i] += -sign_y*fy

            force_x_new[j] += sign_x*fx
            force_y_new[j] += sign_y*fy
            
        x_vel_new[i] = x_vel[i] + dt*((force_x_new[i] + force_x[i])/(2*m))
        y_vel_new[i] = y_vel[i] + dt*((force_y_new[i] + force_y[i])/(2*m))
        
        x_pos_new[i] = x_pos[i] + dt*x_vel[i] + (force_x[i]/(2*m))*(dt**2)
        y_pos_new[i] = y_pos[i] + dt*y_vel[i] + (force_y[i]/(2*m))*(dt**2)
        vel_sum_sq += x_vel[i]**2 + y_vel[i]**2
    
    #print(x_pos[1],y_pos[1])
    #print(x_pos_new[1],y_pos_new[1])
    
    x_pos = x_pos_new
    y_pos = y_pos_new
    
    x_vel = x_vel_new
    y_vel = y_vel_new
    
    force_x = force_x_new
    force_y = force_y_new
    
    KE[count] = 0.5*m*vel_sum_sq
        

#plt.plot(t, KE, 'k')
#plt.show()

count_old = count

#==============================GIK Thermostat===================================#

for count in range(0,2*int(t_max/(3*dt)),1):
    
    var = 0
    
    for i in range(0,N,1):
        var += force_x[i]*x_vel[i] + force_y[i]*y_vel[i]
    
    var = var/vel_sum_sq
    
    vel_sum_sq = 0
    force_x_new = np.zeros(N)
    force_y_new = np.zeros(N)
    
    for i in range(0,N,1):
        for j in range(i+1,N,1):
            
            dist_x = abs(x_pos[i] - x_pos[j])
            if dist_x>d_max: dist_x = d_max
            elif dist_x<d_min: dist_x = d_min
            
            dist_y = abs(y_pos[i] - y_pos[j])
            if dist_y>d_max: dist_y = d_max
            elif dist_y<d_min: dist_y = d_min
            
            fx = 4*epsilon*((12*(sigma**12)/(dist_x)**13)-(6*(sigma**6)/(dist_x)**7))
            fy = 4*epsilon*((12*(sigma**12)/(dist_y)**13)-(6*(sigma**6)/(dist_y)**7))
            sign_x = (x_pos[i] - x_pos[j])/dist_x
            sign_y = (y_pos[i] - y_pos[j])/dist_y
            
            force_x_new[i] += -sign_x*fx
            force_y_new[i] += -sign_y*fy

            force_x_new[j] += sign_x*fx
            force_y_new[j] += sign_y*fy
        
        x_vel_new[i] = x_vel[i] + dt*((force_x_new[i] + force_x[i])/(2*m)) - var*dt*x_vel[i]
        y_vel_new[i] = y_vel[i] + dt*((force_y_new[i] + force_y[i])/(2*m)) - var*dt*y_vel[i]
        
        x_pos_new[i] = x_pos[i] + dt*x_vel[i] + (force_x[i]/(2*m))*(dt**2)
        y_pos_new[i] = y_pos[i] + dt*y_vel[i] + (force_y[i]/(2*m))*(dt**2)
        vel_sum_sq += x_vel[i]**2 + y_vel[i]**2
    
    #print(x_pos[1],y_pos[1])
    #print(x_pos_new[1],y_pos_new[1])
    
    x_pos = x_pos_new
    y_pos = y_pos_new
    
    x_vel = x_vel_new
    y_vel = y_vel_new
    
    force_x = force_x_new
    force_y = force_y_new
    
    KE[count+count_old] = 0.5*m*vel_sum_sq

plt.plot(t[:-1], KE[:-1], 'r')
plt.axvline(x = int(t_max/(3*dt)), color = 'b')
plt.legend(['Kinetic Energy'])
plt.suptitle('Kinetic Energy plot for LJ gas')
plt.title('NVE applied for first one-third and for the rest GIK Thermostat is applied for ' + str(N) + ' particles', fontsize = 10)
plt.savefig('Kinetic Energy plot with GIK Thermostat application with '+str(N)+' particles.png', dpi = 400)
