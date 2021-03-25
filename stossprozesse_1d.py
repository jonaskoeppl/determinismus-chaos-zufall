# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:09:59 2021

@author: SEBASTIAN
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams
import time
import math

# Pfad zu meiner Kopie des FFMPEG Writers zum Speichern der Animation
rcParams['animation.ffmpeg_path'] = r'D:\Users\SEBASTIAN\.matplotlib\ffmpeg-2021-01-01-git-63505fc60a-full_build\bin\ffmpeg.exe'


# ---------- Berechnung Stoßprozesse ----------

def free_propagation(pos, vel, time_step, num_steps): 
    num_particles = pos.shape[0]
    start = np.tile(pos, num_steps).reshape((num_steps, num_particles))
    steps = time_step * np.tile(vel, num_steps).reshape((num_steps, num_particles))
    steps[0] = 0
       
    free_trajectories = start + np.cumsum(steps, axis = 0)
    
    return free_trajectories

    
def free_to_interacting(free_trajectories):   
    ones = np.ones_like(free_trajectories)
    
    max_right_excess = np.max(free_trajectories) - 1
    max_left_excess = 0 - np.min(free_trajectories)
    
    max_bounces = max(max_right_excess, max_left_excess)
    num_bounces = math.floor(max_bounces)
        
    # hier ist der größte Zeitfresser (i.e. Code beschleunigen oder verhindern, dass es viele Reflektionen gibt)   
    for i in range(num_bounces):
        free_trajectories = np.abs( free_trajectories ) # left bounce
        free_trajectories = ones - np.abs( ones - free_trajectories ) # right bounce

    interacting_trajectories = np.sort(free_trajectories, axis = -1)
        
    return interacting_trajectories


def free_to_interacting_circle(free_trajectories):   
    num_steps = free_trajectories.shape[0]
    
    ones = np.ones_like(free_trajectories)
    zeros = np.zeros_like(free_trajectories)
    
    right_excess = np.where( (free_trajectories - ones) > 0, free_trajectories - ones, zeros)
    right_excess = np.ceil(right_excess).astype("int")
    left_excess = np.where( (zeros - free_trajectories) > 0, zeros - free_trajectories, zeros)
    left_excess = np.ceil(left_excess).astype("int")
    
    shifts = - ( np.sum(right_excess, axis = 1) - np.sum(left_excess, axis = 1) )
    
    free_trajectories = free_trajectories % 1.
    interacting_trajectories = np.sort(free_trajectories, axis = -1)
    for n in range(num_steps):
        interacting_trajectories[n] = np.roll(interacting_trajectories[n,:], shift = shifts[n])
        
    return interacting_trajectories, shifts



# ---------- Untersuchung mittlerer Punkt ----------
    
def rescaling1(interacting_trajectories, time_step):
    num_steps = interacting_trajectories.shape[0]
    num_particles = interacting_trajectories.shape[1]
    
    middle_index = math.ceil(num_particles / 2)
    middle_trajectory = interacting_trajectories[:,middle_index]
    
    scale = np.linspace(1, num_steps, endpoint = True, num = num_steps)
    scale = (np.pi / (2 * scale))**(1/4)
    
    middle_trajectory = scale * middle_trajectory
    
    return middle_trajectory
    

def rescaling2_donsker(interacting_trajectories, num_rescale_steps, num_time_steps):

    num_particles = interacting_trajectories.shape[1]
    middle_index = math.ceil(num_particles / 2)
    
    middle_trajectory = interacting_trajectories[:,middle_index]
    
    donsker_trajectory = np.zeros((num_rescale_steps,num_time_steps))


    for n in range(num_rescale_steps): 
        for step in range(num_time_steps): 
            t = step/num_time_steps
            donsker_trajectory[n,step] = 1./(np.sqrt(n+1)) * middle_trajectory[math.ceil(n*t)]
    return donsker_trajectory 



# ---------- Animation Stoßprozesse auf der Geraden ----------
    
def anim_1D_colliding_particles(start_pos, start_vel, time_step, num_steps, num_frames):
    
    free_trajectories = free_propagation(start_pos, start_vel, time_step, num_steps)
    interacting_trajectories = free_to_interacting(free_trajectories)
    
    fig, axes = plt.subplots(1,2, figsize = (11,5))
    fig.subplots_adjust(wspace = 0.4)
        
    anim = animation.FuncAnimation(fig, update_colliding_particles_anim, interval=250, 
                                   frames = num_frames, 
                                   fargs = (interacting_trajectories, time_step), 
                                   repeat = False)
   
    return anim


def update_colliding_particles_anim(i, interacting_trajectories, time_step):
    
    num_particles = interacting_trajectories.shape[1]    
    num_steps = interacting_trajectories.shape[0]
    middle_index = math.ceil(num_particles / 2.)
    
    color_labels = np.full(shape=num_particles, fill_value=10,dtype=np.int)
    color_labels[middle_index]=0
    
    middle_trajectory = interacting_trajectories[:,middle_index]
    times = time_step * np.linspace(0, i, endpoint = False, num = i)
    
    fig = plt.gcf()
    
    [ax1, ax2] = fig.axes
    ax1.clear()
    ax2.clear()
    ax1.set_xlim((-0.1, 1.1))
    ax2.set_xlim(( 0.95 * np.min(middle_trajectory), 1.05 * np.max(middle_trajectory)))
    ax2.set_ylim((- time_step, time_step * (num_steps + 1)))
    
    
    ax1.scatter(interacting_trajectories[i], np.ones_like(interacting_trajectories[i]),
               marker = "o", c = color_labels, cmap = "Set1")
    ax1.plot([-0.02, 1.02], [1,1], c = "#c4c4c4", marker = "|", zorder = 0)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax1.get_yaxis().set_visible(False)
    
    ax2.plot(middle_trajectory[0:i], times, c = "r")
    ax2.set_xlabel(r"Auslenkung $y_0(t)$ des roten Teilchens")
    ax2.set_ylabel(r"$t$")
    
    return anim



def anim_1D_colliding_particles_circle(start_pos, start_vel, time_step, num_steps, num_frames):
    
    free_trajectories = free_propagation(start_pos, start_vel, time_step, num_steps)
    interacting_trajectories, shifts = free_to_interacting_circle(free_trajectories)
    
    fig, ax= plt.subplots()
    
    anim = animation.FuncAnimation(fig, update_colliding_particles_anim_circle, interval=250, 
                                   frames = num_frames, 
                                   fargs = (interacting_trajectories, shifts), 
                                   repeat = False)
   
    return anim


def update_colliding_particles_anim_circle(i, interacting_trajectories, shifts):
    
    num_particles = interacting_trajectories.shape[1]
    
    color_labels = np.linspace(0, num_particles, num = num_particles)
    
    fig = plt.gcf()
    ax1 = fig.axes[0]
    ax1.clear()
    ax1.set_xlim((-1.1,1.1))
    ax1.set_ylim((-1.1,1.1))
    ax1.set_aspect("equal")
    plt.axis("off")
    
    circle = plt.Circle(((0.,0.)), 1., fill=False, edgecolor='#c4c4c4', zorder = 0)
    ax1.add_patch(circle)
    
    ax1.scatter(np.cos(2*np.pi*interacting_trajectories[i]), np.sin(2*np.pi*interacting_trajectories[i]),
               marker = "o", c = color_labels, cmap = "plasma")
    
    return anim



# ---------- Animation Verteilung Rescaling1 ----------

def anim_middle_particle(num_particles, speed, time_step, num_steps, num_reps, num_frames):
    
    middle_trajectory_scale1 = []
    
    for n in range(num_reps):
        start_pos = np.random.random(num_particles)
        start_pos = np.sort(start_pos)
        offset = ( start_pos[ math.ceil(num_particles / 2) ] - 0.5 )
        start_pos -= offset
        #pos_boundary = (pos_boundary[0] - offset, pos_boundary[1] - offset)
        # Reflection notwendig!!!
        
        start_vel = speed * (np.random.random(num_particles) - 0.5)
        start_vel[ math.ceil(num_particles / 2) ] = 0
        
        free_trajectories = free_propagation(start_pos, start_vel, time_step, num_steps)
        interacting_trajectories = free_to_interacting(free_trajectories)
        middle_trajectory_scale1.append( rescaling1(interacting_trajectories, time_step) )
        
    distribution_middle_particle_position = np.array( middle_trajectory_scale1 )

    
    fig, ax = plt.subplots()
    
    steps = np.unique( np.geomspace(1, num_steps, num_frames, 
                                    endpoint = True, dtype = int) ) - 1

    anim = animation.FuncAnimation(fig, update_middle_particle_anim, interval=100, 
                                   frames = steps.size, 
                                   fargs = (distribution_middle_particle_position, steps, num_particles), 
                                   repeat = False)
   
    return anim


def update_middle_particle_anim(i, distribution_middle_particle_position, steps, num_particles):
    
    print(steps[i], end=" ")
        
    fig = plt.gcf()

    mean = np.mean( distribution_middle_particle_position[:,steps[i]] )

    ax = fig.axes[0]
    ax.clear()
    ax.set_xlim((-0.1,0.1))
    ax.hist( distribution_middle_particle_position[:,steps[i]] - mean, bins = 21 )
    ax.set_title(r"Verteilung von $(\frac{\pi}{2t})^{(1/4)} y_0(t)$")
    
    return 


# ---------- Animation Verteilung Rescaling2/Donsker ----------

def anim_middle_particle_donsker(start_pos, start_vel, time_step, num_steps, num_frames, 
                                 anim_num_time_steps, scaling_steps):
    
    free_trajectories = free_propagation(start_pos, start_vel, time_step, num_steps)
    print("Free Trajectories done")
    interacting_trajectories = free_to_interacting(free_trajectories)
    print("Interacting Trajectories done")
    middle_trajectory_donsker = rescaling2_donsker(interacting_trajectories, scaling_steps, anim_num_time_steps)
    print("Rescaling done")
    
    fig, ax = plt.subplots()
    
    anim = animation.FuncAnimation(fig, update_middle_particle_donsker_anim, interval=100, 
                                   frames = num_frames, 
                                   fargs = (middle_trajectory_donsker, anim_num_time_steps), 
                                   repeat = False)
   
    return anim


def update_middle_particle_donsker_anim(i, middle_trajectory_donsker, anim_num_time_steps):
    
    print(i, end=" ")
    
    fig = plt.gcf()
    ax = fig.axes[0]
    
    x = np.linspace(0,1,anim_num_time_steps)
    y = middle_trajectory_donsker[i]
    ax.set_ylim(np.min(y)-0.5, np.max(y)+0.5)
    ax.set_title(r"$Y_A$ für A = " + str(i))
    ax.clear()
    ax.plot(x,y)
    
    return 
      

# ---------- Hauptprogramm ----------

num_particles = 10
num_steps = 200  
time_step = 0.1
num_frames = 200
speed = 0.3

start_pos = np.random.random(num_particles)
start_vel = speed * (np.random.random(num_particles) - 0.5)


free_trajectories = free_propagation(start_pos, start_vel, time_step, num_steps)
interacting_trajectories, shifts = free_to_interacting_circle(free_trajectories)

#anim = anim_middle_particle_donsker(start_pos, start_vel, time_step, num_steps, 100, 100, 100)
#anim = anim_middle_particle(num_particles, speed, time_step, num_steps = 2000, num_reps = 500,
#                            num_frames = 100)

anim = anim_1D_colliding_particles(start_pos, start_vel, time_step, num_steps, num_frames)

writermp4 = animation.FFMpegWriter() 
#anim.save("name.mp4", writer=writermp4)

plt.show()
