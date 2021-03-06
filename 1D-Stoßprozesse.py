# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:09:59 2021

@author: SEBASTIAN
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams

# Pfad zu meiner Kopie des FFMPEG Writers zum Speichern der Animation
#rcParams['animation.ffmpeg_path'] = r'D:\Users\SEBASTIAN\.matplotlib\ffmpeg-2021-01-01-git-63505fc60a-full_build\bin\ffmpeg.exe'


# ---------- Berechnung Stoßprozesse ----------

def free_propagation(pos, vel, time_step, num_steps): 
    num_particles = pos.shape[0]
    start = np.tile(pos, num_steps).reshape((num_steps, num_particles))
    steps = time_step * np.tile(vel, num_steps).reshape((num_steps, num_particles))
    steps[0] = 0
       
    free_trajectories = start + np.cumsum(steps, axis = 0)
    
    return free_trajectories
    
def free_to_interacting(free_trajectories, pos_boundary):   
    max_allowed_path_length = pos_boundary[1] - pos_boundary[0]
    
    max_right_excess = np.max(free_trajectories) - pos_boundary[1]
    max_left_excess = pos_boundary[0] - np.min(free_trajectories)
    
    max_bounces = max(max_right_excess, max_left_excess) / max_allowed_path_length
    max_bounces = max(0, np.ceil(max_bounces))
    num_bounces = int(max_bounces)
        
    for i in range(num_bounces):
        free_trajectories = np.abs( free_trajectories - pos_boundary[0] ) + pos_boundary[0] # left bounce
        free_trajectories = pos_boundary[1] - np.abs( pos_boundary[1] - free_trajectories ) # right bounce
    
    interacting_trajectories = np.sort(free_trajectories, axis = -1)
        
    return interacting_trajectories


# ---------- Animation Stoßprozesse ----------
def anim_1D_colliding_particles(start_pos, start_vel, pos_boundary, time_step, num_steps, num_frames):
    
    free_trajectories = free_propagation(start_pos, start_vel, time_step, num_steps)
    interacting_trajectories = free_to_interacting(free_trajectories, pos_boundary)
    
    fig, ax= plt.subplots(1,2)
    
    
    anim = animation.FuncAnimation(fig, update_colliding_particles_anim, interval=250, 
                                   frames = num_frames, 
                                   fargs = (interacting_trajectories, pos_boundary), 
                                   repeat = False)
   
    return anim


def update_colliding_particles_anim(i, interacting_trajectories, pos_boundary, num_steps=250):
    
    num_particles = interacting_trajectories.shape[1]
    
    fig = plt.gcf()
    ax1 = fig.axes[0]
    ax1.clear()
    ax1.set_xlim(pos_boundary)
    middle_of_array = int(num_particles / 2)
    color_labels = np.full(shape=num_particles, fill_value=10,dtype=np.int)
    color_labels[middle_of_array]=0
    ax1.scatter(interacting_trajectories[i], np.ones_like(interacting_trajectories[i]),
               marker = "o", c = color_labels, cmap = "Set1")
    middle_trajectory = interacting_trajectories[:,middle_of_array]
    time = np.linspace(1, i, num = i)
    ax2 = fig.axes[1]
    ax2.clear()
    ax2.plot(time, middle_trajectory[0:i])
    
    return 


    



        

# ---------- Hauptprogramm ----------

pos_boundary = (1.,2.)
num_particles = 5
num_steps = 250
time_step = 0.25


start_pos = (pos_boundary[1] - pos_boundary[0]) * np.random.random(num_particles) + pos_boundary[0]
start_vel = np.random.random(num_particles) - 0.5


free_trajectories = free_propagation(start_pos, start_vel, time_step, num_steps)
interacting_trajectories = free_to_interacting(free_trajectories, pos_boundary)

anim = anim_1D_colliding_particles(start_pos, start_vel, pos_boundary, time_step, num_steps, 5000)

writermp4 = animation.FFMpegWriter()
plt.show()