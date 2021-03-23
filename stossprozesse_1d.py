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
    
def anim_1D_colliding_particles(start_pos, start_vel):
    
   
    return



        

# ----------  ----------

pos_boundary = (1.,2.)
num_particles = 5
num_steps = 5 
time_step = 0.5


start_pos = (pos_boundary[1] - pos_boundary[0]) * np.random.random(num_particles) + pos_boundary[0]
start_vel = np.random.random(num_particles) - 0.5


free_trajectories = free_propagation(start_pos, start_vel, time_step, num_steps)
interacting_trajectories = free_to_interacting(free_trajectories, pos_boundary)

