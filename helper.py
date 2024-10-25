from IPython.display import HTML, Video, display
import ipywidgets as widgets
import numpy as np
from numpy.random import randn

import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
from swift import Swift

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from neura_dual_quaternions import Quaternion, DualQuaternion
from matplotlib.patches import Circle

np.set_printoptions(precision=2, suppress=True, linewidth=200, formatter={'float': '{:8.3f}'.format})

def create_complex_plane():

    fig, ax = plt.subplots(figsize=(8, 8))  # Increase the plot size
    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.spines[["left", "bottom"]].set_color("black")
    # Hide the top and right spines.
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_facecolor('white')
    ax.grid(True)


    
    # Add the unit circle centered at the origin
    unit_circle = Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=1)
    ax.add_patch(unit_circle)
    
    # Add axis labels
    ax.set_xlabel("Re(z)", fontsize=12, labelpad=15)
    ax.set_ylabel("Im(z)", fontsize=12, rotation=0, labelpad=15)
    ax.xaxis.set_label_coords(0.95, 0.46)
    ax.yaxis.set_label_coords(0.54, 0.95)

    return fig, ax

def create_3d_plot(qr = Quaternion(1,0,0,0)):
    
    plt.ioff()
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    
    start_point = [0, 0, 0]
    R_base = qr.asRotationMatrix()*1.5
    draw_frame(ax, start_point, R_base)
    
    return fig, ax

def draw_frame(ax, start_point, R):
    
    x_axis = ax.quiver(*start_point, *R[:,0], arrow_length_ratio = 0.1, linewidth = 1, color='r')
    y_axis = ax.quiver(*start_point, *R[:,1], arrow_length_ratio = 0.1, linewidth = 1, color='g')
    z_axis = ax.quiver(*start_point, *R[:,2], arrow_length_ratio = 0.1, linewidth = 1, color='b')
    return x_axis, y_axis, z_axis

def create_slider(name, start_val, min_val, max_val):
    slider_width = '70%'

    slider = widgets.FloatSlider(orientation='horizontal',description=name, value=start_val, min=min_val, max=max_val, step = 0.01, layout={'width': slider_width})
    return slider

def create_angle_slider(name):
    slider_width = '70%'

    slider = widgets.FloatSlider(orientation='horizontal',description=name, value=0.0, min=-2*np.pi, max=2*np.pi, step = 0.01, layout={'width': slider_width})
    return slider
  
def create_quaternion_sliders():
    slider_width = '70%'

    angle_slider = widgets.FloatSlider(orientation='horizontal',description='angle:', value=0.0, min=-2*np.pi, max=2*np.pi, step = 0.01, layout={'width': slider_width})
    azimuth_slider = widgets.FloatSlider( orientation='horizontal', description='azimuth:', value=0.0, min=-np.pi, max=np.pi, step=0.01, layout={'width': slider_width})
    elevation_slider = widgets.FloatSlider(orientation='horizontal', description='elevation:', value=0.0, min=-np.pi/2, max=np.pi/2, step=0.01, layout={'width': slider_width})
    
    return angle_slider, azimuth_slider, elevation_slider

def create_position_sliders():
    slider_width = '70%'

    x_slider = widgets.FloatSlider(orientation='horizontal',description='x pos:', value=0.0, min=-1, max=1, step = 0.01, layout={'width': slider_width})
    y_slider = widgets.FloatSlider( orientation='horizontal', description='y pos:', value=0.0, min=-1, max=1, step=0.01, layout={'width': slider_width})
    z_slider = widgets.FloatSlider(orientation='horizontal', description='z pos:', value=0.0, min=-1, max=1, step=0.01, layout={'width': slider_width})
    
    return x_slider, y_slider, z_slider

def spherical_coordinates(azimuth, elevation):
    x = np.sin(np.pi/2 -elevation) * np.cos(azimuth)
    y = np.sin(np.pi/2 -elevation) * np.sin(azimuth)
    z = np.cos(np.pi/2 -elevation)
    vector = np.array([x, y, z])
    return vector

def create_textbox(name):
    slider_width = '70%'
    display = widgets.Text(description=name, value='', layout={'width': slider_width})
    return display

def create_quiver(ax, start_point, direction, w, c):
    quiver = ax.quiver(*start_point, *direction, arrow_length_ratio=0.1, linewidth = w, color = c)
    return quiver
    
def update_arc(arc, quaternion, p):
    t_values = np.linspace(0, 1, 50)
    arc_points = np.array([(Quaternion.slerp(Quaternion(1,0,0,0), quaternion, t)*p*Quaternion.slerp(Quaternion(1,0,0,0), quaternion, t).inverse()).getVector().flatten() for t in t_values])

    arc.set_data(arc_points[:,0], arc_points[:,1])
    arc.set_3d_properties(arc_points[:,2])
    
def draw_sphere(ax, r):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)

    light = LightSource(azdeg=0, altdeg=45)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    shaded = light.shade(z, cmap=plt.cm.Greys, vert_exag=1, blend_mode='soft')
    sphere = ax.plot_surface(x, y, z, facecolors=shaded, linewidth=0, antialiased=True, alpha=0.05)
    sphere_wire = ax.plot_wireframe(x, y, z, color='k', linewidth=0.4, alpha=0.1)
    return sphere, sphere_wire

def rpy_from_R(R):  
    # calculation of roll pitch yaw angles from rotation matrix
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    return roll, pitch, yaw

def Rx(theta):
    Rx = np.array([ [1, 0,                         0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    return Rx

def Ry(theta):
    Ry = np.array([ [np.cos(theta),  0, np.sin(theta)],
                    [0,              1,             0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    return Ry

def Rz(theta):
    Rz = np.array([ [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0,              0,             1]])
    return Rz
    
def Rot_rpy(roll, pitch, yaw):

    # Combined rotation matrix
    R = Rz(yaw)@Ry(pitch)@Rx(roll)
    return R

def show_sliders(function):
    # setup sliders to control the joint angles
    slider_width = '80%'
    w1 = widgets.FloatSlider(value = 0, min = -3.14, max = 3.14, step = 0.01, description = "joint angle 1", layout={'width': slider_width})
    w2 = widgets.FloatSlider(value = 0.6, min = -2.14, max = 2.14, step = 0.01, description = "joint angle 2", layout={'width': slider_width})
    w3 = widgets.FloatSlider(value = 0, min = -3.14, max = 3.14, step = 0.01, description = "joint angle 3", layout={'width': slider_width})
    w4 = widgets.FloatSlider(value = 1.5, min = -2.14, max = 2.14, step = 0.01, description = "joint angle 4", layout={'width': slider_width})
    w5 = widgets.FloatSlider(value = 0, min = -3.14, max = 3.14, step = 0.01, description = "joint angle 5", layout={'width': slider_width})
    w6 = widgets.FloatSlider(value = 0.6, min = -3.14, max = 3.14, step = 0.01, description = "joint angle 6", layout={'width': slider_width})
    w7 = widgets.FloatSlider(value = 0, min = -3.14, max = 3.14, step = 0.01, description = "joint angle 7", layout={'width': slider_width})
    sliders = widgets.interactive(function, q1=w1, q2=w2, q3=w3, q4=w4, q5=w5, q6=w6, q7=w7)
    display(sliders)