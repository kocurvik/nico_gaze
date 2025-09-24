import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns


def plot_angle_distribution(name, vectors):
    """
    Draws a 2D distribution (heatmap) over the space of yaw and pitch angles
    given a list of 3D direction vectors.

    Args:
        vectors (list or numpy.ndarray): A list or array of 3D vectors.
                                         Each vector should be a list, tuple,
                                         or array of length 3, e.g., [[x, y, z], ...].
    """
    # Convert input list to a NumPy array for efficient vector operations
    vectors = np.array(vectors)

    # Check if the input is valid
    if vectors.shape[1] != 3:
        raise ValueError("Input vectors must be 3D (shape must be (N, 3)).")

    yaw_deg, pitch_deg = yaw_pitch_from_direction(vectors)

    # Create the 2D histogram (heatmap) of the angle distribution
    plt.figure(figsize=(7, 3))

    sns.kdeplot(
        x=yaw_deg,
        y=pitch_deg,
        bw_adjust=.5,
        fill=True,
        cmap='viridis',
        levels=10  # Use more levels for a more detailed contour map
    )

    large_size = 20
    small_size = 16

    plt.xlabel('Yaw Angle ($^\circ$)', fontsize=large_size)
    plt.ylabel('Pitch Angle ($^\circ$)', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)

    plt.xlim(-90, 90)
    plt.ylim(-100, 0)

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(f'figs/distribution_{name}.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def yaw_pitch_from_direction(vectors):
    # Extract components for clarity
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]
    yaw_rad = np.arctan2(x, -z)
    pitch_rad = np.arcsin(-y)
    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)
    return yaw_deg, pitch_deg

def direction_from_yaw_pitch(yaw, pitch):
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    if np.isscalar(yaw) :
        direction = np.zeros(3)
    else:
        direction = np.zeros([len(yaw), 3])
    direction[..., 0] = np.cos(pitch) * np.sin(yaw)
    direction[..., 1] = -np.sin(pitch)
    direction[..., 2] = -np.cos(pitch) * np.cos(yaw)
    return direction


def get_corrected_gaze(gaze_dir, head_point):

    yaw_est, pitch_est = yaw_pitch_from_direction(gaze_dir)
    yaw_head, pitch_head = yaw_pitch_from_direction(-head_point / np.linalg.norm(head_point))

    yaw_comibned = yaw_est + yaw_head
    pitch_combined = pitch_est + pitch_head

    return direction_from_yaw_pitch(yaw_comibned, pitch_combined)

def visualize_plane(fig, plane, boundary_points, middle_points):
    A, B, C, D = plane

    all_points = np.vstack((middle_points[:, :3], boundary_points))
    minX = np.min(all_points[:, 0]) 
    maxX = np.max(all_points[:, 0]) 
    minY = np.min(all_points[:, 1]) 
    maxY = np.max(all_points[:, 1]) 

    x = np.linspace(minX, maxX, 5)
    y = np.linspace(minY, maxY, 5)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values based on the plane equation
    Z = (-D - A * X - B * Y) / C

    colorscale = [[0, 'lightblue'], [1, 'lightblue']]

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=colorscale,
        opacity=0.6,
        name='Chessboard Plane',
        hoverinfo='skip',
        showscale=False
    ))
    return fig


def create_base_figure(title='Plane visualization'):
    theta = np.radians(100)  # Rotation around the Z-axis
    phi = np.radians(100)  # Rotation around the X-axis
    x, y, z = 0, 0, 1

    # Rotation around the Z-axis
    x_z = x * np.cos(theta) - y * np.sin(theta)
    y_z = x * np.sin(theta) + y * np.cos(theta)

    # Rotation around the X-axis
    y_x = y_z * np.cos(phi) - z * np.sin(phi)
    z_x = y_z * np.sin(phi) + z * np.cos(phi)

    scale = 2

    fig = go.Figure()
    fig.update_layout(title=title,
            scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
                camera=dict(
                    eye=dict(x=x_z*scale, y=y_x*scale, z=z_x*scale),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0)
                )),
            showlegend=True
            )
    return fig

def plot_3D_point(fig, point, name="3D Point", color="red", mode='markers'):
    x = point[0]
    y = point[1]
    z = point[2]
    colorscale = [[0, color], [1, color]]

    # Adding the scatter plot to the figure
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='markers',
        marker=dict(size=6, color=color, colorscale=colorscale, opacity=0.8),
        name=name
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        scene_aspectmode='auto'
    )
    return fig

def plot_3d_points(fig, points, name="3D Points", color="red", mode='markers'):
    if points.ndim == 3:
        points = points.reshape(-1, points.shape[-1])

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    colorscale = [[0, color], [1, color]]

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=6, color=z, colorscale=colorscale, opacity=0.8),
        name=name
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        scene_aspectmode='auto'
    )
    return fig

def plot_line(fig, point_start, point_end, name="line", color="blue"):
    x_coords = [point_start[0], point_end[0]]
    y_coords = [point_start[1], point_end[1]]
    z_coords = [point_start[2], point_end[2]]

    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        line=dict(
            color=color,
            width=7
        ),
        name=name
    ))
    return fig

if __name__ == '__main__':
    f = create_base_figure()
    f.show()
    print(f.layout['scene']['camera']['eye'])