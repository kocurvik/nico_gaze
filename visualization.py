import numpy as np
import plotly.graph_objects as go


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