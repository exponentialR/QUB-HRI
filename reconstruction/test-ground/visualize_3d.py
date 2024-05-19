import h5py
import numpy as np
import plotly.graph_objects as go


def load_3d_points(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        points_3d = {key: f[key][:] for key in f.keys()}
    return points_3d


def prepare_plotly_data(points_3d, landmark_name):
    x, y, z = [], [], []
    for frame in points_3d[landmark_name]:
        for point in frame:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
    return x, y, z


def create_3d_scatter_plot(x, y, z, title):
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=z,  # Set color to the z-coordinates to give some sense of depth
            colorscale='Viridis',  # Choose a colorscale
            opacity=0.8
        )
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(r=10, l=10, b=10, t=40)
    )

    fig.show()


# Load the 3D points from the output HDF5 file
points_3d = load_3d_points('output.hdf5')

# Example: Prepare data and create a 3D scatter plot for face landmarks
x, y, z = prepare_plotly_data(points_3d, 'face_landmarks')
print(f'Number of face landmarks: {len(x)}')
create_3d_scatter_plot(x, y, z, '3D Face Landmarks')
