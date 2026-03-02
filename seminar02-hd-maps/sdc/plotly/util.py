import numpy as np
import typing as T
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def create_plotly_figure(height: int = 800, bgcolor=None) -> go.Figure:
    equal_aspect_ratio_layout = dict(
        margin={
            'l': 0,
            'r': 0,
            'b': 0,
            't': 0
        },
        scene=dict(
            xaxis=dict(backgroundcolor=bgcolor),
            yaxis=dict(backgroundcolor=bgcolor),
            zaxis=dict(backgroundcolor=bgcolor),
            aspectmode='data',
            bgcolor=bgcolor),
        height=height)
    return go.Figure(layout=equal_aspect_ratio_layout)


def add_point_cloud(
    fig: go.Figure,
    cloud: np.ndarray,
    colors=None,
    labels=None,
    marker_size=None,
) -> go.Figure:
    assert isinstance(fig, go.Figure)
    if colors is None:
        colors = 'red'
    if marker_size is None:
        marker_size = 1
    fig.add_scatter3d(**{
        'x': cloud[:, 0],
        'y': cloud[:, 1],
        'z': cloud[:, 2],
        'mode': 'markers',
        'marker': {
            'size': marker_size,
            'color': colors,
        },
        'text': labels,
    })
    return fig


def apply_min_max_scaling(values, min_value=0., max_value=1.):
    assert min_value < max_value
    values = np.array(values, copy=True, dtype=np.float64)
    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    values = min_value + (max_value - min_value) * values
    return values


def convert_values_to_rgba_tuples_f64(x, cmap='Reds') -> np.ndarray:
    x = np.array(x, dtype=np.float64, copy=False)
    assert np.max(x) <= 1.0, np.max(x)
    assert np.min(x) >= 0.0, np.min(x)
    cmap = plt.get_cmap(cmap)
    return cmap(x)


def add_axis(fig, vector, origin, length=None, alpha=0.1, color=None, label=None):
    """
    :param color: "rgb(84,48,5)"
    """
    vector = np.array(vector, copy=False)
    origin = np.array(origin, copy=False)
    if length is not None:
        assert length > 0
        vector /= np.linalg.norm(vector)
        vector *= length
    vector_end = origin + vector
    fig.add_scatter3d(
        x=[origin[0], vector_end[0]],
        y=[origin[1], vector_end[1]],
        z=[origin[2], vector_end[2]],
        marker=dict(size=1, color=color),
        text=label,
        mode="lines+text",
        line=dict(color=color, width=6),
    )
    fig.add_cone(
        x=[vector_end[0]],
        y=[vector_end[1]],
        z=[vector_end[2]],
        u=[alpha * vector[0]],
        v=[alpha * vector[1]],
        w=[alpha * vector[2]],
    )


def add_axes(fig, transform_matrix, **kwargs):
    assert transform_matrix.shape == (4, 4)
    R = transform_matrix[:3, :3]
    t = transform_matrix[:3, 3]
    colors = ["red", "green", "blue"]
    for axis_idx in range(3):
        add_axis(fig, vector=R[:, axis_idx], origin=t, color=colors[axis_idx], **kwargs)


def add_poses(fig, poses: T.List[np.ndarray], **kwargs):
    for pose in poses:
        add_axes(fig, pose, **kwargs)


def add_positions(
    fig: go.Figure,
    positions: T.List[np.ndarray],
    line_color: str = "red",
    line_width: float = 2,
    marker_size: float = 5
):
    fig.add_scatter3d(**dict(
        x=[position[0] for position in positions],
        y=[position[1] for position in positions],
        z=[position[2] for position in positions],
        mode='lines+markers',
        line=dict(color=line_color, width=line_width),
        marker=dict(size=marker_size),
    ))
