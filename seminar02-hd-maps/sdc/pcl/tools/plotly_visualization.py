import numpy as np
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


def plot_cloud(
        cloud: np.ndarray,
        colors=None,
        labels=None,
        marker_size=None,
        figure=None) -> go.Figure:
    if figure is None:
        figure = create_plotly_figure()
    if colors is None:
        colors = 'red'
    if marker_size is None:
        marker_size = 1
    figure.add_scatter3d(**{
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
    return figure


def apply_min_max_scaling(values, min_value=0., max_value=1.):
    assert min_value < max_value
    values = np.array(values, copy=False, dtype=np.float64)
    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    values = min_value + (max_value - min_value) * values
    return values


def convert_values_to_rgba_tuples_f64(x, cmap='Reds') -> np.ndarray:
    x = np.array(x, dtype=np.float64, copy=False)
    assert np.max(x) <= 1.0, np.max(x)
    assert np.min(x) >= 0.0, np.min(x)
    cmap = plt.get_cmap(cmap)
    return cmap(x)
