import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
from utils.haystack_utils import get_mlp_activations, get_neurons_by_layer
from transformer_lens import HookedTransformer
import scipy.stats as stats
from scipy.stats import skew, kurtosis
import pandas as pd


def plot_loss_comparison(loss_groups, group_names):
    """
    Plots a bar chart comparing different loss groups.

    Parameters:
    - loss_groups: List of lists, each containing loss values for a group.
    - group_names: List of names for each loss group.
    """

    # Function to calculate standard error
    def standard_error(data):
        return np.std(data) / np.sqrt(len(data))

    # Calculate means and standard errors for each group
    means = [np.mean(group) for group in loss_groups]
    std_errors = [standard_error(group) for group in loss_groups]
    ci_95 = [se * 1.96 for se in std_errors]

    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=group_names,
            y=means,
            error_y=dict(type='data', array=ci_95)
        )
    ])

    # Update layout
    fig.update_layout(
        title="Ablation loss comparison for closing quotation prompts '.\"'",
        xaxis_title="Ablation",
        yaxis_title="Loss",
        showlegend=False,
        width=600
    )

    fig.show()


def line(x: list[float], xlabel="", ylabel="", title="", xticks=None, width=800, yaxis=None, hover_data=None, show_legend=True, plot=True):
    
    # Avoid empty plot when x contains a single element
    if len(x) > 1:
        fig = px.line(x, title=title)
    else:
        fig = px.scatter(x, title=title)

    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, width=width, showlegend=show_legend)
    if xticks != None:
        fig.update_layout(
            xaxis = dict(
            tickmode = 'array',
            tickvals = [i for i in range(len(xticks))],
            ticktext = xticks,
            range=[-0.2, len(xticks)-0.8] 
            ),
            yaxis=yaxis,
        )
    
    #fig.update_yaxes(range=[3.45, 3.85])
    if hover_data != None:
        fig.update(data=[{'customdata': hover_data, 'hovertemplate': "Loss: %{y:.4f} (+%{customdata:.2f}%)"}])
    if plot:
        fig.show()
    else:
        return fig

def multiple_line(x: list[list[float]], names: list[str], xlabel="", ylabel="", title="", xticks=None, width=800, yaxis=None, hover_data=None, show_legend=True, plot=True):
    if len(x) != len(names):
        raise ValueError("Length of 'x' and 'names' must be the same.")

    # Create a DataFrame for Plotly
    data = []
    for line_idx, line in enumerate(x):
        for point_idx, point in enumerate(line):
            data.append({'x': point_idx, 'y': point, 'line': names[line_idx]})
    df = pd.DataFrame(data)

    # Create the figure
    fig = px.line(df, x='x', y='y', color='line', title=title)

    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, width=width, showlegend=show_legend)
    if xticks is not None:
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[i for i in range(len(xticks))],
                ticktext=xticks,
                range=[-0.2, len(xticks)-0.8]
            ),
            yaxis=yaxis,
        )

    if hover_data is not None:
        fig.update_traces(customdata=hover_data, hovertemplate="Value: %{y:.4f} (+%{customdata:.2f}%)")

    if plot:
        fig.show()
    else:
        return fig
    

def plot_barplot(data: list[list[float]], names: list[str], short_names = None, xlabel="", ylabel="", title="", 
                 width=1000, yaxis=None, show=True, legend=True, yrange=None, confidence_interval=False):
    means = np.mean(data, axis=1)
    if confidence_interval:
        errors = [stats.sem(d)*stats.t.ppf((1+0.95)/2., len(d)-1) for d in data]
    else:
        errors = np.std(data, axis=1)

    fig = go.Figure()
    if short_names is None:
        short_names = names
    if len(data[0]) > 1:
        for i in range(len(names)):
            fig.add_trace(go.Bar(
                x=[short_names[i]],
                y=[means[i]],
                error_y=dict(
                    type='data',
                    array=[errors[i]],
                    visible=True
                ),
                name=names[i]
            ))
    else:
        for i in range(len(names)):
            fig.add_trace(go.Bar(
                x=[short_names[i]],
                y=[means[i]],
                name=names[i]
            ))
    
    
    if yrange is not None:
        fig.update_yaxes(range=yrange)
        
    fig.update_layout(
        title=title,
        yaxis=yaxis,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        barmode='group',
        width=width,
        showlegend=legend
    )

    
    if show:
        fig.show()
    else: 
        return fig


def color_binned_histogram(data, ranges, labels, title):
    # Check that ranges and labels are of the same length
    if len(ranges) != len(labels):
        raise ValueError("Length of ranges and labels should be the same")

    fig = go.Figure()

    # Plot data in ranges with specific colors
    for r, label, color in zip(ranges, labels, qualitative.Plotly):
        hist_data = [i for i in data if r[0] <= i < r[1]]
        fig.add_trace(go.Histogram(x=hist_data, 
                                    name=label,
                                    marker_color=color))
    
    # Plot data outside ranges with a gray color
    out_range_data = [i for i in data if not any(r[0] <= i < r[1] for r in ranges)]
    fig.add_trace(go.Histogram(x=out_range_data, 
                                name='Outside Ranges',
                                marker_color='gray'))

    fig.update_layout(barmode='stack',
                      xaxis_title='Value',
                      yaxis_title='Count',
                      title=title,
                      width=1200)
    fig.show()


def plot_neuron_acts(
        model: HookedTransformer, data: list[str], neurons: list[tuple[int, int]], disable_tqdm=True, 
        width=700, range_x=None, hook_pre=False
) -> None:
    '''Plot activation histograms for each neuron specified'''
    neurons_by_layer = get_neurons_by_layer(neurons)
    for layer, layer_neurons in neurons_by_layer.items():
        acts = get_mlp_activations(data, layer, model, neurons=torch.tensor(layer_neurons), mean=False, 
                                   disable_tqdm=disable_tqdm, hook_pre=hook_pre).cpu()
        for i, neuron in enumerate(layer_neurons):
            fig = px.histogram(acts[:, i], title=f"L{layer}N{neuron}", width=width)
            fig.update_xaxes(range=range_x)
            fig.show()


def get_neuron_moments(
        model: HookedTransformer, data: list[str], neurons: list[tuple[int, int]], disable_tqdm=True,
        hook_pre=False
) -> pd.DataFrame:
    '''Plot activation histograms for each neuron specified'''
    neurons_by_layer = get_neurons_by_layer(neurons)
    neuron_moments = []
    for layer, layer_neurons in neurons_by_layer.items():
        acts = get_mlp_activations(data, layer, model, neurons=torch.tensor(layer_neurons), mean=False, 
                                   disable_tqdm=disable_tqdm, hook_pre=hook_pre).cpu()
        for i, neuron in enumerate(layer_neurons):
            tensor_numpy = acts[:, i].numpy()
            neuron_moments.append((layer, neuron, skew(tensor_numpy), kurtosis(tensor_numpy)))
    return pd.DataFrame(neuron_moments, columns=['layer', 'neuron', 'skew', 'kurtosis'])


def plot_square_heatmap(data, labels, title=""):
    """
    Plots a square heatmap using Plotly with 
    the diagonal going from top left to bottom right.

    :param data: A square numpy array or PyTorch tensor.
    :param labels: List of labels for the x and y axis.
    """
    # Convert PyTorch tensor to numpy array if necessary
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    labels = [f"{direction}" for direction in labels]
    # Ensuring the data is square
    assert data.shape[0] == data.shape[1], "Data must be a square matrix"

    # Setting the lower triangle to 0 and flipping the matrix for correct orientation
    data = np.flipud(np.triu(data))

    # Creating the heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=labels,
        y=labels[::-1],  # Reverse the y-axis labels to match the flipped data
        colorscale="amp"
    ))

    # Updating the layout
    fig.update_layout(
        title=title,
        xaxis_nticks=len(labels),
        yaxis_nticks=len(labels),
        autosize=False,    # This allows us to set a specific width and height
        width=600,         # Width of the figure in pixels
        height=600
    )

    return fig
