import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics
import haystack_utils
import matplotlib.pyplot as plt
import re
from IPython.display import display, HTML
import numpy as np
from torch.distributions import Normal
import plotly.graph_objects as go


def run_single_neuron_lr(layer, neuron, german_activations, english_activations, num_samples=5000, ):
    """For German context neurons"""
    # Check accuracy of logistic regression
    A = torch.concat([german_activations[layer][:num_samples, neuron], english_activations[layer][:num_samples, neuron]]).view(-1, 1).cpu().numpy()
    y = torch.concat([torch.ones(num_samples), torch.zeros(num_samples)]).cpu().numpy()
    A_train, A_test, y_train, y_test = train_test_split(A, y, test_size=0.2)
    lr_model = LogisticRegression()
    lr_model.fit(A_train, y_train)
    test_acc = lr_model.score(A_test, y_test)
    train_acc = lr_model.score(A_train, y_train)
    f1 = sklearn.metrics.f1_score(y_test, lr_model.predict(A_test))
    return train_acc, test_acc, f1
    

def get_neuron_accuracy(layer, neuron, german_activations, english_activations, plot=False, print_f1s=True):
    """For German context neurons"""
    mean_english_activation = english_activations[layer][:,neuron].mean()
    mean_german_activation = german_activations[layer][:,neuron].mean()
    
    if plot:
        haystack_utils.two_histogram(english_activations[layer][:,neuron], german_activations[layer][:,neuron], "English", "German", "Activation", "Frequency", f"L{layer}N{neuron} activations on English vs German text")
    train_acc, test_acc, f1 = run_single_neuron_lr(layer, neuron, german_activations=german_activations, english_activations=english_activations)
    if print_f1s:
        print(f"\nL{layer}N{neuron}: F1={f1:.2f}, Train acc={train_acc:.2f}, and test acc={test_acc:.2f}")
        print(f"Mean activation English={mean_english_activation:.2f}, German={mean_german_activation:.2f}")
    return f1


def ablation_effect(model, data, fwd_hooks):
    """Full ablation accuracy"""
    original_losses = []
    ablated_losses = []
    batch_size = 50
    for i in range(4):
        original_losses.append(model(data[i * batch_size:i * batch_size + 50], return_type='loss').cpu())
        with model.hooks(fwd_hooks):
            ablated_losses.append(model(data[i * batch_size:i * batch_size + 50], return_type='loss').cpu())

    original_loss = np.mean(original_losses)
    ablated_loss = np.mean(ablated_losses)

    print(original_loss, ablated_loss)
    print(f'{(ablated_loss - original_loss) / original_loss * 100:2f}% loss increase')


def color_strings_by_value(strings: list[str], color_values: list[float], max_value: float=None, additional_measures: list[list[float]] | None = None, additional_measure_names: list[str] | None = None, peak_names = None):
    """
    Magic GPT function that prints a string as HTML and colors it according to a list of color values. 
    Color values are normalized to the max value preserving the sign.
    """

    def normalize(values, max_value=None, min_value=None):
        if max_value is None:
            max_value = max(values)
        if min_value is None:
            min_value = min(values)
        min_value = abs(min_value)
        normalized = [(value / max_value if value > 0 else value / min_value) for value in values]
        return normalized
    
    # Normalize color values
    normalized_values = normalize(color_values, max_value, max_value)

    html = "<div>"
    cmap = cmap=plt.cm.Pastel1

    for i in range(len(strings)):
        normalized_color = normalized_values[i].cpu()
        
        # Use colormap to get RGB values
        red, green, blue, _ = cmap(normalized_color)

        # Scale RGB values to 0-255
        red, green, blue = [int(255*v) for v in (red, green, blue)]
        
        # Calculate luminance to determine if text should be black
        luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
        
        # Determine text color based on background luminance
        text_color = "black" if luminance > 0.5 else "white"

        visible_string = re.sub(r'\s+', '_', strings[i])
        
        html += f'<span style="background-color: rgb({red}, {green}, {blue}); color: {text_color}; padding: 2px;" '
        if peak_names is not None:
            html += f'title="{peak_names[int(color_values[i])]}'
        else:
            html += f'title="Difference: {color_values[i]:.4f}' 
        if additional_measure_names is not None:
            for j in range(len(additional_measure_names)):
                html += f', {additional_measure_names[j]}: {additional_measures[j][i]:.4f}'
        html += f'">{visible_string}</span>'
    html += '</div>'

    # Print the HTML in Jupyter Notebook
    display(HTML(html))


def print_prompt(prompt, model, interest_measure, layer, neuron, names=["Inactive", "Peak 1", "Peak 2", "Unknown Peak"]):
    str_tokens = model.to_str_tokens(model.to_tokens(prompt))
    _, cache = model.run_with_cache(prompt)
    activations = cache["post", layer][0, :, neuron]
    interest = interest_measure(activations)
    color_strings_by_value(str_tokens, interest, peak_names=names)


def em_gmm(data, n_clusters, n_iterations=100, eps=1e-8):
    # Initialize parameters
    mu = torch.randn(n_clusters)
    sigma = torch.ones(n_clusters)
    weights = torch.ones(n_clusters) / n_clusters
    for _ in range(n_iterations):
        # E-Step: compute responsibilities
        norms = [Normal(mu[i], sigma[i]) for i in range(n_clusters)]
        responsibilities = torch.stack([w * norm.log_prob(data).exp() for w, norm in zip(weights, norms)], dim=-1)
        responsibilities /= (responsibilities.sum(dim=-1, keepdim=True) + eps)
        # M-Step: update parameters
        weights = responsibilities.mean(dim=0) + eps
        for i in range(n_clusters):
            weight_resp = responsibilities[:, i] / (weights[i] + eps)
            mu[i] = (data * weight_resp).sum() / (weight_resp.sum() + eps)
            sigma[i] = torch.sqrt((weight_resp * (data - mu[i])**2).sum() / (weight_resp.sum() + eps))
    return mu, sigma, weights


def plot_gaussians_with_histogram(data, mu, sigma, weights, x_range=(-10, 10), num_points=1000, nbins=50):
    x = torch.linspace(*x_range, num_points)
    fig = go.Figure()
    # Add histogram
    fig.add_trace(go.Histogram(x=data.numpy(), nbinsx=nbins, name='Data'))
    # Calculate bin width for normalization
    bin_width = (data.max() - data.min()) / nbins
    # Add gaussians
    for i in range(len(mu)):
        y = weights[i] * Normal(mu[i], sigma[i]).log_prob(x).exp() * len(data) * bin_width
        fig.add_trace(go.Scatter(x=x.numpy(), y=y.numpy(), mode='lines', name=f'Normal {i+1}'))
    fig.update_layout(barmode='overlay', title='Normal Distributions Overlaid with Data Histogram',
                      xaxis_title='x', yaxis_title='Count')
    fig.data[0].marker.opacity = 0.5  # make the histogram semi-transparent
    fig.show()
    

def plot_combined_gaussian_with_histogram(data, mu, sigma, weights, x_range=(-10, 10), num_points=1000, nbins=50):
    x = torch.linspace(*x_range, num_points)
    fig = go.Figure()
    # Add histogram
    fig.add_trace(go.Histogram(x=data.numpy(), nbinsx=nbins, name='Data'))
    # Calculate bin width for normalization
    bin_width = (data.max() - data.min()) / nbins
    # Add gaussians
    for i in range(len(mu)):
        y = weights[i] * Normal(mu[i], sigma[i]).log_prob(x).exp() * len(data) * bin_width
        fig.add_trace(go.Scatter(x=x.numpy(), y=y.numpy(), mode='lines', name=f'Normal {i+1}'))
    # Combined Gaussian
    y_comb = sum(weights[i] * Normal(mu[i], sigma[i]).log_prob(x).exp() for i in range(len(mu))) * len(data) * bin_width
    fig.add_trace(go.Scatter(x=x.numpy(), y=y_comb.numpy(), mode='lines', name='Mixture of Gaussians'))
    fig.update_layout(barmode='overlay', title='Normal Distributions Overlaid with Data Histogram',
                      xaxis_title='x', yaxis_title='Count')
    fig.data[0].marker.opacity = 0.5  # make the histogram semi-transparent
    fig.show()


def gmm_log_likelihood(data, mu, sigma, weights):
    norms = [Normal(mu[i], sigma[i]) for i in range(len(mu))]
    log_likelihoods = torch.stack([torch.log(w) + norm.log_prob(data) for w, norm in zip(weights, norms)], dim=-1)
    log_likelihood = torch.logsumexp(log_likelihoods, dim=-1)
    return log_likelihood.mean()
