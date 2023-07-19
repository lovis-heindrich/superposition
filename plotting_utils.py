import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def line(x, xlabel="", ylabel="", title="", xticks=None, width=800, yaxis=None, hover_data=None, show_legend=True, plot=True):
    
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
    

def plot_barplot(data: list[list[float]], names: list[str], short_names = None, xlabel="", ylabel="", title="", 
                 width=1000, yaxis=None, show=True, legend=True):
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)

    fig = go.Figure()
    if short_names is None:
        short_names = names
    for i in range(len(names)):
        fig.add_trace(go.Bar(
            x=[short_names[i]],
            y=[means[i]],
            error_y=dict(
                type='data',
                array=[stds[i]],
                visible=True
            ),
            name=names[i]
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        yaxis=yaxis,
        barmode='group',
        width=width,
        showlegend=legend
    )
    
    if show:
        fig.show()
    else: 
        return fig