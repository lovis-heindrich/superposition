import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
import plotting_utils
import plotly.graph_objects as go

st.set_page_config(page_title="AND-Neurons", page_icon="📊")
st.sidebar.success("Select an analysis above.")

st.title("MLP5 AND-Neurons Analysis")

#@st.cache_data
def load_data():
    path = Path(__file__).parent / f"../data/and_neurons/"
    df_logits = pd.read_csv(path / f"and_conditions_logits.csv")
    df_loss = pd.read_csv(path / "and_conditions_loss.csv")
    return df_logits, df_loss

df_logits, df_loss = load_data()

st.markdown("""
            ### Looking for non-linearities

            We check if the whole model computes a non-linear function by comparing the correct token's logit or the loss with and without all features being active.
            """)

data_select = st.sidebar.selectbox(label="Select which value to compare", 
             options=["Loss", "Correct Token Logit"], index=1)
df = df_loss if data_select == "Loss" else df_logits
df = df.rename(columns = {"Unnamed: 0": 'Value'})
st.dataframe(df, hide_index=True, height=500, width=500)

# if data_select == "Change in loss":
#     st.latex(r'''\text{Current token: }(NYN-YYY)-((NYN-YYN)+(NYN-NYY))''')
# else:
#     st.latex(r'''\text{Current token: }(YYY-NYN)-((YYN-NYN)+(NYY-NYN))''')

# if data_select == "Change in loss":
#     st.latex(r'''\text{Grouped tokens: }(NNN-YYY)-((NNN-YYN)+(NNN-NNY))''')
# else:
#     st.latex(r'''\text{Current tokens: }(YYY-NNN)-((YYN-NNN)+(NNY-NNN))''')

# if data_select == "Change in loss":
#     st.latex(r'''\text{Single features: }(NNN-YYY)-((NNN-YNN)+(NNN-NYN)+(NNN-NNY))''')
# else:
#     st.latex(r'''\text{Single features: }(YYY-NNN)-((YNN-NNN)+(NYN-NNN)+(NNY-NNN))''')

# if data_select == "Change in loss":
#     st.latex(r'''\text{Two features: }(NNN-YYY)-((NNN-YYN)+(NNN-YNY)+(NNN-NYY))/2''')
# else:
#     st.latex(r'''\text{Two features: }(YYY-NNN)-((YYN-NNN)+(YNY-NNN)+(NYY-NNN))/2''')


# and_condition_data = and_conditions[option][data_select_index]
# data_indices = ["current_token_diffs", "previous_token_diffs", "context_neuron_diffs", "individiual_features_diffs", "two_features_diffs"]
# plot_names = ["Fix Current", "Fix Previous", "Fix Context", "Single features", "Two features"]
# plot_data = [[and_condition_data[index]] for index in data_indices]
# plot = plotting_utils.plot_barplot(plot_data, plot_names, ylabel=data_select,show=False, legend=False, width=600)
# st.plotly_chart(plot)

features = ["current_token_diffs",
        "previous_token_diffs",
        "context_neuron_diffs",
        "individiual_features_diffs",
        "two_features_diffs"]

xlabels = ["Fix Current", "Fix Previous", "Fix Context", "Single features", "Two features"]

def create_grouped_barplot(df, values, xlables):
    # df_long = df[df['Value'].isin(values)].melt(id_vars='Value', var_name='Category', value_name='Number')
    # fig = px.bar(df_long, x='Value', y='Number', color='Category', barmode='group', title='Grouped Barplots for Selected Values')
    
    df_long = df[df['Value'].isin(values)].melt(id_vars='Value', var_name='Category', value_name='Number')
    df_long['Value'] = df_long['Value'].replace(dict(zip(values, xlabels)))
    fig = px.bar(df_long, x='Value', y='Number', color='Category', barmode='group', title='Grouped Barplots for Selected Values')
    fig.update_layout(
        xaxis_title="",
        yaxis_title=data_select,)
    
    return fig

# Test the function
fig = create_grouped_barplot(df, features, xlabels)
st.plotly_chart(fig)