import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
import plotting_utils

st.set_page_config(page_title="AND-Neurons", page_icon="📊")
st.sidebar.success("Select an analysis above.")

st.title("MLP5 AND-Neurons Analysis")

tokens = ["orschlägen", " häufig", " beweglich"]

option = st.sidebar.selectbox(
    'Select the trigram to analyze',
    tokens, index=0)

run_select = st.sidebar.selectbox(
    "Select run", options=["Run 1", "Run 2", "Run 3"], index=0
)

if run_select == "Run 1":
    file_name_append = "_0"
elif run_select == "Run 2":
    file_name_append = "_1000"
else:
    file_name_append = "_2000"

#@st.cache_data
def load_data():
    path = Path(__file__).parent / f"../data/and_neurons/"
    dfs = pd.read_pickle(path / f"activation_dfs{file_name_append}.pkl")
    with open(path / f"ablation_losses{file_name_append}.json", "r") as f:
        ablation_losses = json.load(f)
    return dfs, ablation_losses

dfs, ablation_losses = load_data()

hook_select = st.sidebar.selectbox(label="Select between pre-gelu activations and post-gelu activations", options=["hook_pre", "hook_post"], index=1)
scale_select = st.sidebar.selectbox(label="Select between scaled activation or original activation values", options=["Scaled", "Unscaled"], index=1)

df = dfs[option][hook_select][scale_select]

st.markdown("""
            ### Neuron activation data
            
            The first 8 columns show average pre-gelu activations for different prompt types. The letters encode three binary variables (Yes/No): 
            1. The first trigram token is present (Y), or replaced with random tokens (N) 
            2. The second trigram token is present (Y), or replaced with random tokens (N)
            3. The context token activates normally (Y), or the context neuron is ablated (N)

            """)


with st.expander("Show AND condition formulas"):
    st.latex(r'''\text{Fix Current:}(YYY-NYN)-((YYN-NYN)+(NYY-NYN))''')
    st.latex(r'''\text{Fix Previous:}(YYY-YNN)-((YYN-YNN)+(YNY-YNN))''')
    st.latex(r'''\text{Fix Context:}(YYY-NNY)-((YNY-NNY)+(NYY-NNY))''')
    st.latex(r'''\text{Single feature:}(YYY-NNN)-((YNN-NNN)+(NYN-NNN)+(NNY-NNN))''')
    st.latex(r'''\text{Two features:}(YYY-NNN)-((YYN-NNN)+(YNY-NNN)+(NYY-NNN))/2''')


all_columns = df.columns.tolist()
always_show = ['NNN', 'NNY', 'NYN', 'NYY', 'YNN', 'YNY', 'YYN', 'YYY']
select_names = [name for name in all_columns if name not in always_show]
column_select = st.multiselect(
    'Select visible columns',
    select_names,
    [])


display_df = df.copy()
for column in select_names:
    if column not in column_select:
        display_df = display_df.drop(columns=[column])

st.dataframe(display_df.round(2))

st.markdown("""
            ### Comparing neuron-wise ablation loss increase

            The AND features are used to highlight neurons in a scatter plot displaying their loss increase.
            """)

highlight_mode = st.selectbox(label="Select the type of AND neurons to highlight",
                              options=["Two Features", "Single Features", "Current Token", "Previous Token", "Context Neuron"], index=0)

negative_and_neurons = st.checkbox("Show negative AND neurons", value=False)
if negative_and_neurons:
    color = highlight_mode + " (NEG AND)"
else:
    color = highlight_mode + " (AND)"
plot = px.scatter(df, y="AblationLossIncrease", color=color,
                  color_discrete_sequence=["grey", "red", "blue"], labels=highlight_mode)

plot.update_layout(
    title="Loss increase when patching individual MLP5 neurons",
    yaxis_title="Ablation loss increase"
)
st.plotly_chart(plot)

# Ablation change for pos, neg, and strongly activating neurons

st.markdown("""
            ### Comparing ablation loss increase for groups of neurons

            Uses AND features to select sets of neurons and compares their loss increase.
            """)
#[option][hook_name][include_mode]
keys = list(ablation_losses[option][hook_select][scale_select].keys())
select_mode = st.selectbox(label="Select whether to use all AND neurons or only the neurons where (YYY>others)",
                              options=keys, index=0)


names = list(ablation_losses[option][hook_select][scale_select][select_mode].keys())
short_names = [name.split(" ")[0] for name in names]
loss_values = [[ablation_losses[option][hook_select][scale_select][select_mode][name]] for name in names]
plot = plotting_utils.plot_barplot(loss_values, names,
                            short_names=short_names, ylabel="Last token loss",
                            title=f"Loss increase when patching groups of neurons (ablation mode: YYN)",
                            width=750, show=False)
st.plotly_chart(plot)