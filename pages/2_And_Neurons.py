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

option = st.selectbox(
    'Select the trigram to analyze',
    tokens, index=0)

#@st.cache_data
def load_data(tokens):
    path = Path(__file__).parent / f"../data/and_neurons/"
    df = pd.read_pickle(path / f"df_{tokens.strip()}.pkl")
    with open(path / "ablation_losses.json", "r") as f:
        ablation_losses = json.load(f)
    with open(path / "and_conditions.json", "r") as f:
        and_conditions = json.load(f)
    return df, ablation_losses, and_conditions

df, ablation_losses, and_conditions = load_data(option)

# # Look for neurons that consistently respond to all 3 directions
# df["Boosted"] = (df["YNN"]>df["NNN"])&(df["NYN"]>df["NNN"])&(df["NNY"]>df["NNN"])&\
#                 (df["YYY"]>df["YNN"])&(df["YYY"]>df["NYN"])&(df["YYY"]>df["NNY"])&\
#                 (df["YYY"]>0) # Negative boosts don't matter

# df["Deboosted"] = (df["YNN"]<df["NNN"])&(df["NYN"]<df["NNN"])&(df["NNY"]<df["NNN"])&\
#                 (df["YYY"]<df["YNN"])&(df["YYY"]<df["NYN"])&(df["YYY"]<df["NNY"])&\
#                 (df["NNN"]>0) # Deboosting negative things doesn't matter


st.markdown("""
            ### Neuron activation data
            
            The first 8 columns show average pre-gelu activations for different prompt types. The letters encode three binary variables (Yes/No): 
            1. The first trigram token is present (Y), or replaced with random tokens (N) 
            2. The second trigram token is present (Y), or replaced with random tokens (N)
            3. The context token activates normally (Y), or the context neuron is ablated (N)

            We compute 4 different AND conditions in which we compare the neuron's activation with all features being present (YYY) to different combinations of features being absent:
            - Current token: the current token is always present
            - Grouped tokens: current and previous token are grouped together
            - Single features: all three input features appear individually
            - Two features: two input features appear together
            """)

st.latex(r'''\text{Current token: }(YYY-NYN)-((YYN-NYN)+(NYY-NYN))''')
st.latex(r'''\text{Current tokens: }(YYY-NNN)-((YYN-NNN)+(NNY-NNN))''')
st.latex(r'''\text{Single features: }(YYY-NNN)-((YNN-NNN)+(NYN-NNN)+(NNY-NNN))''')
st.latex(r'''\text{Two features: }(YYY-NNN)-((YYN-NNN)+(YNY-NNN)+(NYY-NNN))/2''')

show_losses = st.checkbox("Show ablation loss changes", value=False)
show_cosine_sims = st.checkbox("Show cosine similarities with token / context directions", value=False)

display_df = df.copy()
if not show_losses:
    display_df = display_df.drop(columns=["FullAblationLossIncrease", "ContextAblationLossIncrease"])
if not show_cosine_sims:
    display_df = display_df.drop(columns=["PrevTokenSim", "CurrTokenSim", "ContextSim"])


st.dataframe(display_df.round(2))

st.markdown("""
            ### Looking for non-linearities

            We check if the whole model computes a non-linear function by comparing the correct token's logit or the loss with and without all features being active.
            """)

data_select = st.selectbox(label="Select which value to compare", 
             options=["Change in loss", "Change in correct token logit"], index=1)
data_select_index = "loss" if data_select == "Change in loss" else "logits"

if data_select == "Change in loss":
    st.latex(r'''\text{Current token: }(NYN-YYY)-((NYN-YYN)+(NYN-NYY))''')
else:
    st.latex(r'''\text{Current token: }(YYY-NYN)-((YYN-NYN)+(NYY-NYN))''')

if data_select == "Change in loss":
    st.latex(r'''\text{Grouped tokens: }(NNN-YYY)-((NNN-YYN)+(NNN-NNY))''')
else:
    st.latex(r'''\text{Current tokens: }(YYY-NNN)-((YYN-NNN)+(NNY-NNN))''')

if data_select == "Change in loss":
    st.latex(r'''\text{Single features: }(NNN-YYY)-((NNN-YNN)+(NNN-NYN)+(NNN-NNY))''')
else:
    st.latex(r'''\text{Single features: }(YYY-NNN)-((YNN-NNN)+(NYN-NNN)+(NNY-NNN))''')

if data_select == "Change in loss":
    st.latex(r'''\text{Two features: }(NNN-YYY)-((NNN-YYN)+(NNN-YNY)+(NNN-NYY))/2''')
else:
    st.latex(r'''\text{Two features: }(YYY-NNN)-((YYN-NNN)+(YNY-NNN)+(NYY-NNN))/2''')


and_condition_data = and_conditions[option][data_select_index]
data_indices = ["current_token_diffs", "grouped_token_diffs", "individiual_features_diffs", "two_features_diffs"]
plot_names = ["Current token", "Grouped tokens", "Single features", "Two features"]
plot_data = [[and_condition_data[index]] for index in data_indices]
plot = plotting_utils.plot_barplot(plot_data, plot_names, ylabel=data_select,show=False, legend=False, width=600)
st.plotly_chart(plot)

# Useful visualizations

# Scatter plots
# Loss increase - average similarity / all sims same sign
# Loss increase - activation diff
# loss increase - Is And
# loss increase - clean increase pattern / decrease pattern
st.markdown("""
            ### Comparing neuron-wise ablation loss increase

            The AND features are used to highlight neurons in a scatter plot displaying their loss increase.
            """)

ablation_mode = st.selectbox(label="Select the ablation mode", 
             options=["Ablate context neuron", "Ablate context neuron and replace both current and previous token with random tokens"], index=0)

highlight_mode = st.selectbox(label="Select the type of AND neurons to highlight",
                              options=["Two Features", "Single Features", "Current Token", "Grouped Tokens"], index=0)

if ablation_mode == "Ablate context neuron":
    ablation_column = "ContextAblationLossIncrease"
else:
    ablation_column = "FullAblationLossIncrease"



plot = px.scatter(df, y=ablation_column, color=highlight_mode + " (AND)", hover_name="Neuron", hover_data=["Neuron", ablation_column],
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

select_mode = st.selectbox(label="Select whether to use all AND neurons or only the neurons where (YYY>others)",
                              options=["All", "Greater", "Top 50 (All)", "Top 50 (Greater)"], index=0)


names = list(ablation_losses[option][select_mode].keys())
short_names = [name.split(" ")[0] for name in names]
loss_values = [[ablation_losses[option][select_mode][name]] for name in names]
plot = plotting_utils.plot_barplot(loss_values, names,
                            short_names=short_names, ylabel="Last token loss",
                            title=f"Loss increase when patching groups of neurons (ablation mode: YYN)",
                            width=750, show=False)
st.plotly_chart(plot)