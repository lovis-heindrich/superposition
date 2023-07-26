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
    with open(path / "set_losses.json", "r") as f:
        set_losses = json.load(f)
    with open(path / "and_conditions.json", "r") as f:
        and_conditions = json.load(f)
    return df, set_losses, and_conditions

df, set_losses, and_conditions = load_data(option)

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

            Positive AND neurons are neurons that only fire (activation>0) when all three input components are present (the context neuron, the previous token, the current token).
            Negative AND neurons are neurons that always fire unless all three input components are present.
            
            Boosted / deboosted columns show whether the neuron has a consistent boost / deboost effect across prompt types. Boosts / deboosts without a post-gelu effect are ignored.
            """)

show_losses = st.checkbox("Show ablation loss changes", value=False)
show_cosine_sims = st.checkbox("Show cosine similarities with token / context directions", value=False)

display_df = df.copy()
if not show_losses:
    display_df = display_df.drop(columns=["FullAblationLossIncrease", "ContextAblationLossIncrease"])
if not show_cosine_sims:
    display_df = display_df.drop(columns=["PrevTokenSim", "CurrTokenSim", "ContextSim", "PosSim", "NegSim"])
display_df = display_df.drop(columns=["AblationDiff"])

st.dataframe(display_df.round(2))

st.markdown("""
            ### Looking for non-linearities

            """)

data_select = st.selectbox(label="Select which value to compare", 
             options=["Change in loss", "Change in correct token logit"], index=1)
data_select_index = "loss" if data_select == "Change in loss" else "logits"

st.markdown("""Current token: the current token is always present""")
if data_select == "Change in loss":
    st.latex(r'''(NYN-YYY)-((NYN-YYN)+(NYN-NYY))''')
else:
    st.latex(r'''(YYY-NYN)-((YYN-NYN)+(NYY-NYN))''')

st.markdown("""Grouped tokens: current and previous token are grouped together""")
if data_select == "Change in loss":
    st.latex(r'''(NNN-YYY)-((NNN-YYN)+(NNN-NNY))''')
else:
    st.latex(r'''(YYY-NNN)-((YYN-NNN)+(NNY-NNN))''')

st.markdown("""Single features: all three input features appear individually""")
if data_select == "Change in loss":
    st.latex(r'''(NNN-YYY)-((NNN-YNN)+(NNN-NYN)+(NNN-NNY))''')
else:
    st.latex(r'''(YYY-NNN)-((YNN-NNN)+(NYN-NNN)+(NNY-NNN))''')

st.markdown("""Two features: two input features appear together""")
if data_select == "Change in loss":
    st.latex(r'''(NNN-YYY)-((NNN-YYN)+(NNN-YNY)+(NNN-NYY))/2''')
else:
    st.latex(r'''(YYY-NNN)-((YYN-NNN)+(YNY-NNN)+(NYY-NNN))/2''')


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

            """)

df["AndName"] = "None"
df.loc[df["And"], "AndName"] = "Positive"
df.loc[df["NegAnd"], "AndName"] = "Negative"

df["BoostName"] = "Inconsistent"
df.loc[df["Boosted"], "BoostName"] = "Boost"
df.loc[df["Deboosted"], "BoostName"] = "Deboost"

ablation_mode = st.selectbox(label="Select the ablation mode", 
             options=["Ablate context neuron", "Ablate context neuron and replace both current and previous token with random tokens"], index=0)

highlight_mode = st.selectbox(label="Select the highlighted neurons",
                              options=["AND neurons", "Consistent boost / deboost neurons"], index=0)

if ablation_mode == "Ablate context neuron":
    ablation_column = "ContextAblationLossIncrease"
else:
    ablation_column = "FullAblationLossIncrease"

if highlight_mode == "AND neurons":
    highlight_name = "AndName"
    labels={"AndName": "AND neuron type"}
else:
    highlight_name = "BoostName"
    labels={"BoostName": "Boosted neuron type"}

plot = px.scatter(df, y=ablation_column, color=highlight_name, 
                  color_discrete_sequence=["grey", "red", "blue"], labels=labels)

plot.update_layout(
    title="Loss increase when patching individual MLP5 neurons",
    yaxis_title="Ablation loss increase"
)
st.plotly_chart(plot)

# Ablation change for pos, neg, and strongly activating neurons

st.markdown("""
            ### Comparing ablation loss increase for groups of neurons

            """)

if ablation_mode == "Ablate context neuron":
    ablation_key = "YYN"
else:
    ablation_key = "NNN"

num_neurons = set_losses[option]["NumNeurons"]

losses = set_losses[option][ablation_key]
short_names = list(losses.keys())
names = [f"{short_names[i]} (N={num_neurons[short_names[i]]})" for i in range(len(short_names))]
loss_values = [[losses[name]] for name in short_names]
plot = plotting_utils.plot_barplot(loss_values, names,
                            short_names=short_names, ylabel="Last token loss",
                            title=f"Loss increase when patching groups of neurons (ablation mode: {ablation_key})",
                            width=750, show=False)
st.plotly_chart(plot)