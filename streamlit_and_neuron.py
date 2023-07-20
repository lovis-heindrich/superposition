import streamlit as st
import json
import plotting_utils
import pandas as pd
import plotly.express as px

st.title("MLP5 AND-Neurons Analysis")

tokens = ["orschlägen", "häufig", "beweglich"]

option = st.selectbox(
    'Select the trigram to analyze',
    tokens, index=0)

#@st.cache_data
def load_data(tokens):
    df = pd.read_pickle(f"data/and_neurons/df_{tokens}.pkl")
    return df

df = load_data(option)

# Look for neurons that consistently respond to all 3 directions
df["Boosted"] = (df["YNN"]>df["NNN"])&(df["NYN"]>df["NNN"])&(df["NNY"]>df["NNN"])&\
                (df["YYY"]>df["YNN"])&(df["YYY"]>df["NYN"])&(df["YYY"]>df["NNY"])&\
                (df["YYY"]>0) # Negative boosts don't matter

df["Deboosted"] = (df["YNN"]<df["NNN"])&(df["NYN"]<df["NNN"])&(df["NNY"]<df["NNN"])&\
                (df["YYY"]<df["YNN"])&(df["YYY"]<df["NYN"])&(df["YYY"]<df["NNY"])&\
                (df["NNN"]>0) # Deboosting negative things doesn't matter


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
    display_df = display_df.drop(columns=["PrevTokenSim", "CurrTokenSim", "ContextSim", "AllSim"])
display_df = display_df.drop(columns=["AblationDiff"])

st.dataframe(display_df.round(2))


# Useful visualizations

# Scatter plots
# Loss increase - average similarity / all sims same sign
# Loss increase - activation diff
# loss increase - Is And
# loss increase - clean increase pattern / decrease pattern
st.markdown("""
            ### Comparing ablation loss increase

            """)

df["AndName"] = "None"
df["AndName"][df["And"]] = "Positive"
df["AndName"][df["NegAnd"]] = "Negative"

df["BoostName"] = "Inconsistent"
df["BoostName"][df["Boosted"]] = "Boost"
df["BoostName"][df["Deboosted"]] = "Deboost"

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