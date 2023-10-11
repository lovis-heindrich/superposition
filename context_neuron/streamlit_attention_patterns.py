# %%
import streamlit as st
import json
import plotting_utils

st.title("N-Gram Attention Head Analysis")

#@st.cache_data
def load_data():
    with open('data/bigram_attention/head_ablation_losses.json', 'r') as f:
        head_ablation_losses = json.load(f)
    return head_ablation_losses

head_ablation_losses = load_data()

tokens = list(head_ablation_losses.keys())

option = st.selectbox(
    'Select the N-Gram to analyze',
    tokens, index=0)

st.markdown("""
            ## Head ablation analysis

            We ablate each head individually on previous ngram positions to see which heads copy relevant information.
            """)

# %%

replacable_tokens = list(head_ablation_losses[option].keys())
replace_column = st.selectbox(
    'Select the token on which to ablate heads',
    replacable_tokens, index=0)

names = ["Original"] + [f"L{layer}H{head}" for layer in range(5) for head in range(8)]
plot = plotting_utils.plot_barplot(
    head_ablation_losses[option][replace_column], names, 
    legend=False, show=False, width=750,
    title=f"Loss for ablated attention heads on '{replace_column}' of '{option}'")

st.plotly_chart(plot)