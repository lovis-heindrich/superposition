# %%
import streamlit as st
import json
import haystack_utils

#st.title("MLP5 N-Gram Analysis")

#@st.cache_data
def load_data():
    with open('data/verify_bigrams/pos_loss_data.json', 'r') as f:
        all_loss_data = json.load(f)

    return all_loss_data

all_loss_data = load_data()

tokens = list(all_loss_data.keys())


st.markdown("""
            ## Loss analysis
            
            ### Setup 
            We generate 100 prompts of 20 random common german tokens and append the tokens of the selected n-gram. 
            
            ### Path patching
            Three loss values are computed: 
            1. The loss when setting the German neuron to its mean active value.
            2. The loss when ablating the German neuron.
            3. MLP5 activations when setting the German neuron to active but ablating MLP4 and L4 attention are cached. Then, the German neuron is ablated and MLP5 is patched with the cached activations.

            ### Replacing n-gram tokens
            One of the tokens of the n-grams is replaced with a random common german token.
            
            """)

# %%

option = st.sidebar.selectbox(
    'Select the N-Gram to analyze',
    tokens, index=0)

replacable_tokens = list(all_loss_data[option].keys())[1:]
print(replacable_tokens)
replace_column = st.selectbox(
    'Select the token to replace with random tokens',
    replacable_tokens, index=0)

# %%

col1, col2 = st.columns(2)

names = ["Original", "Ablated", "MLP5 path patched"]
short_names = ["Original", "Ablated", "MLP5"]

with col1:
    title = f"Last token loss on full prompt"
    original_loss, ablated_loss, only_activated_loss = all_loss_data[option]["None"].values()
    plot = haystack_utils.plot_barplot([original_loss, ablated_loss, only_activated_loss], 
                                       names, legend=False, width=300, short_names=short_names, ylabel="Loss", title=title, show=False)
    st.plotly_chart(plot)

with col2:
    title = f"Last token loss when replacing '{replace_column}' token"
    original_loss, ablated_loss, only_activated_loss = all_loss_data[option][replace_column].values()
    plot = haystack_utils.plot_barplot([original_loss, ablated_loss, only_activated_loss], 
                                       names, legend=False, width=300, short_names=short_names, ylabel="Loss", title=title, show=False)
    st.plotly_chart(plot) 