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
print(tokens)
# %%

option = st.selectbox(
    'Select the N-Gram to analyze',
    tokens, index=0)

replacable_tokens = list(all_loss_data[option].keys())
replace_column = st.selectbox(
    'Select the token to replace with random tokens',
    replacable_tokens, index=0)

losses = all_loss_data[option][replace_column]
