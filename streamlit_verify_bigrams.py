# %%
import streamlit as st
import json
import haystack_utils

st.title("MLP5 N-Gram Analysis")

#@st.cache_data
def load_data():
    with open('data/verify_bigrams/pos_loss_data.json', 'r') as f:
        all_loss_data = json.load(f)
    with open('data/verify_bigrams/neuron_loss_diffs.json', 'r') as f:
        neuron_loss_diffs = json.load(f)
    with open('data/verify_bigrams/neuron_loss_data.json', 'r') as f:
        neuron_loss_data = json.load(f)
    return all_loss_data, neuron_loss_diffs, neuron_loss_data

all_loss_data, neuron_loss_diffs, neuron_loss_data = load_data()

tokens = list(all_loss_data.keys())


st.markdown("""
            ## Position-wise loss analysis
            
            #### Setup 
            We generate 100 prompts of 20 random common german tokens and append the tokens of the selected n-gram. 
            
            #### Path patching
            Three loss values are computed: 
            1. The loss when setting the German neuron to its mean active value.
            2. The loss when ablating the German neuron.
            3. MLP5 activations when setting the German neuron to active but ablating MLP4 and L4 attention are cached. Then, the German neuron is ablated and MLP5 is patched with the cached activations.

            #### Replacing n-gram tokens
            To find out which tokens of the n-gram matter, we replace one of its with a random common german token.
            
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



st.markdown("""
            ## Neuron-wise loss analysis
            
            We compute the difference in loss when path patching all MLP5 neurons compared to ablating a single neuron (i.e. removing it from the set of patched neurons).
            """)

neuron_loss_diffs = neuron_loss_diffs[option]

sort = st.checkbox("Sort by difference", value=True)

if sort:
    neuron_loss_diffs.sort()

plot = haystack_utils.line(neuron_loss_diffs, width=650, xlabel="Neuron", ylabel="Loss change", title="Loss change from ablating individual MLP5 neurons", show_legend=False, plot=False)
st.plotly_chart(plot)

st.markdown("""
            Comparing loss values of removing a set of neurons from the set of path patched MLP5 neurons. Neurons are removed separately from the neurons with the most positive loss change and the most negative loss change.
            """)

top_neurons_count = st.slider("Number of neurons to remove", 1, 25, 10, 1)

names = ["Original", "Ablated", "MLP5 path patched", f"MLP5 path patched + Top {top_neurons_count} MLP5 neurons ablated", f"MLP5 path patched + Bottom {top_neurons_count} MLP5 neurons ablated"]
short_names = ["Original", "Ablated", "MLP5 path patched", f"Top MLP5 removed", f"Bottom MLP5 removed"]

loss_data = neuron_loss_data[option][str(top_neurons_count)]
values = list(loss_data.values())
#values = [original_loss.tolist(), ablated_loss.tolist(), all_MLP5_loss.tolist(), top_MLP5_ablated_loss.tolist(), bottom_MLP5_ablated_loss.tolist()]
plot = haystack_utils.plot_barplot(values, names, short_names=short_names, 
                                   width=650, show=False, ylabel="Loss", 
                                   title=f"Average last token loss when removing top / bottom neurons from path patching")
st.plotly_chart(plot)