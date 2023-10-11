# %%
import streamlit as st
import json
import plotting_utils
from pathlib import Path

st.set_page_config(page_title="Ablation analysis", page_icon="📊",)
st.sidebar.success("Select an analysis above.")

st.title("MLP5 N-Gram Analysis")

#@st.cache_data
def load_data():
    path = Path(__file__).parent
    with open(path / '../data/verify_bigrams/pos_loss_data.json', 'r') as f:
        all_loss_data = json.load(f)
    with open(path / '../data/verify_bigrams/neuron_loss_diffs.json', 'r') as f:
        neuron_loss_diffs = json.load(f)
    with open(path / '../data/verify_bigrams/neuron_loss_data.json', 'r') as f:
        neuron_loss_data = json.load(f)
    with open(path / '../data/verify_bigrams/summed_neuron_boosts.json', 'r') as f:
        summed_neuron_boosts = json.load(f)
    with open(path / '../data/verify_bigrams/summed_split_neuron_boosts.json', 'r') as f:
        summed_individual_neuron_boosts = json.load(f)
    with open(path / '../data/verify_bigrams/individual_neuron_boosts.json', 'r') as f:
        individual_neuron_boosts = json.load(f)
    return all_loss_data, neuron_loss_diffs, neuron_loss_data, summed_neuron_boosts, summed_individual_neuron_boosts, individual_neuron_boosts

all_loss_data, neuron_loss_diffs, neuron_loss_data, summed_neuron_boosts, summed_individual_neuron_boosts, individual_neuron_boosts = load_data()

tokens = list(all_loss_data.keys())

option = st.selectbox(
    'Select the N-Gram to analyze',
    tokens, index=0)

st.markdown("""
            ## Position-wise loss analysis
            
            We generate 100 prompts of 20 random common german tokens and append the tokens of the selected n-gram. 
            To find out which tokens of the n-gram matter, we replace one of its tokens with a random common german token.
            
            Three loss values are computed: 
            1. The loss when setting the German neuron to its mean active value.
            2. The loss when ablating the German neuron.
            3. MLP5 activations when setting the German neuron to active but ablating MLP4 and L4 attention are cached. Then, the German neuron is ablated and MLP5 is patched with the cached activations.

            """)

# %%

replacable_tokens = list(all_loss_data[option].keys())[1:]
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
    plot = plotting_utils.plot_barplot([original_loss, ablated_loss, only_activated_loss], 
                                       names, legend=False, width=300, yaxis=dict(range=[0, 15]), short_names=short_names, ylabel="Loss", title=title, show=False)
    st.plotly_chart(plot)

with col2:
    title = f"Last token loss when replacing '{replace_column}' token"
    original_loss, ablated_loss, only_activated_loss = all_loss_data[option][replace_column].values()
    plot = plotting_utils.plot_barplot([original_loss, ablated_loss, only_activated_loss], 
                                       names, legend=False, width=300, yaxis=dict(range=[0, 15]), short_names=short_names, ylabel="Loss", title=title, show=False)
    st.plotly_chart(plot) 



st.markdown("""
            ## Neuron-wise loss analysis
            
            We compute the difference in loss when path patching all MLP5 neurons compared to ablating a single neuron (i.e. removing it from the set of patched neurons).
            """)

neuron_loss_diffs = neuron_loss_diffs[option]

sort = st.checkbox("Sort by difference", value=True)

if sort:
    neuron_loss_diffs.sort()

plot = plotting_utils.line(neuron_loss_diffs, width=650, xlabel="Neuron", ylabel="Loss change", title="Loss change from ablating individual MLP5 neurons", show_legend=False, plot=False)
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
plot = plotting_utils.plot_barplot(values, names, short_names=short_names, 
                                   width=650, yaxis=dict(range=[0, 10]), show=False, ylabel="Loss", 
                                   title=f"Average last token loss when removing top / bottom neurons from path patching")
st.plotly_chart(plot)


st.markdown("""
            ## Boosted tokens from summed neurons

            We compute the difference in logprobs when removing the top / bottom top neurons from the set of path patched MLP5 neurons.

            To isolate positive / negative effects, the analysis is also run on each neuron individually, with their individual positive / negative effects summed afterwards. This (hopefully) allows to see effects that are cancelled out (through destructive interference) when summing the logprobs directly.

            Only tokens with a logprob greater than -7 are shown.
            """)

top_neurons_count_logprob = st.slider("Number of neurons", 1, 25, 10, 1)
top_neurons_checkbox = st.checkbox("Top neurons", value=True)
isolate_effects = st.checkbox("Sum individual neuron effects", value=False)

if top_neurons_checkbox:
    top_select_str = "Top"
    title_append = f" from top {top_neurons_count_logprob} neurons"
else:
    top_select_str = "Bottom"
    title_append = f" from bottom {top_neurons_count_logprob} neurons"

if isolate_effects:
    summed_neuron_boosted = summed_individual_neuron_boosts[option][str(top_neurons_count_logprob)][top_select_str]["Boosted"]
    summed_neuron_deboosted = summed_individual_neuron_boosts[option][str(top_neurons_count_logprob)][top_select_str]["Deboosted"]
else:
    summed_neuron_boosted = summed_neuron_boosts[option][str(top_neurons_count_logprob)][top_select_str]["Boosted"]
    summed_neuron_deboosted = summed_neuron_boosts[option][str(top_neurons_count_logprob)][top_select_str]["Deboosted"]

summed_neuron_boosted_values = summed_neuron_boosted["Logprob difference"]
summed_neuron_boosted_tokens = summed_neuron_boosted["Tokens"]

summed_neuron_deboosted_values = summed_neuron_deboosted["Logprob difference"]
summed_neuron_deboosted_tokens = summed_neuron_deboosted["Tokens"]

col1, col2 = st.columns(2)

with col1:
    xticks = summed_neuron_boosted_tokens
    xlabel = ""
    ylabel = "Logprob difference"
    title = "Boosted tokens" + title_append
    plot = plotting_utils.line(summed_neuron_boosted_values, 
                               width=320, plot=False,
                               xlabel=xlabel, ylabel=ylabel, title=title, xticks=xticks, show_legend=False)
    st.plotly_chart(plot)

with col2:
    xticks = [x[0] for x in summed_neuron_deboosted_tokens]
    ylabel = ""
    xlabel = ""
    title = "Deboosted tokens" + title_append
    plot = plotting_utils.line(summed_neuron_deboosted_values, 
                               width=320, plot=False,
                               xlabel=xlabel, ylabel=ylabel, title=title, xticks=xticks, show_legend=False)
    st.plotly_chart(plot)


st.markdown("""
            ## Boosted tokens from individual neurons

            We also compute the difference in logprobs when removing individual neurons from the set of path patched MLP5 neurons.
            """)

top_neuron_selector = st.slider("Top / bottom neuron", 1, 25, 1, 1)
top_individual_neuron_checkbox = st.checkbox("Top neuron", value=True)

bottom_neuron = individual_neuron_boosts[option][str(top_neuron_selector-1)][top_select_str]["Neuron"]

if top_individual_neuron_checkbox:
    top_select_str = "Top"
    top_neuron = individual_neuron_boosts[option][str(top_neuron_selector-1)][top_select_str]["Neuron"]
    title_append = f" from neuron {top_neuron}"
else:
    top_select_str = "Bottom"
    bottom_neuron = individual_neuron_boosts[option][str(top_neuron_selector-1)][top_select_str]["Neuron"]
    title_append = f" from neuron {bottom_neuron}"

individual_neuron_boosted = individual_neuron_boosts[option][str(top_neuron_selector-1)][top_select_str]["Boosted"]
individual_neuron_deboosted = individual_neuron_boosts[option][str(top_neuron_selector-1)][top_select_str]["Deboosted"]

individual_neuron_boosted_values = individual_neuron_boosted["Logprob difference"]
individual_neuron_boosted_tokens = individual_neuron_boosted["Tokens"]

individual_neuron_deboosted_values = individual_neuron_deboosted["Logprob difference"]
individual_neuron_deboosted_tokens = individual_neuron_deboosted["Tokens"]

col1, col2 = st.columns(2)

with col1:
    xticks = individual_neuron_boosted_tokens
    xlabel = ""
    ylabel = "Logprob difference"
    title = "Boosted tokens" + title_append
    plot = plotting_utils.line(individual_neuron_boosted_values, 
                               width=320, plot=False,
                               xlabel=xlabel, ylabel=ylabel, title=title, xticks=xticks, show_legend=False)
    st.plotly_chart(plot)

with col2:
    xticks = [x[0] for x in individual_neuron_deboosted_tokens]
    ylabel = ""
    xlabel = ""
    title = "Deboosted tokens" + title_append
    plot = plotting_utils.line(individual_neuron_deboosted_values, 
                               width=320, plot=False,
                               xlabel=xlabel, ylabel=ylabel, title=title, xticks=xticks, show_legend=False)
    st.plotly_chart(plot)