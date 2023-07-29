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
    df_logits = pd.read_csv(path / f"and_conditions_logits{file_name_append}.csv")
    df_loss = pd.read_csv(path / f"and_conditions_loss{file_name_append}.csv")
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


features = ["Fix Current", "Fix Previous", "Fix Context", "Single Feature", "Two Features"]

def create_grouped_barplot(df, values):
    df_long = df[df['Value'].isin(values)].melt(id_vars='Value', var_name='Category', value_name='Number')
    fig = px.bar(df_long, x='Value', y='Number', color='Category', barmode='group', title='Grouped Barplots for Selected Values')
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title=data_select,)
    
    return fig

# Test the function
fig = create_grouped_barplot(df, features)
st.plotly_chart(fig)