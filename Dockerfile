FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y git

RUN pip install nltk kaleido tqdm einops seaborn plotly-express \
    scikit-learn \torchmetrics ipykernel ipywidgets nbformat \
    git+https://github.com/neelnanda-io/TransformerLens \
    git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python \
    git+https://github.com/neelnanda-io/neelutils.git \
    git+https://github.com/neelnanda-io/neel-plotly.git