FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /workspace/code

# Copiamos requirements primero (para cache de capas)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Jupyter
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--notebook-dir=/workspace/code"]

