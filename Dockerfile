FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip
RUN pip install -e .


CMD ["bash"]