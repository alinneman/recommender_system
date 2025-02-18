# Use the nvidia/cuda base image
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip

# Install Poetry
RUN pip install poetry

COPY . /app

# Set the working directory
WORKDIR /app

# Install dependencies from the Poetry lockfile
RUN poetry install --no-root

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash"]