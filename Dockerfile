# Use the RAPIDS AI base image with CUDA 12.2 and Python 3.10
FROM rapidsai/base:24.08-cuda12.2-py3.10

# Ensure we're running as root
USER root

# Update and install dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y chromium-browser chromium-driver xvfb wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/chromedriver /usr/local/bin/chromedriver

# Install additional system dependencies
RUN apt-get update \
    && apt-get install -y libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages using pip for Python 3.10
RUN python3.10 -m pip install --no-cache-dir \
        'git+https://github.com/exorde-labs/exorde_data.git' \
        'git+https://github.com/oldandnerd/exorde-client.git' \
        selenium==4.2.0 \
        wtpsplit==1.3.0 \
    && python3.10 -m pip install --no-cache-dir --upgrade 'git+https://github.com/JustAnotherArchivist/snscrape.git'

# Clean up pip cache
RUN rm -rf /root/.cache/* \
    && rm -rf /root/.local/cache/*

# Set the working directory
WORKDIR /exorde

# Copy the pre-installation script for models
COPY exorde/pre_install.py /exorde/exorde_install_models.py

# Download and install models for fastText, spaCy, and others
RUN mkdir -p /tmp/fasttext-langdetect \
    && wget -O /tmp/fasttext-langdetect/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
    && python3.10 -m spacy download en_core_web_trf \
    && python3.10 exorde_install_models.py

# Copy the application data into the image
COPY data /exorde

# Set the release version
ARG RELEASE_VERSION
RUN echo ${RELEASE_VERSION} > .release

# Configure environment variables for offline use
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# Set display port to avoid crash
ENV DISPLAY=:99

# Set protocol buffers implementation to Python
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Copy and prepare the keep_alive script
COPY keep_alive.sh /exorde/keep_alive.sh
RUN chmod +x /exorde/keep_alive.sh \
    && sed -i 's/\r$//' /exorde/keep_alive.sh

# Set the entry point for the Docker container
ENTRYPOINT ["/bin/bash", "/exorde/keep_alive.sh"]
