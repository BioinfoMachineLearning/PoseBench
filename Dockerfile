ARG PYTORCH_TAG=2.3.0-cuda11.8-cudnn8-devel
FROM pytorch/pytorch:${PYTORCH_TAG}

# Add system dependencies
RUN apt-get update \
    # Update image
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        software-properties-common \
        curl \
        gnupg \
    && echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu jammy main" > /etc/apt/sources.list.d/ubuntu-toolchain-r-test.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F \
    && apt-get update \
    # Install essential dependencies
    && apt-get install --no-install-recommends -y \
        build-essential \
        git \
        wget \
        libxrender1 \
        libxtst6 \
        libxext6 \
        libxi6 \
        kalign \
        gcc-11 \
        g++-11 \
    # Install Git LFS
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install --no-install-recommends -y git-lfs \
    && git lfs install \
    # Configure gcc/g++ versions
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    # Clean up dependencies
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Install Conda dependencies
RUN conda install -y -c conda-forge python=3.10 gcc=11.4.0 gxx=11.4.0 libstdcxx=14.1.0 libstdcxx-ng=14.1.0 libgcc=14.1.0 libgcc-ng=14.1.0 compilers=1.5.2 libconeangle=0.1.1 pybind11=3.0.0 && \
    conda clean -afy

# Set work directory
WORKDIR /app/posebench

# Clone and install the package + requirements
ARG GIT_TAG=main
RUN git clone https://github.com/BioinfoMachineLearning/posebench . --branch ${GIT_TAG} \
    && conda env update -f environments/posebench_environment.yaml \
    && pip install -e . \
    && pip install numpy==1.26.4 --no-dependencies \
    && pip install prody==2.4.1 --no-dependencies \
    && conda clean -afy \
    && rm -rf /root/.cache/pip
