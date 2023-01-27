FROM python:3.9

ARG USER_ID
ARG GROUP_ID
ARG USER

# Create a non-root user
RUN groupadd --gid $GROUP_ID $USER
RUN useradd --uid $USER_ID --gid $GROUP_ID $USER

# Install tmux
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y tmux

# Create conda env 
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torchcontentarea
