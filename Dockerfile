FROM python:3.9

ARG USER_ID
ARG GROUP_ID
ARG USER

# Create a non-root user
RUN groupadd --gid $GROUP_ID $USER
RUN useradd --uid $USER_ID --gid $GROUP_ID $USER

# Create conda env 
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torchcontentarea
