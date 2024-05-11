FROM firedrakeproject/firedrake-vanilla:latest
LABEL authors="hosseinkhan"


# Rename firedrake to user_firedrake as it is more descriptive as root
#USER root
#RUN usermod --login user_firedrake firedrake

# Switch back to user_firedrake
#USER user_firedrake

# Source the firedrake environment
ENV PATH_FIREDRAKE_VENV_ACTIVATION_SCRIPT=/home/firedrake/firedrake/bin/activate \
    URL_HYDROGYM=https://github.com/ReHoss/hydrogym.git@main

RUN /bin/bash -c "source $PATH_FIREDRAKE_VENV_ACTIVATION_SCRIPT && \
    pip install --upgrade pip && \
    pip install git+$URL_HYDROGYM && \
    pip install  \
    cma \
    colorednoise \
    gpflow \
    git+https://github.com/j-wilson/GPflowSampling.git@develop \
    gymnasium \
    jax \
    jaxlib \
    matplotlib \
    mlflow==1.27.0 \
    omegaconf \
    pystan \
    pyyaml \
    scikit-learn \
    scipy \
    tf_keras \
    tqdm"


# Create a directory for mounting
RUN mkdir -p /home/firedrake/mount_dir/project_root/


# PRINT THE CURRENT ENVIRONMENT TO THE TERMINAL
RUN /bin/bash -c "echo 'Current python: $(which python)' && \
    echo 'Current pip: $(which pip)' && \
    echo 'Current python version: $(python --version)' && \
    echo 'Current pip version: $(pip --version)' && \
    echo 'Current pip list: $(pip list)'"


# Add a new user with the same id and group id as the host user
# --disabled-password: Do not assign a password
# --gecos: Set the user information
# --gid: Set the group id
# --uid: Set the user id

#====================
# Switch to root user to create a new user
#USER root
#ARG USER_ID
#ARG GROUP_ID
# Create a group with the same id as the host user
#RUN addgroup --gid $GROUP_ID host_group_replica

# Add firedrake user to the group
#RUN usermod -aG host_group_replica firedrake

#RUN adduser \
#    --disabled-password \
#    --gecos '' \
#    --gid $GROUP_ID \
#    --uid $USER_ID \
#    host_user_replica

# Set appropriate permissions on the directories
#RUN chown -R host_user_replica:host_group_replica /home/firedrake/

#USER host_user_replica
#====================

# Firedrake change uid approach
USER root
ARG USER_ID
ARG GROUP_ID
RUN usermod -u $USER_ID firedrake
RUN groupmod -g $GROUP_ID firedrake

USER firedrake

ENTRYPOINT ["/bin/bash", "-l", "-c", "source /home/firedrake/firedrake/bin/activate && /bin/bash"]
