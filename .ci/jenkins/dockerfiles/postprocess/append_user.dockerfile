# This dockerfile is used to add user to Docker image matching the one executing CI
ARG base_image
FROM ${base_image}

ARG USERNAME
ARG UID
ARG GID

# Add user and group
RUN groupadd -g $GID -o $USERNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $USERNAME
USER $USERNAME
