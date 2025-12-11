FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

RUN apt update -y
RUN apt install -y graphviz 

COPY pyproject.toml uv.lock .python-version /app/
COPY packages/ /app/packages/
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=cache,target=/root/.local/share/uv/python uv sync

COPY . /app

RUN echo 'PS1="\[\e[96;1m\]AutoRecLab\[\e[0m\] \\$ "' >> /etc/bash.bashrc

ENTRYPOINT ["bash"]
