FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

RUN apt update -y
RUN apt install -y graphviz git

COPY pyproject.toml uv.lock .python-version /app/
COPY packages/ /app/packages/
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=cache,target=/root/.local/share/uv/python uv sync

COPY . /app

RUN mkdir -p /app/ragEmbeddings
RUN chmod +x /app/docker-entrypoint.sh

RUN echo 'PS1="\[\e[96;1m\]AutoRecLab\[\e[0m\] \\$ "' >> /etc/bash.bashrc
RUN echo 'source /app/.venv/bin/activate' >> /etc/bash.bashrc

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["bash"]
