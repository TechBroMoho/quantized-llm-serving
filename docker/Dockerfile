# vLLM serving image: the pinned upstream OpenAI server plus our load tester.
# Modal builds this file directly (Image.from_dockerfile); docker-compose.yml
# runs the same image on any local NVIDIA GPU.
FROM vllm/vllm-openai:v0.10.2@sha256:607442e407b0fea97f8a132a78b787c121a996dd4de181fa08e8da06e71ec2db

# Our code only needs packages the base image already ships (aiohttp,
# transformers, pyyaml), so it is copied rather than pip-installed: the
# image's CUDA torch build and vLLM's resolved dependencies stay untouched.
ENV PYTHONPATH=/opt/llmbench/src \
    PYTHONUNBUFFERED=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    LLMBENCH_VLLM_IMAGE=vllm/vllm-openai:v0.10.2@sha256:607442e407b0fea97f8a132a78b787c121a996dd4de181fa08e8da06e71ec2db

# The base image ships only `python3`; Modal detects the image's Python by
# running `python`, the same symlink its own add_python images create.
RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY src/llmbench /opt/llmbench/src/llmbench
COPY configs /opt/llmbench/configs

# The base ENTRYPOINT (python3 -m vllm.entrypoints.openai.api_server) is kept
# for `docker compose`; Modal clears it with `.entrypoint([])`.
