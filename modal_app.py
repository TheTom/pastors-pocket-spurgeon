"""Serverless GPU serving of the Spurgeon-Gemma-4-12B model on Modal.

Scale-to-zero: an L4 spins up only when a request arrives and idles back down
after SCALEDOWN_WINDOW seconds, so a live hackathon demo costs nothing while idle.
The TurboQuant llama.cpp fork serves the model with compressed KV cache
(q8_0 K + turbo4 V, the config validated clean on L4 silicon; turbo4/turbo4 is
garbage on Ada, so K stays q8_0 here).

Deploy:  python3 -m modal deploy modal_app.py
URL:     printed on deploy, ends in .modal.run  ->  append /v1 for the OpenAI base
"""
import subprocess

import modal

MODEL_REPO = "thetom-ai/Spurgeon-Gemma-4-12B-v1"
MODEL_FILE = "Spurgeon-Gemma-4-12B-v1-Q8_0.gguf"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
MODEL_PATH = "/models/spurgeon-q8.gguf"
PORT = 8000
SCALEDOWN_WINDOW = 120  # keep warm 2 min after last request, then scale to zero

# CUDA devel image: compile the TurboQuant fork for L4 (sm_89), bake the GGUF in
# so cold starts only pay model load from local disk, not a 12.6 GB download.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git", "cmake", "build-essential", "ca-certificates", "curl", "libgomp1"
    )
    .run_commands(
        "git clone --depth 1 --branch feature/turboquant-kv-cache "
        "https://github.com/TheTom/llama-cpp-turboquant.git /src",
        # LLAMA_CURL=OFF: GGUF is baked below, no in-binary HTTPS download needed.
        "cmake -B /src/build -S /src -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 "
        "-DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF "
        "-DCMAKE_BUILD_TYPE=Release",
        "cmake --build /src/build --config Release -j 4 --target llama-server",
    )
    .run_commands(
        f"mkdir -p /models && curl -fL --retry 3 --retry-delay 5 "
        f"-o {MODEL_PATH} {MODEL_URL}"
    )
)

app = modal.App("spurgeon-turboquant", image=image)


@app.function(
    gpu="L4",
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=3600,
    max_containers=1,  # cap GPU spend to a single L4
)
@modal.web_server(port=PORT, startup_timeout=300)
def serve():
    cmd = [
        "/src/build/bin/llama-server",
        "--model", MODEL_PATH,
        "-ngl", "99",
        "-c", "8192",
        "-fa", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "turbo4",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--alias", "spurgeon",
    ]
    subprocess.Popen(cmd)
