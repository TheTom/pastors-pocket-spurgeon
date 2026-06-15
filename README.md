# Pastor's Pocket Spurgeon

A Victorian study companion that **answers, prepares, and grades sermons** in the voice and
Reformed (Calvinist) theology of **Charles Haddon Spurgeon**, the "Prince of Preachers",
running fully **offline** on a single consumer GPU.

Built for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon)
(Backyard AI track).

| | |
|---|---|
| 🤖 **Model** | [`thetom-ai/Spurgeon-Gemma-4-12B-v1`](https://huggingface.co/thetom-ai/Spurgeon-Gemma-4-12B-v1) (Q8_0 GGUF) |
| 🚀 **Live demo** | [Space: build-small-hackathon/pastors-pocket-spurgeon](https://huggingface.co/spaces/build-small-hackathon/pastors-pocket-spurgeon) |
| ⚡ **Serving engine** | [TurboQuant llama.cpp fork](https://github.com/TheTom/llama-cpp-turboquant) (`turbo4` KV compression) |
| 🎬 **Demo video** | see the Space, or `out/spurgeon-demo.mp4` |

## What it does

Three modes, all in Spurgeon's voice and Reformed doctrine:

- **The Counsel** — ask a pastoral or theological question, get a shepherd's answer.
- **Sermon Prep** — give a passage or topic, receive a Spurgeon-style outline.
- **Sermon Review** — submit a sermon draft and Mr. Spurgeon grades it: a **Summary**,
  **Strengths**, **Concerns**, **A Word of Exhortation**, and a verdict on the
  **Sword & Trowel** scale:

  | Marks | Tier |
  |------:|------|
  | 5 | A Trumpet in Zion |
  | 4 | Sound Timber, Well Hewn |
  | 3 | A Lamp Half-Trimmed |
  | 2 | A Skeleton Unclothed |
  | 1 | A Cloud Without Rain |

## How it works

A lightweight **Gradio** app talks to any OpenAI-compatible endpoint (`SPURGEON_ENDPOINT`).
It uses BM25 retrieval over Spurgeon's own sermons to surface real cited passages alongside
each answer, and falls back to canned answers if no model endpoint is configured (so the UI
always demos). The Sermon Review is built section-by-section with forced headers, and the
grade is derived deterministically from the model's mark so the verdict format is always
exact.

## Run it locally

**1. Get the model** ([from Hugging Face](https://huggingface.co/thetom-ai/Spurgeon-Gemma-4-12B-v1)):

```bash
huggingface-cli download thetom-ai/Spurgeon-Gemma-4-12B-v1 \
  Spurgeon-Gemma-4-12B-v1-Q8_0.gguf --local-dir ./model
```

**2. Serve it with TurboQuant KV compression.** Build the
[TurboQuant llama.cpp fork](https://github.com/TheTom/llama-cpp-turboquant), then:

```bash
# q8_0 keys + turbo4 values: TurboQuant compresses the V cache so long
# conversations fit on constrained hardware (up to ~2.5x more context vs fp16 KV).
# turbo4 requires flash attention (-fa on).
llama-server -m ./model/Spurgeon-Gemma-4-12B-v1-Q8_0.gguf \
  -c 131072 -fa on -ctk q8_0 -ctv turbo4 --jinja \
  --host 0.0.0.0 --port 8080 --alias spurgeon
```

(On stock llama.cpp without the fork, use `-ctv q8_0` — you lose the extra compression but
everything else works.)

**3. Run the app**:

```bash
pip install -r requirements.txt
export SPURGEON_ENDPOINT=http://127.0.0.1:8080/v1
export SPURGEON_MODEL=spurgeon
python app.py          # http://127.0.0.1:7860
```

Optional env: `SPURGEON_TOKEN` (bearer token if your endpoint is behind auth).

## TurboQuant

[TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) is a llama.cpp fork that adds
compressed KV-cache types (`turbo2`/`turbo3`/`turbo4`). `turbo4` stores the V cache at
~4.25 bits/value (≈3.8x smaller than fp16); paired with `q8_0` keys it fits **up to ~2.5x
longer conversations** on the same GPU, so a 12B pastor runs comfortably offline on consumer
hardware. It requires flash attention and is validated for this model with `q8_0` keys +
`turbo4` values.

## License

- **Code** in this repository: [Apache 2.0](./LICENSE).
- **Model weights** (`Spurgeon-Gemma-4-12B-v1`): inherit the
  [Gemma license](https://ai.google.dev/gemma/terms) from the base model.
- Spurgeon's sermons used for retrieval and fine-tuning are public domain.
