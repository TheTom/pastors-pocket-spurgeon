# Field Notes: Building Pastor's Pocket Spurgeon

How I fine-tuned a small model into Charles Spurgeon, served it offline, and taught it to
grade sermons. Written for the Build Small Hackathon.

## What I set out to build

A study companion for pastors that runs on hardware you already own, no cloud, no
subscription. Three modes: ask the "Prince of Preachers" for counsel, have him outline a
sermon, or submit your own draft and let him grade it. The whole thing had to sound like
Spurgeon (warm, vivid, Reformed) in clear modern English, and hold the historic Calvinist
line on hard questions instead of refusing or hedging.

The base model is Gemma-4-12B-it. The hard part was turning a general assistant into a
specific 19th-century Baptist preacher without breaking it.

## Training: the first attempt failed (instructively)

The obvious first move was to train on the raw text: ~1.3M words of Spurgeon's sermons,
chunked, with a "preach to me" prompt as the input and a sermon paragraph as the target.

That model failed on three of five rubric gates:

- **Modern English: failed.** Heavy thee/thou/hath. Raw archaic targets produce archaic
  output.
- **On-topic answering: failed.** It learned to emit generic sermon pastiche on autopilot
  and ignored the actual question. Asked about the papacy, it rambled about Judas.
- **Theology: failed, and often backwards.** Asked about perseverance of the saints it gave
  an Arminian answer (the opposite of Reformed). Asked about hell it denied judgment.

The lesson was clear: training on (generic prompt to raw chunk) teaches *style* while
actively destroying *instruction-following* and teaching *no correct topical theology*. Pure
style-completion is the wrong recipe.

## Training: the recipe that worked

The redesign, refined across four more passes, came down to four ideas:

1. **Self-distillation, not raw text.** Prompt the base Gemma with a strong Spurgeon persona
   and ask it real questions. Base Gemma already writes excellent modern-English
   Spurgeon-flavored prose when told to. Capture those answers as the training set. You are
   teaching the model to make its own best behavior the default.
2. **Ground in his real words, output modern English.** For each question, retrieve the top
   real Spurgeon passages with a pure-Python BM25 index over ~9,400 sermon chunks, and feed
   them as context with an explicit instruction: draw on their spirit and imagery, but answer
   in modern English, never copy the archaic wording. This bakes in his illustrations and
   theology without the thee/thou.
3. **Mix in neutral questions.** About 15 percent plain assistant questions (Python, trivia,
   recipes) with a neutral system prompt, so the model stays a usable assistant and does not
   collapse into "preacher mode" on everyday asks. This prevents catastrophic forgetting.
4. **A confident persona, no abliteration.** Gemma is heavily safety and progressive tuned
   and tends to refuse or hedge on controversial-but-biblical topics. The plan had a fallback
   to an abliterated base. It was never needed. A confident persona plus the LoRA fully
   overrides Gemma's defaults: the final model answered 8 of 8 hard stress-test questions in
   a biblically Reformed register with zero refusals.

### Hyperparameters that mattered

QLoRA, 4-bit NF4, bf16 compute, on all seven projection modules. Rank 32, alpha 64: enough
to capture cadence, not so much it parrots verbatim. Three epochs over about 290 examples,
learning rate 1.5e-4 cosine. Small set, few epochs: more of either pushed toward
memorization.

### Convergence

Five passes total: raw-chunk completion (fail), self-distilled Q and A (pass), broader plus
register-aware self-distillation (the winner), RAG-grounded generation (roughly equal to the
winner), and a consolidated union of the last two (the definitive model). The interesting
result: RAG-grounding did not dramatically beat plain self-distillation. Base Gemma's
imitation was already very good. Diminishing returns set in fast, so I stopped.

Final artifact: `Spurgeon-Gemma-4-12B-v1`, a Q8_0 GGUF, 12.7 GB, published to the Hub.

## Serving it small and offline

Merge the LoRA into the base, convert to GGUF, quantize to Q8_0, and serve with llama.cpp.
The model holds 128K context with an 8-bit KV cache on a single consumer GPU.

To stretch that further I served it through my TurboQuant llama.cpp fork, which adds
compressed KV-cache types. With `q8_0` keys and `turbo4` values, the V cache drops to about
4.25 bits per value (roughly 3.8x smaller than fp16), which fits up to about 2.5x longer
conversations on the same hardware versus an uncompressed KV cache. One gotcha worth noting:
symmetric `turbo4` keys and values produced garbage on an Ada-class GPU, while `q8_0` keys
with `turbo4` values is clean on both Ada and Blackwell. Turbo cache types require flash
attention.

## Building the app

The frontend is a lightweight Gradio app that talks to any OpenAI-compatible endpoint, so
the same app runs against a local llama.cpp server or a hosted one. BM25 retrieval surfaces
real cited Spurgeon passages beside each answer, and the app falls back to canned answers if
no model is connected, so the demo never breaks.

Two engineering notes from the build:

- **Sermon Review grading is deterministic.** The model only picks a number from 1 to 5; the
  app maps that number to a fixed "Sword and Trowel" tier name (5 is "A Trumpet in Zion"
  down to 1 is "A Cloud Without Rain") and the exact verdict format. Early versions let the
  model write the whole rating line and it would invent tier names or sign its own name. The
  number comes from a low-temperature read of the critique it just wrote, so the grade is
  stable and calibrated (good sermons score higher than weak ones, consistently).
- **Completion via assistant-prefill, not a "continue" turn.** On long answers the 12B
  sometimes emits an early end-of-turn mid-sentence. The fix is to resend the conversation
  ending on the partial assistant message, with no new user turn, so the server continues the
  text directly. Sending a trailing "please continue" user message made this fine-tune reply
  with a stray token instead. The same prefill trick builds the review section by section,
  forcing the Summary, Strengths, Concerns, and A Word of Exhortation headers so the model
  cannot drop a section.

## Lessons

- Self-distill from a strong base plus a persona prompt. It is cheap, fast, and high quality.
- Ground in real text but instruct modern output to get the voice without the archaism.
- Mix in neutral examples to keep general assistant ability.
- Rank 32, alpha 64, about 3 epochs, about 300 examples is a good recipe for voice transfer.
- Do not train on a raw archaic corpus. It overfits to thee/thou and breaks instructions.
- Do not over-train. More passes gave diminishing returns past the third.
- A confident persona can override a model's safety and stylistic defaults without
  abliteration.
- For reliable structured output, do not trust the model with the format. Force the
  structure and derive anything load-bearing (like a grade) deterministically.

## Links

- Model: https://huggingface.co/thetom-ai/Spurgeon-Gemma-4-12B-v1
- Code: https://github.com/TheTom/pastors-pocket-spurgeon
- TurboQuant llama.cpp fork: https://github.com/TheTom/llama-cpp-turboquant
- Live demo Space: https://huggingface.co/spaces/build-small-hackathon/pastors-pocket-spurgeon
