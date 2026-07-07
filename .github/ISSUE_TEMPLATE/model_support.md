---
name: Model support
about: A model won't load, convert, or serve
title: ""
labels: model-support
---

**Run the preflight first**
Before filing, check whether Athena can load the model:
```sh
athena pull --check <hf-id>
```
Paste the output — it reports the detected modality and whether the checkpoint's packaging is loadable, without downloading weights.

**Model**
- Hugging Face id (or local path):
- Modality (chat / vision / embeddings / transcription / diarization / speaker):
- Quantization / format if known:

**What failed**
- Command (`athena pull`, `athena convert`, or a request) and the exact error:
- `athena --version`:

**Note**
Athena's architecture coverage is inherited from its MLX substrate; a checkpoint can be a valid model yet be packaged in a way Athena can't load. The preflight names the *structural* requirement that's missing (e.g. a tokenizer or head config), which is the most useful thing to include here.
