# OmniEVIE support for llama.cpp

Requirements: llama.cpp b12811 as the base, CMake 3.14 or newer (3.18 with CUDA), a C++17 compiler, and for NVIDIA GPUs the CUDA Toolkit 11.8 or newer. Apply with `git apply omnievie-llamacpp-v3.patch` (8 files, +221 lines).

What it adds:

- **Converter.** Adds `OmniEVIEBackbone` to the text and mmproj maps. The text GGUF reuses the Qwen3.5 classes (the DeltaNet reorder included). It maps the `language_model.` prefix, writes `qwen35.attention.causal = false` and pooling none, and stores `custom_text_proj` as the dense-out weight and bias. The mmproj takes the `visual.*` tower.
- **`src/models/qwen35.cpp`.** Loads the optional `DENSE_2_OUT` weight and bias. `llama_model::build_graph` already applies them per token when pooling is none.
- **`src/llama-graph.cpp`.** An embeddings context of a non-causal model with a dense-out head builds no LM head. Before this, the tied 248K-row head made an 8.4 GB compute buffer at 8,192 tokens.
- **`tools/mtmd/mtmd-embedding.cpp`** (`llama-mtmd-embedding`). Evaluates one sequence in a single pass: text tokens and image or video rows go into one mixed `llama_batch_ext`, using mtmd's M-RoPE positions. The full-attention layers attend over the whole sequence, as the model was trained; the DeltaNet layers stay a causal scan. The output is one L2-normalized float32 row per token. `--serve` keeps the model loaded and takes one request per stdin line.

Agreement with the reference pipeline (sentence-transformers 6.3.0, CPU):

- The model card's text example gives MaxSim [[23.06, 4.95, 5.10], [4.97, 24.11, 3.44]], against the card's [[23.03, 4.95, 5.10], [4.97, 24.19, 3.43]], with identical token counts.
- An image document gives the same token count as sentence-transformers.
- On a 128-query x 64-page sample, BF16 per-token cosine to the fp32 reference is 0.9994 (queries), 0.9997 (text documents) and 0.973 (page images). Sentence-transformers' own bf16 run gives 0.998, 0.999 and 0.960.

```sh
python convert_hf_to_gguf.py OmniEVIE-4B --outfile OmniEVIE-4B-BF16.gguf --outtype bf16
python convert_hf_to_gguf.py OmniEVIE-4B --mmproj --outfile mmproj-OmniEVIE-4B-BF16.gguf --outtype bf16
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-mtmd-embedding
S=$(printf '<|endoftext|>%.0s' 1 2 3 4 5 6 7 8 9 10)
./build/bin/llama-mtmd-embedding OmniEVIE-4B-Q8_0.gguf mmproj-OmniEVIE-4B-BF16.gguf - "a text query$S" q.f32 4608
./build/bin/llama-mtmd-embedding OmniEVIE-4B-Q8_0.gguf mmproj-OmniEVIE-4B-BF16.gguf page.png $'<|im_start|>user\n<__media__><|im_end|><|endoftext|>' d.f32 4608
```

Prompts follow the model's `chat_template.jinja`, with `<__media__>` in place of `<|vision_start|><|image_pad|><|vision_end|>`. MAX_TOKENS must cover the whole sequence; a full page is about 3,700 tokens.

Video input runs through mtmd's video helper; its frame grouping and timestamps are not yet matched to the reference video processor.
