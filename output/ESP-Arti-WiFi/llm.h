// llama2.c-style inference engine for the M5Stack Cardputer (ESP32-S3FN8,
// 512KB SRAM, 8MB flash, no PSRAM). Runs GPT-Neo (TinyStories-Instruct,
// CRDP v3 blobs) and LLaMA (TinyLLama-v0, CRDP v1 blobs).
//
//   - Q4_0 quantized weights, embedded in the app .bin and read zero-copy
//     from MMU-mapped flash. No SD card, no partition install step.
//   - GPT-Neo matmuls run int8-quantized activations against row-planar Q4
//     weights on the ESP32-S3 PIE SIMD unit (main/dot_q4_pie.S), split
//     across both cores; host builds use a matching scalar path.
//   - GPT-Neo KV cache is int4 with per-32-group bf16 scales.
//   - Tokenizer is walked from flash; no 256 KB index in heap.
//   - Sampler: argmax (T=0), full-vocab multinomial (top_p >= 1) or top-p
//     nucleus over a capped candidate set — no full-vocab sort or probindex.
#pragma once
#include <stdint.h>
#include <stddef.h>

enum LlmArch { ARCH_LLAMA = 1, ARCH_GPTNEO = 2 };

typedef struct {
  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size;
  int seq_len;     // llama: training max; gptneo: wpe rows kept (= hard ctx cap)
  int quant_type;  // 4 = Q4_0
  int shared_classifier;
  int arch;        // LlmArch
} Config;

typedef struct {
  float* x;       // [dim]
  float* xb;      // [dim]
  float* xb2;     // [dim]
  float* hb;      // [hidden_dim]
  float* hb2;     // [hidden_dim]
  float* q;       // [dim]
  float* k;       // [kv_dim] fp32 scratch for the current position
  float* v;       // [kv_dim] fp32 scratch for the current position
  float* att;     // [n_heads, kv_seq_len]
  float* logits;  // [vocab_size]
  // LLaMA path: KV cache stored as bf16 (2 KB/position instead of 4 KB), which
  // lets KV_SEQ_LEN=64 fit in the same 128 KB an fp32 cache needed for 32.
  uint16_t* key_cache;   // bf16 [n_layers, kv_seq_len, kv_dim]
  uint16_t* value_cache; // bf16 [n_layers, kv_seq_len, kv_dim]
  // GPT-Neo path: int4 nibbles with one bf16 scale per 32-element group.
  // Every head (head_size 8 or 16) lies inside a single group, so attention
  // applies one scale per (head, position). ~1 KB/position at dim=256 —
  // that's what lets a 72-token window fit the 8M model in internal SRAM.
  // Nibble packing: dim i lives in byte i/2 (low nibble if even, high if odd),
  // biased by +8 (stored 1..15, symmetric int4 = clip(round(v/s), ±7)).
  uint8_t*  key_cache4;   // [n_layers, kv_seq_len, kv_dim/2]
  uint8_t*  value_cache4; // [n_layers, kv_seq_len, kv_dim/2]
  uint16_t* k_gscales;    // bf16 [n_layers, kv_seq_len, kv_dim/32]
  uint16_t* v_gscales;    // bf16 [n_layers, kv_seq_len, kv_dim/32]
} RunState;

typedef struct {
  Config config;
  RunState state;
  int kv_seq_len;
  const uint8_t* base;   // start of embedded model bytes

  // Pointers into `base` (set after parsing the header).
  // For GPT-Neo, rms_att/rms_ffn/rms_final hold the LayerNorm gammas (ln_1,
  // ln_2, ln_f) and the *_b pointers below hold the betas; w1/w2 are the MLP
  // c_fc/c_proj matrices and w3 is unused.
  const float* rms_att;          // fp32 [n_layers, dim]
  const float* rms_ffn;          // fp32 [n_layers, dim]
  const float* rms_final;        // fp32 [dim]
  const float* ln_att_b;         // gptneo: fp32 [n_layers, dim]
  const float* ln_ffn_b;         // gptneo: fp32 [n_layers, dim]
  const float* ln_final_b;       // gptneo: fp32 [dim]
  const float* bo;               // gptneo: attn out_proj bias [n_layers, dim]
  const float* b_fc;             // gptneo: mlp c_fc bias [n_layers, hidden_dim]
  const float* b_proj;           // gptneo: mlp c_proj bias [n_layers, dim]
  const float* wpe;              // gptneo: position embeddings [seq_len, dim]
  const uint8_t* token_embed_q4; // Q4_0 [vocab, dim]
  const uint8_t* wq_q4, *wk_q4, *wv_q4, *wo_q4;
  const uint8_t* w1_q4, *w2_q4, *w3_q4;
  const uint8_t* wcls_q4;

  // Per-layer stride in bytes for the Q4 tensors above.
  size_t stride_wq, stride_wk, stride_wv, stride_wo;
  size_t stride_w1w3, stride_w2;
} Transformer;

// Walking tokenizer — no in-RAM vocab table. The tokenizer.bin lives in flash
// and we linear-scan it for each lookup. Trades CPU for ~128 KB of heap.
//
// Two blob formats are supported:
//   - llama2.c style (SentencePiece): [u32 max_len] then [f32 score][i32 len][bytes]*
//   - CTK2 (GPT-2 byte-level BPE, pruned): [u32 magic][i32 vocab][i32 max_len]
//     [i32 eos][i32 n_merges][u16 byte_ids[256]][(u16 a,u16 b,u16 c) sorted by
//     (a,b)]* then [i32 len][bytes]* — exact BPE via the merge pair table.
typedef struct {
  const uint8_t* base;        // start of tokenizer blob (in flash)
  size_t size;
  int vocab_size;
  int max_token_length;
  int style;                  // ARCH_LLAMA (sentencepiece) or ARCH_GPTNEO (bpe)
  int eos_id;                 // gptneo: <|endoftext|> id in the pruned vocab
  int n_merges;               // gptneo
  const uint16_t* byte_ids;   // gptneo: byte -> token id [256]
  const uint16_t* merges;     // gptneo: (a,b,c) triples sorted by (a,b)
  const uint8_t* pieces;      // gptneo: start of [i32 len][bytes] records
  char byte_pieces[512];      // 256 single-byte fallback pieces (data only)
} Tokenizer;

typedef struct {
  int vocab_size;
  float temperature;
  float top_p;        // nucleus mass; >= 1.0 disables (full multinomial)
  uint64_t rng_state;
} Sampler;

// Initialize a transformer that lives in flash. `model_bytes` must point at
// the converter's model_q4.bin contents. `kv_seq_len` is the active context
// window; the runstate is sized for it.
bool   llm_init_embedded(Transformer* t, const uint8_t* model_bytes, size_t model_size,
                         int kv_seq_len);

// Parse a tokenizer blob (also in flash). Allocates almost nothing.
bool   llm_tokenizer_from_memory(Tokenizer* tk, const uint8_t* data, size_t size,
                                 int vocab_size);

void   llm_build_sampler(Sampler* s, int vocab_size, float temperature, float top_p,
                         uint64_t seed);

float* llm_forward(Transformer* t, int token, int pos);

// Like llm_forward, but decouples the physical KV write slot (`write_slot`,
// bounds the cache and the causal attention range) from the absolute sequence
// position (`abspos`) fed to the GPT-Neo learned position embedding. This is
// what makes a sliding context window possible: after evicting old slots the
// write slot drops back down while abspos keeps climbing. llm_forward() passes
// pos for both. (LLaMA/RoPE ignores abspos and rotates on write_slot.)
float* llm_forward_at(Transformer* t, int token, int write_slot, int abspos);

// Sliding-window KV eviction (GPT-Neo int4 cache only). Keeps slots
// [0, keep_head) pinned — a prefix, which the caller sizes and which is NOT
// necessarily the whole prompt; where that band lands decides what the model
// forgets, so choose it deliberately. Drops `evict` slots starting at
// keep_head, and shifts every later slot down to close the gap — for all
// layers and both key/value nibble planes and their group scales. Cached keys
// keep the absolute position baked into them at encode time; only the physical
// layout moves, so the caller must advance abspos independently.
// Returns the number of slots actually evicted, or 0 if the cache was left
// untouched (unsupported arch, or degenerate keep_head/evict). Callers MUST
// rewind their write position by the return value, not by what they asked for.
int    llm_kv_slide(Transformer* t, int keep_head, int evict);

int    llm_sample(Sampler* s, float* logits);

void   llm_encode(Tokenizer* tk, const char* text, int8_t bos, int8_t eos,
                  int* tokens, int* n_tokens);
const char* llm_decode(Tokenizer* tk, int prev, int token, char* scratch, size_t scratch_size);
