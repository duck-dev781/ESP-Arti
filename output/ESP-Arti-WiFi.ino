/*
  ESP-Arti Wi-Fi firmware
  ------------------------
  ESP32 WROVER-E / PSRAM
  - Wi-Fi only
  - No BLE
  - No SD
  - No LittleFS
  - Browser chat endpoint: POST /chat
  - GitHub supplies the runtime model/config files.
  - Neural inference runs locally on the ESP32.

  Model:
    TinyStories-260K, llama2.c checkpoint format.
    dim=64, hidden_dim=172, layers=5, heads=8, kv_heads=4, vocab=512.
    The checkpoint is about 1.06 MB and is streamed directly into PSRAM.

  NOTE:
    This is a real trained neural language model, but TinyStories-260K was
    trained for short children's stories, not instruction-following chat.
    It is therefore a local neural-model milestone, not a ChatGPT-sized model.
*/

#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static const char *WIFI_SSID = "";
static const char *WIFI_PASSWORD = "";

static const char *GITHUB_RAW_BASE =
  "https://raw.githubusercontent.com/duck-dev781/ESP-Arti/main/";

static const char *MODEL_CONFIG_PATH = "ai/config.h";

static const char *MODEL_URL =
  "https://raw.githubusercontent.com/maddiedreese/gbc-transformer/main/stories260K.bin";

static const char *TOKENIZER_URL =
  "https://raw.githubusercontent.com/maddiedreese/gbc-transformer/main/tok512.bin";

static const size_t EXPECTED_MODEL_BYTES = 1056540;
static const size_t EXPECTED_TOKENIZER_BYTES = 6227;

static const int MAX_RUNTIME_SEQ = 256;
static const int MAX_GENERATION_TOKENS = 64;
static const int MAX_PROMPT_TOKENS = 96;
static const int VOCAB_SIZE = 512;

WebServer server(80);

String modelConfig;
String lastError;

bool modelLoaded = false;
bool modelLoading = false;

struct ModelConfig {
  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size;
  int max_seq_len;
};

struct TransformerWeights {
  float *token_embedding_table;
  float *rms_att_weight;
  float *rms_ffn_weight;
  float *wq;
  float *wk;
  float *wv;
  float *wo;
  float *w1;
  float *w2;
  float *w3;
  float *rms_final_weight;
  float *freq_cis_real;
  float *freq_cis_imag;
  float *wcls;
};

struct Tokenizer {
  char *tokenizerData;
  char *vocab[VOCAB_SIZE];
  float vocab_scores[VOCAB_SIZE];
  int vocab_size;
  int max_token_length;
};

static ModelConfig modelCfg;
static ModelConfig runtimeCfg;
static TransformerWeights weights;
static Tokenizer tokenizer;

static uint8_t *modelData = nullptr;

static float *s_x = nullptr;
static float *s_xb = nullptr;
static float *s_xb2 = nullptr;
static float *s_hb = nullptr;
static float *s_hb2 = nullptr;
static float *s_q = nullptr;
static float *s_k = nullptr;
static float *s_v = nullptr;
static float *s_att = nullptr;
static float *s_logits = nullptr;
static float *s_key_cache = nullptr;
static float *s_value_cache = nullptr;

static bool allocateInferenceMemory() {
  s_x = (float *)ps_malloc(64 * sizeof(float));
  s_xb = (float *)ps_malloc(64 * sizeof(float));
  s_xb2 = (float *)ps_malloc(64 * sizeof(float));
  s_hb = (float *)ps_malloc(172 * sizeof(float));
  s_hb2 = (float *)ps_malloc(172 * sizeof(float));
  s_q = (float *)ps_malloc(64 * sizeof(float));
  s_k = (float *)ps_malloc(32 * sizeof(float));
  s_v = (float *)ps_malloc(32 * sizeof(float));
  s_att = (float *)ps_malloc(8 * MAX_RUNTIME_SEQ * sizeof(float));
  s_logits = (float *)ps_malloc(VOCAB_SIZE * sizeof(float));

  const size_t kvFloats =
    (size_t)5 * MAX_RUNTIME_SEQ * 32;

  s_key_cache = (float *)ps_malloc(kvFloats * sizeof(float));
  s_value_cache = (float *)ps_malloc(kvFloats * sizeof(float));

  if (!s_x || !s_xb || !s_xb2 ||
      !s_hb || !s_hb2 || !s_q ||
      !s_k || !s_v || !s_att ||
      !s_logits || !s_key_cache ||
      !s_value_cache) {
    lastError = "Could not allocate neural inference buffers in PSRAM.";
    return false;
  }

  return true;
}

static bool downloadToBuffer(
  const char *url,
  uint8_t **outBuffer,
  size_t expectedBytes
) {
  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;
  http.setTimeout(30000);

  if (!http.begin(client, url)) {
    lastError = String("HTTP begin failed: ") + url;
    return false;
  }

  int code = http.GET();

  if (code != HTTP_CODE_OK) {
    lastError = String("HTTP ") + code + " while downloading " + url;
    http.end();
    return false;
  }

  int contentLength = http.getSize();

  if (contentLength > 0 &&
      (size_t)contentLength != expectedBytes) {
    lastError =
      String("Unexpected file size. Expected ") +
      expectedBytes +
      ", got " +
      contentLength;
    http.end();
    return false;
  }

  uint8_t *buffer = (uint8_t *)ps_malloc(expectedBytes);

  if (!buffer) {
    lastError =
      String("Not enough PSRAM for download buffer: ") +
      expectedBytes;
    http.end();
    return false;
  }

  WiFiClient *stream = http.getStreamPtr();

  size_t received = 0;
  unsigned long lastData = millis();

  while (received < expectedBytes) {
    size_t availableBytes = stream->available();

    if (availableBytes) {
      size_t wanted = expectedBytes - received;

      if (availableBytes > wanted) {
        availableBytes = wanted;
      }

      int n = stream->readBytes(
        buffer + received,
        availableBytes
      );

      if (n > 0) {
        received += (size_t)n;
        lastData = millis();
      }
    } else {
      if (millis() - lastData > 30000) {
        free(buffer);
        http.end();
        lastError = "Timed out while downloading model data.";
        return false;
      }

      delay(2);
    }
  }

  http.end();

  if (received != expectedBytes) {
    free(buffer);
    lastError =
      String("Incomplete download. Expected ") +
      expectedBytes +
      ", got " +
      received;
    return false;
  }

  *outBuffer = buffer;
  return true;
}

static bool loadTokenizer() {
  uint8_t *data = nullptr;

  Serial.println("Downloading TinyStories tokenizer...");

  if (!downloadToBuffer(
        TOKENIZER_URL,
        &data,
        EXPECTED_TOKENIZER_BYTES
      )) {
    return false;
  }

  tokenizer.tokenizerData = (char *)data;

  size_t pos = 0;

  if (EXPECTED_TOKENIZER_BYTES < sizeof(int32_t)) {
    lastError = "Tokenizer file is too small.";
    return false;
  }

  int32_t maxTokenLength = 0;
  memcpy(
    &maxTokenLength,
    data + pos,
    sizeof(int32_t)
  );
  pos += sizeof(int32_t);

  tokenizer.max_token_length = maxTokenLength;
  tokenizer.vocab_size = VOCAB_SIZE;

  for (int i = 0; i < VOCAB_SIZE; ++i) {
    if (pos + sizeof(float) + sizeof(int32_t) >
        EXPECTED_TOKENIZER_BYTES) {
      lastError = "Tokenizer structure is invalid.";
      return false;
    }

    float score = 0.0f;
    int32_t len = 0;

    memcpy(&score, data + pos, sizeof(float));
    pos += sizeof(float);

    memcpy(&len, data + pos, sizeof(int32_t));
    pos += sizeof(int32_t);

    if (len < 0 ||
        pos + (size_t)len > EXPECTED_TOKENIZER_BYTES) {
      lastError = "Tokenizer token length is invalid.";
      return false;
    }

    char *token = (char *)malloc((size_t)len + 1);

    if (!token) {
      lastError = "Could not allocate tokenizer token.";
      return false;
    }

    memcpy(token, data + pos, (size_t)len);
    token[len] = '\0';

    tokenizer.vocab[i] = token;
    tokenizer.vocab_scores[i] = score;

    pos += (size_t)len;
  }

  return true;
}

static void initWeights(
  TransformerWeights *w,
  const ModelConfig *p,
  float *ptr,
  bool sharedWeights
) {
  int headSize = p->dim / p->n_heads;

  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;

  w->rms_att_weight = ptr;
  ptr += p->n_layers * p->dim;

  w->wq = ptr;
  ptr += p->n_layers * p->dim * p->dim;

  int kvDim = (p->dim * p->n_kv_heads) / p->n_heads;

  w->wk = ptr;
  ptr += p->n_layers * p->dim * kvDim;

  w->wv = ptr;
  ptr += p->n_layers * p->dim * kvDim;

  w->wo = ptr;
  ptr += p->n_layers * p->dim * p->dim;

  w->rms_ffn_weight = ptr;
  ptr += p->n_layers * p->dim;

  w->w1 = ptr;
  ptr += p->n_layers * p->dim * p->hidden_dim;

  w->w2 = ptr;
  ptr += p->n_layers * p->hidden_dim * p->dim;

  w->w3 = ptr;
  ptr += p->n_layers * p->dim * p->hidden_dim;

  w->rms_final_weight = ptr;
  ptr += p->dim;

  w->freq_cis_real = ptr;
  ptr += p->max_seq_len * headSize / 2;

  w->freq_cis_imag = ptr;
  ptr += p->max_seq_len * headSize / 2;

  w->wcls =
    sharedWeights
      ? w->token_embedding_table
      : ptr;
}

static void rmsnorm(
  float *out,
  const float *x,
  const float *weight,
  int size
) {
  float ss = 0.0f;

  for (int i = 0; i < size; ++i) {
    ss += x[i] * x[i];
  }

  ss /= (float)size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

  for (int i = 0; i < size; ++i) {
    out[i] = weight[i] * (ss * x[i]);
  }
}

static void softmax(float *x, int size) {
  float maxValue = x[0];

  for (int i = 1; i < size; ++i) {
    if (x[i] > maxValue) {
      maxValue = x[i];
    }
  }

  float sum = 0.0f;

  for (int i = 0; i < size; ++i) {
    x[i] = expf(x[i] - maxValue);
    sum += x[i];
  }

  if (sum <= 0.0f) {
    return;
  }

  for (int i = 0; i < size; ++i) {
    x[i] /= sum;
  }
}

static void matmul(
  float *out,
  const float *x,
  const float *w,
  int n,
  int d
) {
  for (int i = 0; i < d; ++i) {
    float value = 0.0f;

    for (int j = 0; j < n; ++j) {
      value += w[i * n + j] * x[j];
    }

    out[i] = value;
  }
}

static float *forward(
  const ModelConfig *p,
  const TransformerWeights *w,
  int token,
  int pos
) {
  const int dim = p->dim;
  const int kvDim =
    (p->dim * p->n_kv_heads) / p->n_heads;
  const int kvMul =
    p->n_heads / p->n_kv_heads;
  const int hiddenDim = p->hidden_dim;
  const int headSize = dim / p->n_heads;

  const float *contentRow =
    w->token_embedding_table +
    token * dim;

  memcpy(
    s_x,
    contentRow,
    (size_t)dim * sizeof(float)
  );

  for (int layer = 0;
       layer < p->n_layers;
       ++layer) {

    rmsnorm(
      s_xb,
      s_x,
      w->rms_att_weight + layer * dim,
      dim
    );

    matmul(
      s_q,
      s_xb,
      w->wq + layer * dim * dim,
      dim,
      dim
    );

    matmul(
      s_k,
      s_xb,
      w->wk + layer * dim * kvDim,
      dim,
      kvDim
    );

    matmul(
      s_v,
      s_xb,
      w->wv + layer * dim * kvDim,
      dim,
      kvDim
    );

    for (int i = 0; i < dim; i += 2) {
      int headDim = i % headSize;

      int freqIndex =
        pos * headSize / 2 +
        headDim / 2;

      float freq =
        w->freq_cis_real[freqIndex];

      float fci =
        w->freq_cis_imag[freqIndex];

      float q0 = s_q[i];
      float q1 = s_q[i + 1];

      s_q[i] =
        q0 * freq -
        q1 * fci;

      s_q[i + 1] =
        q0 * fci +
        q1 * freq;

      if (i < kvDim) {
        float k0 = s_k[i];
        float k1 = s_k[i + 1];

        s_k[i] =
          k0 * freq -
          k1 * fci;

        s_k[i + 1] =
          k0 * fci +
          k1 * freq;
      }
    }

    size_t layerOffset =
      (size_t)layer *
      (size_t)MAX_RUNTIME_SEQ *
      (size_t)kvDim;

    memcpy(
      s_key_cache +
        layerOffset +
        (size_t)pos * kvDim,
      s_k,
      (size_t)kvDim * sizeof(float)
    );

    memcpy(
      s_value_cache +
        layerOffset +
        (size_t)pos * kvDim,
      s_v,
      (size_t)kvDim * sizeof(float)
    );

    for (int head = 0;
         head < p->n_heads;
         ++head) {

      float *qHead =
        s_q + head * headSize;

      float *attHead =
        s_att + head * MAX_RUNTIME_SEQ;

      for (int t = 0;
           t <= pos;
           ++t) {

        const float *kHead =
          s_key_cache +
          layerOffset +
          (size_t)t * kvDim +
          (size_t)(head / kvMul) * headSize;

        float score = 0.0f;

        for (int i = 0;
             i < headSize;
             ++i) {
          score +=
            qHead[i] * kHead[i];
        }

        attHead[t] =
          score /
          sqrtf((float)headSize);
      }

      softmax(attHead, pos + 1);

      float *outHead =
        s_xb + head * headSize;

      memset(
        outHead,
        0,
        (size_t)headSize * sizeof(float)
      );

      for (int t = 0;
           t <= pos;
           ++t) {

        const float *vHead =
          s_value_cache +
          layerOffset +
          (size_t)t * kvDim +
          (size_t)(head / kvMul) * headSize;

        float a = attHead[t];

        for (int i = 0;
             i < headSize;
             ++i) {
          outHead[i] +=
            a * vHead[i];
        }
      }
    }

    matmul(
      s_xb2,
      s_x,
      w->wo + layer * dim * dim,
      dim,
      dim
    );

    for (int i = 0; i < dim; ++i) {
      s_x[i] += s_xb2[i];
    }

    rmsnorm(
      s_xb,
      s_x,
      w->rms_ffn_weight + layer * dim,
      dim
    );

    matmul(
      s_hb,
      s_xb,
      w->w1 + layer * dim * hiddenDim,
      dim,
      hiddenDim
    );

    matmul(
      s_hb2,
      s_xb,
      w->w3 + layer * dim * hiddenDim,
      dim,
      hiddenDim
    );

    for (int i = 0;
         i < hiddenDim;
         ++i) {

      float value = s_hb[i];

      value *=
        1.0f /
        (1.0f + expf(-value));

      s_hb[i] =
        value * s_hb2[i];
    }

    matmul(
      s_xb,
      s_hb,
      w->w2 + layer * hiddenDim * dim,
      hiddenDim,
      dim
    );

    for (int i = 0; i < dim; ++i) {
      s_x[i] += s_xb[i];
    }
  }

  rmsnorm(
    s_x,
    s_x,
    w->rms_final_weight,
    dim
  );

  matmul(
    s_logits,
    s_x,
    w->wcls,
    dim,
    p->vocab_size
  );

  return s_logits;
}

static int vocabLookup(
  const char *text,
  int length
) {
  for (int token = 0;
       token < tokenizer.vocab_size;
       ++token) {

    const char *v =
      tokenizer.vocab[token];

    if ((int)strlen(v) != length) {
      continue;
    }

    if (memcmp(
          v,
          text,
          (size_t)length
        ) == 0) {
      return token;
    }
  }

  return -1;
}

static int bpeEncode(
  const String &text,
  uint16_t *tokens,
  int maxTokens
) {
  int count = 0;

  for (size_t i = 0;
       i < text.length() &&
       count < maxTokens;
       ++i) {

    char single[2];

    single[0] = text[i];
    single[1] = '\0';

    int id =
      vocabLookup(single, 1);

    if (id >= 0) {
      tokens[count++] =
        (uint16_t)id;
    }
  }

  for (;;) {
    float bestScore = -1e30f;
    int bestId = -1;
    int bestIndex = -1;

    for (int i = 0;
         i + 1 < count;
         ++i) {

      const char *left =
        tokenizer.vocab[tokens[i]];

      const char *right =
        tokenizer.vocab[tokens[i + 1]];

      size_t leftLen = strlen(left);
      size_t rightLen = strlen(right);

      if (leftLen + rightLen >= 128) {
        continue;
      }

      char merge[128];

      memcpy(
        merge,
        left,
        leftLen
      );

      memcpy(
        merge + leftLen,
        right,
        rightLen
      );

      merge[leftLen + rightLen] =
        '\0';

      int id =
        vocabLookup(
          merge,
          (int)(leftLen + rightLen)
        );

      if (id >= 0 &&
          tokenizer.vocab_scores[id] >
            bestScore) {

        bestScore =
          tokenizer.vocab_scores[id];

        bestId = id;
        bestIndex = i;
      }
    }

    if (bestIndex < 0) {
      break;
    }

    tokens[bestIndex] =
      (uint16_t)bestId;

    for (int j = bestIndex + 1;
         j + 1 < count;
         ++j) {
      tokens[j] =
        tokens[j + 1];
    }

    --count;
  }

  return count;
}

static int argmax(
  const float *values,
  int count
) {
  int best = 0;
  float bestValue = values[0];

  for (int i = 1;
       i < count;
       ++i) {

    if (values[i] > bestValue) {
      bestValue = values[i];
      best = i;
    }
  }

  return best;
}

static String generateNeuralReply(
  const String &userMessage,
  const String &onlineContext
) {
  if (!modelLoaded) {
    return "The neural model is not loaded.";
  }

  uint16_t promptTokens[MAX_PROMPT_TOKENS];

  String prompt;

  /*
    TinyStories was not instruction-tuned, so this is a simple text prompt.
    The model will continue the text locally rather than calling an API.
  */
  prompt = "The user said: ";
  prompt += userMessage;

  if (onlineContext.length()) {
    prompt += "\nCurrent information: ";
    prompt += onlineContext.substring(0, 500);
  }

  prompt += "\nArti said:";

  int promptCount =
    bpeEncode(
      prompt,
      promptTokens,
      MAX_PROMPT_TOKENS
    );

  if (promptCount <= 0) {
    promptCount = 0;
  }

  if (promptCount >= MAX_RUNTIME_SEQ - 1) {
    promptCount =
      MAX_RUNTIME_SEQ - 2;
  }

  memset(
    s_key_cache,
    0,
    (size_t)5 *
      MAX_RUNTIME_SEQ *
      32 *
      sizeof(float)
  );

  memset(
    s_value_cache,
    0,
    (size_t)5 *
      MAX_RUNTIME_SEQ *
      32 *
      sizeof(float)
  );

  int token = 1;
  int position = 0;

  for (int i = 0;
       i < promptCount;
       ++i) {

    token = promptTokens[i];

    forward(
      &runtimeCfg,
      &weights,
      token,
      position
    );

    ++position;

    if (position >= MAX_RUNTIME_SEQ - 1) {
      break;
    }
  }

  if (position == 0) {
    forward(
      &runtimeCfg,
      &weights,
      1,
      0
    );

    position = 1;
  }

  String reply;
  reply.reserve(700);

  for (int generated = 0;
       generated < MAX_GENERATION_TOKENS &&
       position < MAX_RUNTIME_SEQ;
       ++generated) {

    int next =
      argmax(
        s_logits,
        runtimeCfg.vocab_size
      );

    if (next == 1 || next == 2) {
      break;
    }

    const char *piece =
      tokenizer.vocab[next];

    if (piece) {
      reply += piece;
    }

    token = next;

    forward(
      &runtimeCfg,
      &weights,
      token,
      position
    );

    ++position;
  }

  if (reply.length() == 0) {
    return "(The local neural model generated no text.)";
  }

  return reply;
}

static bool loadNeuralModel() {
  modelLoading = true;
  modelLoaded = false;
  lastError = "";

  Serial.println();
  Serial.println("Loading local neural model...");
  Serial.println("Downloading TinyStories-260K model to PSRAM.");

  if (!downloadToBuffer(
        MODEL_URL,
        &modelData,
        EXPECTED_MODEL_BYTES
      )) {
    modelLoading = false;
    Serial.println(lastError);
    return false;
  }

  if (EXPECTED_MODEL_BYTES < sizeof(ModelConfig)) {
    lastError = "Model file is too small.";
    modelLoading = false;
    return false;
  }

  memcpy(
    &modelCfg,
    modelData,
    sizeof(ModelConfig)
  );

  Serial.println("Model header:");
  Serial.print("dim = ");
  Serial.println(modelCfg.dim);

  Serial.print("hidden_dim = ");
  Serial.println(modelCfg.hidden_dim);

  Serial.print("layers = ");
  Serial.println(modelCfg.n_layers);

  Serial.print("heads = ");
  Serial.println(modelCfg.n_heads);

  Serial.print("kv_heads = ");
  Serial.println(modelCfg.n_kv_heads);

  Serial.print("vocab = ");
  Serial.println(
    abs(modelCfg.vocab_size)
  );

  Serial.print("checkpoint seq = ");
  Serial.println(modelCfg.max_seq_len);

  if (modelCfg.dim != 64 ||
      modelCfg.hidden_dim != 172 ||
      modelCfg.n_layers != 5 ||
      modelCfg.n_heads != 8 ||
      modelCfg.n_kv_heads != 4 ||
      abs(modelCfg.vocab_size) != VOCAB_SIZE ||
      modelCfg.max_seq_len != 512) {

    lastError =
      "Downloaded model does not match TinyStories-260K configuration.";

    modelLoading = false;
    return false;
  }

  bool sharedWeights =
    modelCfg.vocab_size > 0;

  modelCfg.vocab_size =
    abs(modelCfg.vocab_size);

  runtimeCfg = modelCfg;

  if (runtimeCfg.max_seq_len >
      MAX_RUNTIME_SEQ) {
    runtimeCfg.max_seq_len =
      MAX_RUNTIME_SEQ;
  }

  float *weightStart =
    (float *)(modelData + sizeof(ModelConfig));

  initWeights(
    &weights,
    &modelCfg,
    weightStart,
    sharedWeights
  );

  if (!loadTokenizer()) {
    modelLoading = false;
    Serial.println(lastError);
    return false;
  }

  if (!allocateInferenceMemory()) {
    modelLoading = false;
    Serial.println(lastError);
    return false;
  }

  modelLoaded = true;
  modelLoading = false;

  Serial.print("Neural model loaded. Free PSRAM: ");
  Serial.println(ESP.getFreePsram());

  return true;
}

static bool downloadRuntimeConfig() {
  String cfg;

  Serial.println(
    "Downloading GitHub runtime configuration..."
  );

  if (!downloadToBuffer(
        String(GITHUB_RAW_BASE) +
          MODEL_CONFIG_PATH,
        (uint8_t **)&cfg,
        0
      )) {
    /*
      This path is not used for the model anymore.
      Keep the GitHub config download optional so the neural model can load
      even if the small config file is absent.
    */
    return false;
  }

  return true;
}

static String jsonEscape(
  const String &s
) {
  String r;
  r.reserve(s.length() + 16);

  for (size_t i = 0;
       i < s.length();
       ++i) {

    char c = s[i];

    switch (c) {
      case '\\':
        r += "\\\\";
        break;

      case '"':
        r += "\\\"";
        break;

      case '\n':
        r += "\\n";
        break;

      case '\r':
        r += "\\r";
        break;

      case '\t':
        r += "\\t";
        break;

      default:
        if ((unsigned char)c < 32) {
          r += ' ';
        } else {
          r += c;
        }
        break;
    }
  }

  return r;
}

static String getJsonString(
  const String &body,
  const String &key
) {
  String needle =
    "\"" + key + "\"";

  int p =
    body.indexOf(needle);

  if (p < 0) {
    return "";
  }

  p = body.indexOf(':', p);

  if (p < 0) {
    return "";
  }

  ++p;

  while (
    p < (int)body.length() &&
    isspace(
      (unsigned char)body[p]
    )
  ) {
    ++p;
  }

  if (
    p >= (int)body.length() ||
    body[p] != '"'
  ) {
    return "";
  }

  ++p;

  String result;
  bool escaped = false;

  for (; p < (int)body.length(); ++p) {
    char c = body[p];

    if (escaped) {
      if (c == 'n') {
        result += '\n';
      } else if (c == 'r') {
        result += '\r';
      } else if (c == 't') {
        result += '\t';
      } else {
        result += c;
      }

      escaped = false;
      continue;
    }

    if (c == '\\') {
      escaped = true;
      continue;
    }

    if (c == '"') {
      break;
    }

    result += c;
  }

  return result;
}

static String searchOnline(
  const String &query
) {
  if (WiFi.status() != WL_CONNECTED) {
    return "";
  }

  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;
  http.setTimeout(12000);

  String encoded;
  encoded.reserve(
    query.length() * 2
  );

  for (size_t i = 0;
       i < query.length();
       ++i) {

    char c = query[i];

    if (
      isalnum((unsigned char)c) ||
      c == '-' ||
      c == '_' ||
      c == '.'
    ) {
      encoded += c;
    } else if (c == ' ') {
      encoded += '+';
    } else {
      char buf[4];

      snprintf(
        buf,
        sizeof(buf),
        "%%%02X",
        (unsigned char)c
      );

      encoded += buf;
    }
  }

  String url =
    "https://html.duckduckgo.com/html/?q=" +
    encoded;

  if (!http.begin(client, url)) {
    return "";
  }

  int code = http.GET();

  if (code != HTTP_CODE_OK) {
    http.end();
    return "";
  }

  String page = http.getString();
  http.end();

  String plain;
  plain.reserve(5000);

  bool inTag = false;

  for (
    size_t i = 0;
    i < page.length() &&
    plain.length() < 5000;
    ++i
  ) {
    char c = page[i];

    if (c == '<') {
      inTag = true;
      continue;
    }

    if (c == '>') {
      inTag = false;
      plain += ' ';
      continue;
    }

    if (!inTag) {
      plain += c;
    }
  }

  while (
    plain.indexOf("  ") >= 0
  ) {
    plain.replace("  ", " ");
  }

  return plain;
}

static bool queryNeedsSearch(
  const String &q
) {
  String s = q;
  s.toLowerCase();

  if (s.startsWith("search ")) {
    return true;
  }

  if (s.startsWith("look up ")) {
    return true;
  }

  if (s.startsWith("find ")) {
    return true;
  }

  if (s.indexOf("latest") >= 0) {
    return true;
  }

  if (s.indexOf("current") >= 0) {
    return true;
  }

  if (s.indexOf("today") >= 0) {
    return true;
  }

  if (s.indexOf("right now") >= 0) {
    return true;
  }

  if (s.indexOf("news") >= 0) {
    return true;
  }

  return false;
}

static String searchQueryFromMessage(
  const String &message
) {
  String q = message;
  String low = q;
  low.toLowerCase();

  if (low.startsWith("search ")) {
    return q.substring(7);
  }

  if (low.startsWith("look up ")) {
    return q.substring(8);
  }

  if (low.startsWith("find ")) {
    return q.substring(5);
  }

  return q;
}

static void handleRoot() {
  String page;
  page.reserve(2400);

  page += "<!doctype html><html><head>";
  page += "<meta name='viewport' content='width=device-width,initial-scale=1'>";
  page += "<title>ESP-Arti</title>";
  page += "<style>";
  page += "body{font-family:system-ui;background:#111;color:#eee;margin:0}";
  page += "main{max-width:850px;margin:auto;padding:16px}";
  page += "#chat{height:65vh;overflow:auto;border:1px solid #444;border-radius:12px;padding:12px}";
  page += ".msg{padding:10px;margin:8px 0;border-radius:10px;background:#222;white-space:pre-wrap}";
  page += ".user{background:#333}";
  page += "form{display:flex;gap:8px;margin-top:10px}";
  page += "input{flex:1;padding:13px;border-radius:9px;border:1px solid #555;background:#181818;color:#fff}";
  page += "button{padding:13px 18px;border:0;border-radius:9px}";
  page += "</style></head><body><main>";
  page += "<h1>ESP-Arti</h1><div id='chat'></div>";
  page += "<form onsubmit='sendMsg(event)'>";
  page += "<input id='msg' autocomplete='off' placeholder='Talk to Arti...'>";
  page += "<button>Send</button></form>";
  page += "<script>";
  page += "const c=document.getElementById('chat');";
  page += "function add(w,t,k){let d=document.createElement('div');";
  page += "d.className='msg '+k;d.textContent=w+': '+t;";
  page += "c.appendChild(d);c.scrollTop=c.scrollHeight}";
  page += "async function sendMsg(e){e.preventDefault();";
  page += "let i=document.getElementById('msg');let t=i.value.trim();";
  page += "if(!t)return;i.value='';add('You',t,'user');";
  page += "try{let r=await fetch('/chat',{method:'POST',";
  page += "headers:{'Content-Type':'application/json'},";
  page += "body:JSON.stringify({message:t})});";
  page += "let x=await r.json();add('Arti',x.reply||x.error,'arti')";
  page += "}catch(e){add('System','Connection error: '+e,'arti')}}";
  page += "</script></main></body></html>";

  server.send(
    200,
    "text/html",
    page
  );
}

static void handleStatus() {
  String json = "{";

  json += "\"model_loaded\":";
  json += modelLoaded
            ? "true"
            : "false";

  json += ",\"neural_model\":\"TinyStories-260K\"";

  json += ",\"psram\":";
  json += psramFound()
            ? "true"
            : "false";

  json += ",\"free_psram\":";
  json += String(
    ESP.getFreePsram()
  );

  json += ",\"wifi\":";
  json += WiFi.status() == WL_CONNECTED
            ? "true"
            : "false";

  json += ",\"ip\":\"";
  json += WiFi.localIP().toString();
  json += "\"";

  json += ",\"error\":\"";
  json += jsonEscape(lastError);
  json += "\"}";

  server.send(
    200,
    "application/json",
    json
  );
}

static void handleChat() {
  if (!server.hasArg("plain")) {
    server.send(
      400,
      "application/json",
      "{\"error\":\"Expected JSON body.\"}"
    );
    return;
  }

  String body =
    server.arg("plain");

  String message =
    getJsonString(
      body,
      "message"
    );

  if (message.length() == 0) {
    server.send(
      400,
      "application/json",
      "{\"error\":\"Missing message.\"}"
    );
    return;
  }

  String onlineContext;

  if (queryNeedsSearch(message)) {
    Serial.println(
      "Online search requested."
    );

    onlineContext =
      searchOnline(
        searchQueryFromMessage(
          message
        )
      );
  }

  unsigned long start =
    millis();

  String reply =
    generateNeuralReply(
      message,
      onlineContext
    );

  unsigned long elapsed =
    millis() - start;

  Serial.print(
    "Neural generation took "
  );
  Serial.print(elapsed);
  Serial.println(" ms.");

  String json =
    "{\"reply\":\"";

  json += jsonEscape(reply);
  json += "\"}";

  server.send(
    200,
    "application/json",
    json
  );
}

static void handleNotFound() {
  server.send(
    404,
    "text/plain",
    "ESP-Arti: endpoint not found."
  );
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println();
  Serial.println("==============================");
  Serial.println("ESP-Arti");
  Serial.println("ESP32 WROVER-E / Wi-Fi");
  Serial.println("LOCAL NEURAL INFERENCE");
  Serial.println("==============================");

  if (!psramFound()) {
    Serial.println(
      "ERROR: PSRAM was not detected."
    );
    lastError =
      "PSRAM is required for the neural model.";
    return;
  }

  Serial.print("PSRAM: ");
  Serial.print(
    ESP.getPsramSize()
  );
  Serial.println(" bytes");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  Serial.print(
    "Connecting to Wi-Fi"
  );

  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );

  unsigned long wifiStart =
    millis();

  while (
    WiFi.status() != WL_CONNECTED &&
    millis() - wifiStart < 30000
  ) {
    delay(400);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    lastError =
      "Wi-Fi connection failed.";
    Serial.println(lastError);
    return;
  }

  Serial.println(
    "Wi-Fi connected."
  );

  Serial.print("ESP32 IP: ");
  Serial.println(
    WiFi.localIP()
  );

  /*
    The small config file is still available from this repository, but the
    actual trained neural checkpoint and tokenizer are downloaded directly
    from GitHub at runtime.
  */
  loadNeuralModel();

  server.on(
    "/",
    HTTP_GET,
    handleRoot
  );

  server.on(
    "/status",
    HTTP_GET,
    handleStatus
  );

  server.on(
    "/chat",
    HTTP_POST,
    handleChat
  );

  server.onNotFound(
    handleNotFound
  );

  server.begin();

  Serial.println(
    "HTTP server started."
  );

  Serial.print(
    "Open http://"
  );

  Serial.print(
    WiFi.localIP()
  );

  Serial.println("/");
}

void loop() {
  server.handleClient();

  static unsigned long lastStatus =
    0;

  if (
    millis() - lastStatus > 10000
  ) {
    lastStatus = millis();

    if (
      WiFi.status() != WL_CONNECTED
    ) {
      WiFi.reconnect();
    }
  }

  delay(1);
}
