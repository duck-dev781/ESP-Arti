/*
  ESP-Arti Wi-Fi firmware
  ------------------------
  ESP32 WROVER-E / PSRAM

  This firmware is the PERMANENT runtime/inference engine.
  The trained Arti model is NOT stored permanently in the ESP32.

  On every boot:
    1. Connect to Wi-Fi.
    2. Download our current project-owned model from GitHub.
    3. Put the model in PSRAM.
    4. Run neural inference locally on the ESP32.

  A reset/power loss clears PSRAM, so the model is downloaded again.

  Model format:
    ESPARTI1
    byte vocabulary (256)
    context 128
    d_model 32
    1 Transformer block
    4 attention heads
    FFN 64
    float32 weights

  No BLE.
  No SD.
  No LittleFS.
  No external AI inference API.
*/

#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static const char *WIFI_SSID = "";
static const char *WIFI_PASSWORD = "";

static const char *MODEL_URL =
  "https://raw.githubusercontent.com/duck-dev781/ESP-Arti/main/ai/arti-v1.bin";

static const int MODEL_VOCAB = 256;
static const int MODEL_CONTEXT = 128;
static const int MODEL_DIM = 32;
static const int MODEL_LAYERS = 1;
static const int MODEL_HEADS = 4;
static const int MODEL_FFN = 64;

static const int MAX_PROMPT_BYTES = 120;
static const int MAX_GENERATION_BYTES = 256;

WebServer server(80);

static uint8_t *modelData = nullptr;
static size_t modelDataSize = 0;
static bool modelLoaded = false;
static String lastError;

struct ModelHeader {
  char magic[8];
  uint32_t version;
  uint32_t vocab;
  uint32_t context;
  uint32_t dim;
  uint32_t layers;
  uint32_t heads;
  uint32_t ffn;
};

struct ModelWeights {
  float *embedding;
  float *ln1;
  float *q;
  float *k;
  float *v;
  float *o;
  float *ln2;
  float *ff1;
  float *ff2;
  float *lnFinal;
  float *output;
};

static ModelHeader header;
static ModelWeights weights;

static float *x = nullptr;
static float *xb = nullptr;
static float *xb2 = nullptr;
static float *q = nullptr;
static float *k = nullptr;
static float *v = nullptr;
static float *att = nullptr;
static float *hb = nullptr;
static float *hb2 = nullptr;
static float *logits = nullptr;
static float *keyCache = nullptr;
static float *valueCache = nullptr;

static bool allocateRuntimeMemory() {
  x = (float *)ps_malloc(MODEL_DIM * sizeof(float));
  xb = (float *)ps_malloc(MODEL_DIM * sizeof(float));
  xb2 = (float *)ps_malloc(MODEL_DIM * sizeof(float));
  q = (float *)ps_malloc(MODEL_DIM * sizeof(float));
  k = (float *)ps_malloc(MODEL_DIM * sizeof(float));
  v = (float *)ps_malloc(MODEL_DIM * sizeof(float));
  att = (float *)ps_malloc(MODEL_HEADS * MODEL_CONTEXT * sizeof(float));
  hb = (float *)ps_malloc(MODEL_FFN * sizeof(float));
  hb2 = (float *)ps_malloc(MODEL_FFN * sizeof(float));
  logits = (float *)ps_malloc(MODEL_VOCAB * sizeof(float));

  const size_t cacheFloats =
    (size_t)MODEL_LAYERS *
    MODEL_CONTEXT *
    MODEL_DIM;

  keyCache = (float *)ps_malloc(cacheFloats * sizeof(float));
  valueCache = (float *)ps_malloc(cacheFloats * sizeof(float));

  if (!x || !xb || !xb2 || !q || !k || !v ||
      !att || !hb || !hb2 || !logits ||
      !keyCache || !valueCache) {
    lastError = "Could not allocate neural runtime buffers in PSRAM.";
    return false;
  }

  return true;
}

static bool downloadModel() {
  modelLoaded = false;
  lastError = "";

  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;
  http.setTimeout(30000);

  Serial.println();
  Serial.println("=== ESP-Arti model loader ===");
  Serial.println("Downloading OUR Arti model from GitHub...");
  Serial.println(MODEL_URL);

  if (!http.begin(client, MODEL_URL)) {
    lastError = "Could not start HTTPS model download.";
    Serial.println(lastError);
    return false;
  }

  int code = http.GET();

  if (code != HTTP_CODE_OK) {
    lastError = String("Model download HTTP error: ") + code;
    Serial.println(lastError);
    http.end();
    return false;
  }

  int contentLength = http.getSize();

  if (contentLength <= 0) {
    lastError = "GitHub did not provide a model content length.";
    Serial.println(lastError);
    http.end();
    return false;
  }

  if ((size_t)contentLength > 1024 * 1024) {
    lastError = "Model is larger than the 1 MB safety limit.";
    Serial.println(lastError);
    http.end();
    return false;
  }

  if (modelData) {
    free(modelData);
    modelData = nullptr;
    modelDataSize = 0;
  }

  modelDataSize = (size_t)contentLength;
  modelData = (uint8_t *)ps_malloc(modelDataSize);

  if (!modelData) {
    lastError = String("Could not allocate ") +
                modelDataSize +
                " bytes for the model in PSRAM.";
    Serial.println(lastError);
    http.end();
    return false;
  }

  WiFiClient *stream = http.getStreamPtr();
  size_t received = 0;
  unsigned long lastData = millis();

  while (received < modelDataSize) {
    size_t availableBytes = stream->available();

    if (availableBytes > 0) {
      size_t wanted = modelDataSize - received;

      if (availableBytes > wanted) {
        availableBytes = wanted;
      }

      int n = stream->readBytes(
        modelData + received,
        availableBytes
      );

      if (n > 0) {
        received += (size_t)n;
        lastData = millis();
      }
    } else {
      if (millis() - lastData > 30000) {
        lastError = "Timed out while downloading the Arti model.";
        free(modelData);
        modelData = nullptr;
        modelDataSize = 0;
        http.end();
        Serial.println(lastError);
        return false;
      }

      delay(2);
    }
  }

  http.end();

  if (received != modelDataSize) {
    lastError = "Model download was incomplete.";
    free(modelData);
    modelData = nullptr;
    modelDataSize = 0;
    Serial.println(lastError);
    return false;
  }

  Serial.print("Downloaded ");
  Serial.print(modelDataSize);
  Serial.println(" bytes into PSRAM.");

  if (modelDataSize < sizeof(ModelHeader)) {
    lastError = "Downloaded model is too small.";
    return false;
  }

  memcpy(&header, modelData, sizeof(ModelHeader));

  if (memcmp(header.magic, "ESPARTI1", 8) != 0) {
    lastError = "Downloaded file is not an ESP-Arti model.";
    return false;
  }

  if (header.version != 1 ||
      header.vocab != MODEL_VOCAB ||
      header.context != MODEL_CONTEXT ||
      header.dim != MODEL_DIM ||
      header.layers != MODEL_LAYERS ||
      header.heads != MODEL_HEADS ||
      header.ffn != MODEL_FFN) {
    lastError = "Downloaded model configuration does not match this firmware.";
    return false;
  }

  const size_t expectedWeights =
    (size_t)MODEL_VOCAB * MODEL_DIM +
    (size_t)MODEL_DIM +
    (size_t)MODEL_DIM * MODEL_DIM +
    (size_t)MODEL_DIM * MODEL_DIM +
    (size_t)MODEL_DIM * MODEL_DIM +
    (size_t)MODEL_DIM * MODEL_DIM +
    (size_t)MODEL_DIM +
    (size_t)MODEL_DIM * MODEL_FFN +
    (size_t)MODEL_FFN * MODEL_DIM +
    (size_t)MODEL_DIM +
    (size_t)MODEL_DIM * MODEL_VOCAB;

  const size_t expectedBytes =
    sizeof(ModelHeader) +
    expectedWeights * sizeof(float);

  if (modelDataSize != expectedBytes) {
    lastError =
      String("Wrong model size. Expected ") +
      expectedBytes +
      ", got " +
      modelDataSize;
    Serial.println(lastError);
    return false;
  }

  uint8_t *ptr = modelData + sizeof(ModelHeader);

  weights.embedding = (float *)ptr;
  ptr += MODEL_VOCAB * MODEL_DIM * sizeof(float);

  weights.ln1 = (float *)ptr;
  ptr += MODEL_DIM * sizeof(float);

  weights.q = (float *)ptr;
  ptr += MODEL_DIM * MODEL_DIM * sizeof(float);

  weights.k = (float *)ptr;
  ptr += MODEL_DIM * MODEL_DIM * sizeof(float);

  weights.v = (float *)ptr;
  ptr += MODEL_DIM * MODEL_DIM * sizeof(float);

  weights.o = (float *)ptr;
  ptr += MODEL_DIM * MODEL_DIM * sizeof(float);

  weights.ln2 = (float *)ptr;
  ptr += MODEL_DIM * sizeof(float);

  weights.ff1 = (float *)ptr;
  ptr += MODEL_DIM * MODEL_FFN * sizeof(float);

  weights.ff2 = (float *)ptr;
  ptr += MODEL_FFN * MODEL_DIM * sizeof(float);

  weights.lnFinal = (float *)ptr;
  ptr += MODEL_DIM * sizeof(float);

  weights.output = (float *)ptr;

  memset(
    keyCache,
    0,
    (size_t)MODEL_LAYERS *
    MODEL_CONTEXT *
    MODEL_DIM *
    sizeof(float)
  );

  memset(
    valueCache,
    0,
    (size_t)MODEL_LAYERS *
    MODEL_CONTEXT *
    MODEL_DIM *
    sizeof(float)
  );

  modelLoaded = true;

  Serial.println("OUR Arti model loaded into PSRAM.");
  Serial.print("Free PSRAM after load: ");
  Serial.println(ESP.getFreePsram());

  return true;
}

static void rmsNorm(
  float *out,
  const float *input,
  const float *weight
) {
  float sum = 0.0f;

  for (int i = 0; i < MODEL_DIM; ++i) {
    sum += input[i] * input[i];
  }

  float scale =
    1.0f / sqrtf(
      sum / (float)MODEL_DIM + 1e-5f
    );

  for (int i = 0; i < MODEL_DIM; ++i) {
    out[i] = input[i] * scale * weight[i];
  }
}

static void matmul(
  float *out,
  const float *input,
  const float *matrix,
  int inputSize,
  int outputSize
) {
  for (int row = 0; row < outputSize; ++row) {
    float sum = 0.0f;

    for (int col = 0; col < inputSize; ++col) {
      sum += matrix[row * inputSize + col] * input[col];
    }

    out[row] = sum;
  }
}

static void softmax(float *values, int count) {
  float maxValue = values[0];

  for (int i = 1; i < count; ++i) {
    if (values[i] > maxValue) {
      maxValue = values[i];
    }
  }

  float sum = 0.0f;

  for (int i = 0; i < count; ++i) {
    values[i] = expf(values[i] - maxValue);
    sum += values[i];
  }

  if (sum <= 0.0f) {
    return;
  }

  for (int i = 0; i < count; ++i) {
    values[i] /= sum;
  }
}

static void forwardToken(
  uint8_t token,
  int position
) {
  const int headSize = MODEL_DIM / MODEL_HEADS;

  memcpy(
    x,
    weights.embedding +
      (size_t)token * MODEL_DIM,
    MODEL_DIM * sizeof(float)
  );

  rmsNorm(xb, x, weights.ln1);

  matmul(q, xb, weights.q, MODEL_DIM, MODEL_DIM);
  matmul(k, xb, weights.k, MODEL_DIM, MODEL_DIM);
  matmul(v, xb, weights.v, MODEL_DIM, MODEL_DIM);

  float *keyAtPosition =
    keyCache +
    (size_t)position * MODEL_DIM;

  float *valueAtPosition =
    valueCache +
    (size_t)position * MODEL_DIM;

  memcpy(
    keyAtPosition,
    k,
    MODEL_DIM * sizeof(float)
  );

  memcpy(
    valueAtPosition,
    v,
    MODEL_DIM * sizeof(float)
  );

  for (int head = 0;
       head < MODEL_HEADS;
       ++head) {

    float *qHead = q + head * headSize;
    float *attention =
      att + head * MODEL_CONTEXT;

    for (int t = 0; t <= position; ++t) {
      const float *kHead =
        keyCache +
        (size_t)t * MODEL_DIM +
        head * headSize;

      float score = 0.0f;

      for (int j = 0; j < headSize; ++j) {
        score += qHead[j] * kHead[j];
      }

      attention[t] =
        score / sqrtf((float)headSize);
    }

    softmax(attention, position + 1);

    float *outHead =
      xb + head * headSize;

    memset(
      outHead,
      0,
      headSize * sizeof(float)
    );

    for (int t = 0; t <= position; ++t) {
      const float *vHead =
        valueCache +
        (size_t)t * MODEL_DIM +
        head * headSize;

      for (int j = 0; j < headSize; ++j) {
        outHead[j] +=
          attention[t] * vHead[j];
      }
    }
  }

  matmul(
    xb2,
    xb,
    weights.o,
    MODEL_DIM,
    MODEL_DIM
  );

  for (int i = 0; i < MODEL_DIM; ++i) {
    x[i] += xb2[i];
  }

  rmsNorm(xb, x, weights.ln2);

  matmul(
    hb,
    xb,
    weights.ff1,
    MODEL_DIM,
    MODEL_FFN
  );

  for (int i = 0; i < MODEL_FFN; ++i) {
    // GELU approximation matching the training model.
    float z = hb[i];
    float gelu =
      0.5f * z *
      (1.0f + tanhf(
        0.79788456f *
        (z + 0.044715f * z * z * z)
      ));
    hb[i] = gelu;
  }

  // The second feed-forward matrix maps FFN -> model dimension.
  matmul(
    xb,
    hb,
    weights.ff2,
    MODEL_FFN,
    MODEL_DIM
  );

  for (int i = 0; i < MODEL_DIM; ++i) {
    x[i] += xb[i];
  }

  rmsNorm(
    x,
    x,
    weights.lnFinal
  );

  matmul(
    logits,
    x,
    weights.output,
    MODEL_DIM,
    MODEL_VOCAB
  );
}

static uint8_t chooseNextToken() {
  int best = 0;
  float bestValue = logits[0];

  for (int i = 1; i < MODEL_VOCAB; ++i) {
    if (logits[i] > bestValue) {
      bestValue = logits[i];
      best = i;
    }
  }

  return (uint8_t)best;
}

static String generateNeuralReply(
  const String &message
) {
  if (!modelLoaded) {
    return "The Arti model is not loaded. Check Serial Monitor.";
  }

  memset(
    keyCache,
    0,
    (size_t)MODEL_CONTEXT *
    MODEL_DIM *
    sizeof(float)
  );

  memset(
    valueCache,
    0,
    (size_t)MODEL_CONTEXT *
    MODEL_DIM *
    sizeof(float)
  );

  String prompt =
    "You are Arti, a local neural assistant running on an ESP32 WROVER-E. "
    "Your model runs locally. Follow the user's current role. "
    "User: ";

  prompt += message;
  prompt += "\nArti:";

  if (prompt.length() > MAX_PROMPT_BYTES) {
    prompt =
      prompt.substring(
        prompt.length() - MAX_PROMPT_BYTES
      );
  }

  int position = 0;

  for (size_t i = 0;
       i < prompt.length() &&
       position < MODEL_CONTEXT - 1;
       ++i) {

    forwardToken(
      (uint8_t)prompt[i],
      position
    );

    ++position;
  }

  String reply;
  reply.reserve(MAX_GENERATION_BYTES);

  for (int i = 0;
       i < MAX_GENERATION_BYTES &&
       position < MODEL_CONTEXT;
       ++i) {

    uint8_t next =
      chooseNextToken();

    if (next == 0 || next == '\n') {
      break;
    }

    char c = (char)next;

    if (isprint((unsigned char)c) ||
        c == '\n' ||
        c == '\r' ||
        c == '\t') {
      reply += c;
    }

    forwardToken(
      next,
      position
    );

    ++position;
  }

  if (reply.length() == 0) {
    return "(Arti generated no text.)";
  }

  return reply;
}

static String jsonEscape(
  const String &value
) {
  String out;
  out.reserve(value.length() + 16);

  for (size_t i = 0; i < value.length(); ++i) {
    char c = value[i];

    if (c == '\\') {
      out += "\\\\";
    } else if (c == '"') {
      out += "\\\"";
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else if (c == '\t') {
      out += "\\t";
    } else if ((unsigned char)c < 32) {
      out += ' ';
    } else {
      out += c;
    }
  }

  return out;
}

static String getJsonString(
  const String &body,
  const String &key
) {
  String needle =
    String("\"") + key + "\"";

  int start = body.indexOf(needle);

  if (start < 0) {
    return "";
  }

  int colon = body.indexOf(':', start);

  if (colon < 0) {
    return "";
  }

  int quote = body.indexOf('"', colon + 1);

  if (quote < 0) {
    return "";
  }

  String result;
  bool escaped = false;

  for (int i = quote + 1;
       i < (int)body.length();
       ++i) {

    char c = body[i];

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

static void handleRoot() {
  String page;
  page.reserve(2600);

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

  server.send(200, "text/html", page);
}

static void handleStatus() {
  String json = "{";

  json += "\"model_loaded\":";
  json += modelLoaded ? "true" : "false";

  json += ",\"model\":\"ESP-Arti project-owned neural model\"";

  json += ",\"psram\":";
  json += psramFound() ? "true" : "false";

  json += ",\"free_psram\":";
  json += String(ESP.getFreePsram());

  json += ",\"wifi\":";
  json += WiFi.status() == WL_CONNECTED ? "true" : "false";

  json += ",\"ip\":\"";
  json += WiFi.localIP().toString();
  json += "\"";

  json += ",\"error\":\"";
  json += jsonEscape(lastError);
  json += "\"}";

  server.send(200, "application/json", json);
}

static void handleChat() {
  if (!server.hasArg("plain")) {
    server.send(
      400,
      "application/json",
      " {\"error\":\"Expected JSON body.\"}"
    );
    return;
  }

  String message =
    getJsonString(
      server.arg("plain"),
      "message"
    );

  if (message.length() == 0) {
    server.send(
      400,
      "application/json",
      " {\"error\":\"Missing message.\"}"
    );
    return;
  }

  unsigned long start = millis();

  String reply =
    generateNeuralReply(message);

  unsigned long elapsed =
    millis() - start;

  Serial.print("Inference time: ");
  Serial.print(elapsed);
  Serial.println(" ms");

  String json =
    String("{\"reply\":\"") +
    jsonEscape(reply) +
    "\"}";

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
  Serial.println("PROJECT-OWNED LOCAL AI");
  Serial.println("==============================");

  if (!psramFound()) {
    lastError =
      "PSRAM was not detected. It is required.";
    Serial.println(lastError);
    return;
  }

  Serial.print("PSRAM total: ");
  Serial.println(ESP.getPsramSize());

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  Serial.print("Connecting to Wi-Fi");

  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );

  unsigned long start = millis();

  while (
    WiFi.status() != WL_CONNECTED &&
    millis() - start < 30000
  ) {
    delay(400);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    lastError = "Wi-Fi connection failed.";
    Serial.println(lastError);
    return;
  }

  Serial.println("Wi-Fi connected.");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  if (!allocateRuntimeMemory()) {
    Serial.println(lastError);
    return;
  }

  if (!downloadModel()) {
    Serial.println();
    Serial.println("MODEL LOAD FAILED:");
    Serial.println(lastError);
  }

  server.on("/", HTTP_GET, handleRoot);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/chat", HTTP_POST, handleChat);
  server.onNotFound(handleNotFound);

  server.begin();

  Serial.println("HTTP server started.");
  Serial.print("Open http://");
  Serial.print(WiFi.localIP());
  Serial.println("/");
}

void loop() {
  server.handleClient();

  static unsigned long lastWiFiCheck = 0;

  if (millis() - lastWiFiCheck > 10000) {
    lastWiFiCheck = millis();

    if (WiFi.status() != WL_CONNECTED) {
      WiFi.reconnect();
    }
  }

  delay(1);
}
