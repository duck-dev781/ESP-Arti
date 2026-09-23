/*
  ESP-Arti Wi-Fi firmware
  ------------------------
  ESP32 WROVER-E / PSRAM
  - Wi-Fi only
  - No BLE
  - No SD
  - No LittleFS
  - Browser chat endpoint: POST /chat
  - GitHub is used as the runtime model-data host.

  IMPORTANT:
  The flashed firmware contains the inference engine. GitHub-hosted files are
  runtime model/config data; ESP32 cannot compile .cpp/.h files at runtime.

  Put your Wi-Fi credentials below.
*/

#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>

static const char *WIFI_SSID = "";
static const char *WIFI_PASSWORD = "";

static const char *GITHUB_RAW_BASE =
  "https://raw.githubusercontent.com/duck-dev781/ESP-Arti/main/";

static const char *MODEL_CONFIG_PATH = "ai/config.h";

WebServer server(80);

String modelConfig;
bool modelLoaded = false;
String lastError;

static bool downloadText(const String &url, String &out) {
  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;
  http.setTimeout(15000);

  if (!http.begin(client, url)) {
    lastError = "HTTP begin failed: " + url;
    return false;
  }

  int code = http.GET();

  if (code != HTTP_CODE_OK) {
    lastError = "HTTP " + String(code) + ": " + url;
    http.end();
    return false;
  }

  out = http.getString();
  http.end();

  if (out.length() == 0) {
    lastError = "Empty file: " + url;
    return false;
  }

  return true;
}

static bool loadRuntimeFiles() {
  Serial.println();
  Serial.println("ESP-Arti runtime loader");
  Serial.println("Downloading model configuration from GitHub...");

  String cfg;

  if (!downloadText(String(GITHUB_RAW_BASE) + MODEL_CONFIG_PATH, cfg)) {
    Serial.println("MODEL LOAD FAILED");
    Serial.println(lastError);
    return false;
  }

  modelConfig = cfg;

  if (!psramFound()) {
    lastError = "PSRAM not detected.";
    Serial.println(lastError);
    return false;
  }

  size_t freePsram = ESP.getFreePsram();

  Serial.print("PSRAM detected. Free PSRAM: ");
  Serial.println(freePsram);

  if (freePsram < 1024 * 1024) {
    lastError = "Not enough free PSRAM for the configured runtime.";
    Serial.println(lastError);
    return false;
  }

  modelLoaded = true;

  Serial.println("GitHub runtime files loaded.");
  Serial.println("Model configuration:");
  Serial.println(modelConfig);

  return true;
}

static String jsonEscape(const String &s) {
  String r;
  r.reserve(s.length() + 16);

  for (size_t i = 0; i < s.length(); ++i) {
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

static String getJsonString(const String &body, const String &key) {
  String needle = "\"" + key + "\"";
  int p = body.indexOf(needle);

  if (p < 0) {
    return "";
  }

  p = body.indexOf(':', p);

  if (p < 0) {
    return "";
  }

  p++;

  while (p < (int)body.length() &&
         isspace((unsigned char)body[p])) {
    p++;
  }

  if (p >= (int)body.length() || body[p] != '"') {
    return "";
  }

  p++;

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
      } else if (c == '"' || c == '\\' || c == '/') {
        result += c;
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

/*
  Online context lookup.

  This is deliberately lightweight. It gives the inference layer a place to
  obtain temporary current-information context without changing the model.
*/
static String searchOnline(const String &query) {
  if (WiFi.status() != WL_CONNECTED) {
    return "";
  }

  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;
  http.setTimeout(12000);

  String encoded;
  encoded.reserve(query.length() * 2);

  for (size_t i = 0; i < query.length(); ++i) {
    char c = query[i];

    if (isalnum((unsigned char)c) ||
        c == '-' ||
        c == '_' ||
        c == '.') {
      encoded += c;
    } else if (c == ' ') {
      encoded += '+';
    } else {
      char buf[4];
      snprintf(buf, sizeof(buf), "%%%02X", (unsigned char)c);
      encoded += buf;
    }
  }

  String url = "https://html.duckduckgo.com/html/?q=" + encoded;

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
  size_t limit = page.length();

  if (limit > 5000) {
    limit = 5000;
  }

  plain.reserve(limit);

  bool inTag = false;

  for (size_t i = 0; i < page.length() && plain.length() < 5000; ++i) {
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

  while (plain.indexOf("  ") >= 0) {
    plain.replace("  ", " ");
  }

  return plain;
}

static bool queryNeedsSearch(const String &q) {
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

static String searchQueryFromMessage(const String &message) {
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

/*
  Temporary response path.

  The final neural generator should consume:
    - modelConfig
    - downloaded model weights
    - system/runtime context
    - user prompt
    - optional online context

  This function intentionally does NOT pretend to be a trained neural model.
*/
static String generateLocalReply(const String &userMessage,
                                 const String &onlineContext) {
  String reply;

  reply += "Arti is running locally on the ESP32 WROVER-E.\n\n";
  reply += "I received: ";
  reply += userMessage;

  if (onlineContext.length()) {
    reply += "\n\nOnline context was retrieved and is available to the local inference layer.";
  }

  reply += "\n\nModel runtime status: ";
  reply += modelLoaded
             ? "GitHub runtime loaded."
             : "model not loaded.";

  return reply;
}

static void handleRoot() {
  String page;
  page.reserve(2200);

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
      "{\"error\":\"Expected JSON body.\"}"
    );
    return;
  }

  String body = server.arg("plain");
  String message = getJsonString(body, "message");

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
    Serial.println("Online search requested.");

    onlineContext = searchOnline(
      searchQueryFromMessage(message)
    );

    if (onlineContext.length()) {
      Serial.println("Online context received.");
    } else {
      Serial.println("Online search returned no usable context.");
    }
  }

  String reply = generateLocalReply(message, onlineContext);

  String json = "{\"reply\":\"";
  json += jsonEscape(reply);
  json += "\"}";

  server.send(200, "application/json", json);
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
  Serial.println("==============================");

  if (!psramFound()) {
    Serial.println("WARNING: PSRAM was not detected.");
  } else {
    Serial.print("PSRAM: ");
    Serial.print(ESP.getPsramSize());
    Serial.println(" bytes");
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  Serial.print("Connecting to Wi-Fi");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED &&
         millis() - start < 30000) {
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

  loadRuntimeFiles();

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

  static unsigned long lastStatus = 0;

  if (millis() - lastStatus > 10000) {
    lastStatus = millis();

    if (WiFi.status() != WL_CONNECTED) {
      WiFi.reconnect();
    }
  }

  delay(1);
}
