#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <esp_heap_caps.h>
#include <string.h>
#include "llm.h"
#include "tokenizer_data.h"

static const char *WIFI_SSID = "";
static const char *WIFI_PASSWORD = "";

static const char *MODEL_URL = "https://raw.githubusercontent.com/duck-dev781/ESP-Arti/main/ai/arti.bin";

static constexpr size_t MODEL_LIMIT = 7 * 1024 * 1024;
static constexpr int KV_WINDOW = 72;
static constexpr int MAX_PROMPT_TOKENS = 58;
static constexpr int MAX_REPLY_TOKENS = 24;
static constexpr int MAX_MESSAGE_CHARS = 180;
static constexpr int MAX_ROLE_CHARS = 180;

WebServer server(80);
Transformer transformer{};
Tokenizer tokenizer{};
Sampler sampler{};
uint8_t *modelBlob = nullptr;
uint8_t *tokenizerBlob = nullptr;
size_t modelSize = 0;
size_t tokenizerSize = 0;
bool modelReady = false;
String lastError;
String roleText = "You are Arti, a modern AI assistant. Follow the user's instructions, stay on topic, and adapt your tone and personality to what the user asks.";
String histUser[2];
String histBot[2];
int histCount = 0;

static String jsonEscape(const String &s) {
  String out; out.reserve(s.length()+16);
  for (size_t i=0;i<s.length();++i) {
    char c=s[i];
    if(c=='\\') out+="\\\\";
    else if(c=='"') out+="\\\"";
    else if(c=='\n') out+="\\n";
    else if(c=='\r') out+="\\r";
    else if(c=='\t') out+="\\t";
    else if((uint8_t)c<32) out+=' ';
    else out+=c;
  }
  return out;
}

static String jsonString(const String &body,const String &key){
  String needle="\""+key+"\""; int p=body.indexOf(needle); if(p<0)return "";
  int colon=body.indexOf(':',p); if(colon<0)return "";
  int q=body.indexOf('"',colon+1); if(q<0)return "";
  String r; bool esc=false;
  for(int i=q+1;i<(int)body.length();++i){
    char c=body[i];
    if(esc){
      if(c=='n')r+='\n'; else if(c=='r')r+='\r'; else if(c=='t')r+='\t'; else r+=c;
      esc=false; continue;
    }
    if(c=='\\'){esc=true;continue;}
    if(c=='"')break;
    r+=c;
  }
  return r;
}

static bool downloadBlob(const char *url,uint8_t **dst,size_t *outSize,size_t limit,const char *label){
  WiFiClientSecure client; client.setInsecure();
  HTTPClient http; http.setTimeout(60000);
  Serial.printf("Downloading %s...\n%s\n",label,url);
  if(!http.begin(client,url)){lastError=String("Could not start ")+label+" download.";return false;}
  int code=http.GET();
  if(code!=HTTP_CODE_OK){lastError=String(label)+" HTTP error "+code;http.end();return false;}
  int len=http.getSize();
  if(len<=0 || (size_t)len>limit){lastError=String(label)+" has an invalid size.";http.end();return false;}
  uint8_t *buf=(uint8_t*)ps_malloc((size_t)len);
  if(!buf){lastError=String("Not enough PSRAM for ")+label+".";http.end();return false;}
  WiFiClient *stream=http.getStreamPtr(); size_t got=0; unsigned long last=millis();
  while(got<(size_t)len){
    size_t a=stream->available();
    if(a){
      size_t want=(size_t)len-got;
      if(a>want)a=want;
      int n=stream->readBytes(buf+got,a);
      if(n>0){got+=(size_t)n;last=millis();}
    } else {
      if(millis()-last>60000){
        free(buf);http.end();lastError=String("Timed out downloading ")+label+".";return false;
      }
      delay(2);
    }
  }
  http.end(); *dst=buf;*outSize=got;
  Serial.printf("%s: %u bytes\n",label,(unsigned)got);
  return true;
}

static bool loadModel(){
  modelReady=false; lastError="";
  if(modelBlob){free(modelBlob);modelBlob=nullptr;}
  if(!downloadBlob(MODEL_URL,&modelBlob,&modelSize,MODEL_LIMIT,"TinyTalk 2 model"))return false;
  if(!llm_init_embedded(&transformer,modelBlob,modelSize,KV_WINDOW)){
    lastError="TinyTalk model header/runtime initialization failed.";return false;
  }
  if(!llm_tokenizer_from_memory(&tokenizer,TINY_TALK_TOKENIZER_DATA,TINY_TALK_TOKENIZER_SIZE,transformer.config.vocab_size)){
    lastError="Embedded TinyTalk tokenizer initialization failed.";return false;
  }
  if(tokenizer.vocab_size!=transformer.config.vocab_size){
    lastError="Model/tokenizer vocabulary mismatch.";return false;
  }
  llm_build_sampler(&sampler,transformer.config.vocab_size,0.8f,0.9f,esp_random());
  modelReady=true;
  Serial.printf("TinyTalk ready: dim=%d layers=%d vocab=%d context=%d\n",transformer.config.dim,transformer.config.n_layers,transformer.config.vocab_size,transformer.config.seq_len);
  Serial.printf("Free PSRAM: %u\n",(unsigned)ESP.getFreePsram());
  return true;
}

static void resetHistory(){
  for(int i=0;i<2;i++){histUser[i]="";histBot[i]="";}
  histCount=0;
}

static void pushHistory(const String &u,const String &b){
  if(histCount<2){
    histUser[histCount]=u;histBot[histCount]=b;histCount++;
  }else{
    histUser[0]=histUser[1];histBot[0]=histBot[1];
    histUser[1]=u;histBot[1]=b;
  }
}

static bool appendTextTokens(const String &s,int *tokens,int *n){
  if(*n>=MAX_PROMPT_TOKENS)return false;
  int cap=s.length()+8;
  int *tmp=(int*)malloc(sizeof(int)*cap);
  if(!tmp)return false;
  int m=0;
  llm_encode(&tokenizer,s.c_str(),0,0,tmp,&m);
  bool ok=*n+m<=MAX_PROMPT_TOKENS;
  if(ok){memcpy(tokens+*n,tmp,sizeof(int)*m);*n+=m;}
  free(tmp);
  return ok;
}

static String generateReply(const String &message){
  if(!modelReady)return "Arti is not loaded: "+lastError;
  int tokens[MAX_PROMPT_TOKENS]; int n=0;
  String role=roleText;
  if(role.length()>MAX_ROLE_CHARS)role=role.substring(0,MAX_ROLE_CHARS);

  String prompt="Role: "+role+"\n";
  for(int i=0;i<histCount;i++){
    prompt += "User: "+histUser[i]+"\nBot: "+histBot[i]+"\n";
  }
  prompt += "User: "+message+"\nBot:";

  if(!appendTextTokens(prompt,tokens,&n)){
    n=0;
    String shortPrompt="Role: "+role+"\nUser: "+message+"\nBot:";
    appendTextTokens(shortPrompt,tokens,&n);
  }
  if(n<1)return "I couldn't prepare that message.";

  memset(transformer.state.key_cache4,0,(size_t)transformer.config.n_layers*transformer.kv_seq_len*(transformer.config.dim/2));
  memset(transformer.state.value_cache4,0,(size_t)transformer.config.n_layers*transformer.kv_seq_len*(transformer.config.dim/2));
  memset(transformer.state.k_gscales,0,(size_t)transformer.config.n_layers*transformer.kv_seq_len*(transformer.config.dim/32)*sizeof(uint16_t));
  memset(transformer.state.v_gscales,0,(size_t)transformer.config.n_layers*transformer.kv_seq_len*(transformer.config.dim/32)*sizeof(uint16_t));

  int pos=0; float *logits=nullptr;
  for(int i=0;i<n && pos<KV_WINDOW;i++){
    logits=llm_forward(&transformer,tokens[i],pos);
    pos++;
  }

  String reply; reply.reserve(220);
  int prev=tokens[n-1];
  for(int step=0;step<MAX_REPLY_TOKENS && pos<KV_WINDOW;step++){
    int next=llm_sample(&sampler,logits);
    if(next==tokenizer.eos_id)break;
    char scratch[64];
    const char *piece=llm_decode(&tokenizer,prev,next,scratch,sizeof(scratch));
    if(piece && *piece){
      if(reply.length()+strlen(piece)>220)break;
      reply+=piece;
      if(reply.indexOf("\nUser:")>=0)break;
    }
    prev=next;
    logits=llm_forward(&transformer,next,pos);
    pos++;
  }
  reply.trim();
  if(reply.length()==0)reply="(Arti did not generate a reply.)";
  return reply;
}

static void handleRoot(){
  String p="<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'><title>ESP-Arti</title><style>body{font-family:system-ui;background:#111;color:#eee;margin:0}main{max-width:850px;margin:auto;padding:16px}input,textarea{box-sizing:border-box;width:100%;padding:11px;border-radius:9px;border:1px solid #555;background:#181818;color:#fff}textarea{min-height:70px;resize:vertical}#chat{height:55vh;overflow:auto;border:1px solid #444;border-radius:12px;padding:12px}.msg{padding:10px;margin:8px 0;border-radius:10px;background:#222;white-space:pre-wrap}.user{background:#333}button{padding:11px 16px;border:0;border-radius:9px;margin-top:8px}form{display:flex;gap:8px;margin-top:10px}form input{flex:1}.small{font-size:.85em;color:#aaa}</style></head><body><main><h1>ESP-Arti</h1><p class='small'>Published TinyTalk 2 runtime • local inference • model downloaded at boot</p><label>Role / personality</label><textarea id='role'></textarea><button onclick='setRole()'>Set role</button><div id='chat'></div><form onsubmit='sendMsg(event)'><input id='msg' autocomplete='off' placeholder='Talk to Arti...'><button>Send</button></form><script>const c=document.getElementById('chat'),r=document.getElementById('role');async function boot(){try{let x=await (await fetch('/status')).json();r.value=x.role||'';add('System',x.model+' • '+(x.model_loaded?'ready':'not ready'),'arti')}catch(e){}}function add(w,t,k){let d=document.createElement('div');d.className='msg '+k;d.textContent=w+': '+t;c.appendChild(d);c.scrollTop=c.scrollHeight}async function setRole(){await fetch('/role',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({role:r.value})});add('System','Role updated.','arti')}async function sendMsg(e){e.preventDefault();let i=document.getElementById('msg'),t=i.value.trim();if(!t)return;i.value='';add('You',t,'user');try{let x=await (await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})})).json();add('Arti',x.reply||x.error,'arti')}catch(e){add('System','Connection error: '+e,'arti')}}boot();</script></main></body></html>";
  server.send(200,"text/html",p);
}

static void handleStatus(){
  String j="{\"model_loaded\":"+(modelReady?String("true"):String("false"))+",\"model\":\"TinyTalk 2 8M Q4\",\"role\":\""+jsonEscape(roleText)+"\",\"free_psram\":"+String(ESP.getFreePsram())+",\"ip\":\""+WiFi.localIP().toString()+"\",\"error\":\""+jsonEscape(lastError)+"\"}";
  server.send(200,"application/json",j);
}

static void handleRole(){
  if(!server.hasArg("plain")){server.send(400,"application/json","{\"error\":\"Expected JSON.\"}");return;}
  String r=jsonString(server.arg("plain"),"role");
  r.trim();
  if(!r.length()){server.send(400,"application/json","{\"error\":\"Role cannot be empty.\"}");return;}
  if(r.length()>MAX_ROLE_CHARS)r=r.substring(0,MAX_ROLE_CHARS);
  roleText=r;resetHistory();
  server.send(200,"application/json","{\"ok\":true}");
}

static void handleChat(){
  if(!server.hasArg("plain")){server.send(400,"application/json","{\"error\":\"Expected JSON.\"}");return;}
  String m=jsonString(server.arg("plain"),"message");
  m.trim();
  if(!m.length()){server.send(400,"application/json","{\"error\":\"Missing message.\"}");return;}
  if(m.length()>MAX_MESSAGE_CHARS)m=m.substring(0,MAX_MESSAGE_CHARS);
  unsigned long t=millis();
  String reply=generateReply(m);
  unsigned long ms=millis()-t;
  Serial.printf("Inference: %lu ms\n",ms);
  pushHistory(m,reply);
  server.send(200,"application/json","{\"reply\":\""+jsonEscape(reply)+"\"}");
}

void setup(){
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== ESP-Arti TinyTalk 2 ===");
  Serial.printf("PSRAM: %u bytes\n",(unsigned)ESP.getPsramSize());
  if(!psramFound()){lastError="PSRAM is required.";Serial.println(lastError);return;}
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID,WIFI_PASSWORD);
  Serial.print("Wi-Fi");
  unsigned long start=millis();
  while(WiFi.status()!=WL_CONNECTED && millis()-start<30000){delay(400);Serial.print('.');}
  Serial.println();
  if(WiFi.status()!=WL_CONNECTED){lastError="Wi-Fi connection failed.";return;}
  Serial.print("IP: ");Serial.println(WiFi.localIP());
  if(!loadModel())Serial.println("MODEL ERROR: "+lastError);
  server.on("/",HTTP_GET,handleRoot);
  server.on("/status",HTTP_GET,handleStatus);
  server.on("/chat",HTTP_POST,handleChat);
  server.on("/role",HTTP_POST,handleRole);
  server.begin();
  Serial.println("HTTP server started.");
}

void loop(){server.handleClient();delay(1);}
