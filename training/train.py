"""Train/export the first ESP-Arti byte-level neural LM.

This is intentionally a small reference trainer. It produces the exact tensor
order consumed by ESP-Arti.ino. The corpus is general language/instruction text;
current facts should come from online retrieval at runtime rather than being
stuffed into the model weights.

Run locally or from the manual GitHub Actions workflow.
"""
import os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)
random.seed(7)

VOCAB=256
CONTEXT=256
D=32
HEADS=4
FF=64
LAYERS=1
STEPS=2500
LR=3e-3

CORPUS = r'''Arti is a small local computer program that runs on an ESP32. It reads
language as a sequence of bytes and predicts what text should come next.
A language model learns patterns in text. It does not need a fixed personality.
A user can give an assistant a role such as tutor, pirate, programmer, or helper.
The current instruction should guide the response.
A good assistant should explain things clearly, admit uncertainty, and avoid
making up facts. Current information can be retrieved from the internet and
placed into the current context without changing the model weights.

Question: What is a computer?
Answer: A computer is a machine that processes information according to instructions.
Question: What is a program?
Answer: A program is a set of instructions that a computer can execute.
Question: Explain math.
Answer: Mathematics uses symbols, definitions, and logical rules to describe quantities and relationships.
Question: Explain science.
Answer: Science uses observation, measurement, experiments, and reasoning to study the natural world.
Question: You are a pirate assistant.
Answer: Aye! I can speak like a pirate while still trying to answer the question clearly.
Question: Now you are a math tutor.
Answer: I will help solve the problem step by step and explain why each step works.
Question: What should you do when information may be current?
Answer: Use the available online information and distinguish retrieved facts from the model's own knowledge.
Question: What should you do if you do not know?
Answer: Say that you are uncertain instead of inventing an answer.
'''

# Repeat corpus to make the tiny training set less sparse.
text = (CORPUS + "\\n") * 80
data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(D)
        self.q=nn.Linear(D,D,bias=False)
        self.k=nn.Linear(D,D,bias=False)
        self.v=nn.Linear(D,D,bias=False)
        self.o=nn.Linear(D,D,bias=False)
        self.ln2=nn.LayerNorm(D)
        self.ff1=nn.Linear(D,FF,bias=False)
        self.ff2=nn.Linear(FF,D,bias=False)
    def forward(self,x):
        B,T,C=x.shape
        h=self.ln1(x)
        q=self.q(h).view(B,T,HEADS,C//HEADS).transpose(1,2)
        k=self.k(h).view(B,T,HEADS,C//HEADS).transpose(1,2)
        v=self.v(h).view(B,T,HEADS,C//HEADS).transpose(1,2)
        a=(q@k.transpose(-2,-1))/math.sqrt(C//HEADS)
        mask=torch.triu(torch.ones(T,T,device=x.device),1).bool()
        a=a.masked_fill(mask,-1e9).softmax(-1)
        h=(a@v).transpose(1,2).contiguous().view(B,T,C)
        x=x+self.o(h)
        x=x+self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(VOCAB,D)
        self.blocks=nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln=nn.LayerNorm(D)
        self.out=nn.Linear(D,VOCAB,bias=False)
    def forward(self,idx,targets=None):
        x=self.emb(idx)
        for b in self.blocks: x=b(x)
        logits=self.out(self.ln(x))
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1))
        return logits,loss

m=Model()
opt=torch.optim.AdamW(m.parameters(),lr=LR)

for step in range(STEPS):
    ix=torch.randint(0,len(data)-CONTEXT-1,(8,))
    x=torch.stack([data[i:i+CONTEXT] for i in ix])
    y=torch.stack([data[i+1:i+CONTEXT+1] for i in ix])
    _,loss=m(x,y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step%250==0: print('step',step,'loss',float(loss))

os.makedirs('ai',exist_ok=True)
with open('ai/config.h','w') as f:
    f.write('# Generated ESP-Arti runtime model\\n')
    f.write('ESPARTI_FORMAT=1\\nVOCAB=256\\nCONTEXT=256\\nD_MODEL=32\\nLAYERS=1\\nHEADS=4\\nFFN=64\\nWEIGHT_FILES=1\\nWEIGHTS=24768\\n')
with open('ai/tokenizer.h','w') as f:
    f.write('# Byte tokenizer\\nBYTE_TOKENIZER=1\\nVOCAB=256\\n')

# Exact order consumed by the firmware.
tensors=[m.emb.weight]
for b in m.blocks:
    tensors += [b.ln1.weight,b.ln1.bias,b.q.weight,b.k.weight,b.v.weight,b.o.weight,
                b.ln2.weight,b.ln2.bias,b.ff1.weight,b.ff2.weight]
tensors += [m.ln.weight,m.ln.bias,m.out.weight]

with open('ai/weights_00.h','w') as f:
    f.write('# Generated float32 runtime weights; one value per line\\n')
    with torch.no_grad():
        for t in tensors:
            for x in t.detach().cpu().reshape(-1).tolist():
                f.write('%.9g\\n' % x)
print('exported',sum(t.numel() for t in tensors),'weights')
