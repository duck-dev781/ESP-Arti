"""Train ESP-Arti's own tiny byte-level Transformer.

This script does not download a pretrained model.
The corpus in training/corpus.txt is the training source for this
project-owned model.

The exporter writes a simple ESP-Arti runtime format.
The ESP32 loader will later be updated to read this exact format.
"""
from pathlib import Path
import math
import random
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)

VOCAB = 256
CONTEXT = 128
D = 32
HEADS = 4
LAYERS = 1
FF = 64
STEPS = 3000
BATCH = 16
LR = 3e-3

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "training" / "corpus.txt"
OUT_DIR = ROOT / "ai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

raw = CORPUS_PATH.read_bytes()
data = torch.tensor(list(raw) * 80, dtype=torch.long)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.ln2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF, bias=False)
        self.ff2 = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        h = self.ln1(x)
        q = self.q(h).view(b, t, HEADS, c // HEADS).transpose(1, 2)
        k = self.k(h).view(b, t, HEADS, c // HEADS).transpose(1, 2)
        v = self.v(h).view(b, t, HEADS, c // HEADS).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(c // HEADS)
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        a = a.masked_fill(mask, -1e9).softmax(-1)
        h = (a @ v).transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.o(h)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x

class ArtiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(D)
        self.out = nn.Linear(D, VOCAB, bias=False)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.out(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        return logits, loss

model = ArtiModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for step in range(STEPS):
    starts = torch.randint(0, len(data) - CONTEXT - 1, (BATCH,))
    x = torch.stack([data[i:i + CONTEXT] for i in starts])
    y = torch.stack([data[i + 1:i + CONTEXT + 1] for i in starts])
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 250 == 0:
        print(f"step={step} loss={loss.item():.4f}")

header = struct.pack(
    "<8sIIIIIII",
    b"ESPARTI1",
    1,
    VOCAB,
    CONTEXT,
    D,
    LAYERS,
    HEADS,
    FF,
)

tensors = [model.emb.weight]
for block in model.blocks:
    tensors += [
        block.ln1.weight,
        block.q.weight,
        block.k.weight,
        block.v.weight,
        block.o.weight,
        block.ln2.weight,
        block.ff1.weight,
        block.ff2.weight,
    ]
tensors += [model.ln.weight, model.out.weight]

with torch.no_grad():
    payload = b"".join(
        t.detach().cpu().contiguous().float().numpy().tobytes()
        for t in tensors
    )

model_path = OUT_DIR / "arti-v1.bin"
model_path.write_bytes(header + payload)

(OUT_DIR / "config.h").write_text(
    "# ESP-Arti project-owned model configuration\n"
    "ESPARTI_FORMAT=1\n"
    "MODEL=arti-v1\n"
    f"VOCAB={VOCAB}\n"
    f"CONTEXT={CONTEXT}\n"
    f"D_MODEL={D}\n"
    f"LAYERS={LAYERS}\n"
    f"HEADS={HEADS}\n"
    f"FFN={FF}\n",
    encoding="utf-8",
)

print(f"wrote {model_path} ({model_path.stat().st_size} bytes)")
