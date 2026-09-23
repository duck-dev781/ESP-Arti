"""Train ESP-Arti's own project-owned byte-level Transformer.

The model is trained only from training/corpus.txt. No pretrained model
or remote inference service is used.

The binary layout written here is consumed by output/ESP-Arti-WiFi.ino.
The training and firmware implementations intentionally use the same:
  * learned token embeddings
  * learned positional embeddings
  * standard LayerNorm (weight + bias)
  * causal multi-head self-attention
  * GELU feed-forward blocks
  * residual connections
  * final LayerNorm
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
D = 64
HEADS = 4
LAYERS = 2
FF = 128
STEPS = 8000
BATCH = 16
LR = 2e-3

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "training" / "corpus.txt"
OUT_DIR = ROOT / "ai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

raw = CORPUS_PATH.read_bytes()
if not raw:
    raise RuntimeError("training/corpus.txt is empty")

if len(raw) < CONTEXT + 2:
    raise RuntimeError("training/corpus.txt is too short for the context window")

data = torch.tensor(list(raw), dtype=torch.long)


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
        mask = torch.triu(
            torch.ones(t, t, device=x.device),
            diagonal=1,
        ).bool()
        a = a.masked_fill(mask, -1e9).softmax(-1)

        h = (a @ v).transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.o(h)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class ArtiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.pos = nn.Parameter(torch.zeros(CONTEXT, D))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(D)
        self.out = nn.Linear(D, VOCAB, bias=False)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        x = self.emb(idx) + self.pos[:t]

        for block in self.blocks:
            x = block(x)

        logits = self.out(self.ln(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB),
                targets.reshape(-1),
            )

        return logits, loss


model = ArtiModel()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.01,
)

for step in range(STEPS):
    starts = torch.randint(
        0,
        len(data) - CONTEXT - 1,
        (BATCH,),
    )

    x = torch.stack([
        data[i:i + CONTEXT]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + CONTEXT + 1]
        for i in starts
    ])

    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 500 == 0 or step == STEPS - 1:
        print(f"step={step} loss={loss.item():.4f}")


HEADER_FORMAT = "<8sIIIIIII"
header = struct.pack(
    HEADER_FORMAT,
    b"ESPARTI1",
    2,
    VOCAB,
    CONTEXT,
    D,
    LAYERS,
    HEADS,
    FF,
)

# Binary tensor order MUST stay synchronized with the ESP32 loader.
tensors = [model.emb.weight, model.pos]

for block in model.blocks:
    tensors += [
        block.ln1.weight,
        block.ln1.bias,
        block.q.weight,
        block.k.weight,
        block.v.weight,
        block.o.weight,
        block.ln2.weight,
        block.ln2.bias,
        block.ff1.weight,
        block.ff2.weight,
    ]

tensors += [
    model.ln.weight,
    model.ln.bias,
    model.out.weight,
]

with torch.no_grad():
    payload = b"".join(
        t.detach().cpu().contiguous().numpy().tobytes()
        for t in tensors
    )

model_path = OUT_DIR / "arti-v1.bin"
model_path.write_bytes(header + payload)

expected_floats = (
    VOCAB * D +
    CONTEXT * D +
    LAYERS * (
        D + D +
        D * D + D * D + D * D + D * D +
        D + D +
        D * FF +
        FF * D
    ) +
    D + D +
    D * VOCAB
)

expected_bytes = struct.calcsize(HEADER_FORMAT) + expected_floats * 4

if model_path.stat().st_size != expected_bytes:
    raise RuntimeError(
        f"Bad model export size: expected {expected_bytes}, "
        f"got {model_path.stat().st_size}"
    )

(OUT_DIR / "config.h").write_text(
    "# ESP-Arti project-owned model configuration\n"
    "ESPARTI_FORMAT=2\n"
    "MODEL=arti-v1\n"
    f"VOCAB={VOCAB}\n"
    f"CONTEXT={CONTEXT}\n"
    f"D_MODEL={D}\n"
    f"LAYERS={LAYERS}\n"
    f"HEADS={HEADS}\n"
    f"FFN={FF}\n",
    encoding="utf-8",
)

print(f"corpus bytes: {len(raw)}")
print(f"wrote {model_path} ({model_path.stat().st_size} bytes)")
