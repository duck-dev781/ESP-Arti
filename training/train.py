"""Train ESP-Arti's project-owned v2 byte-level language model.

The ESP32 v2 binary format is intentionally unchanged. This trainer keeps
the exact same architecture and tensor order, but fixes the training/runtime
alignment so the model sees the same positional layout during training that
it sees on the ESP32.

No pretrained model or remote inference service is used.
"""

from pathlib import Path
import math
import random
import re
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)

# ESP-Arti v2 FORMAT -- DO NOT CHANGE WITHOUT CHANGING THE ESP32 LOADER.
VOCAB = 256
CONTEXT = 128
D = 64
HEADS = 4
LAYERS = 2
FF = 128

STEPS = 20000
BATCH = 32
LR = 2e-3
MIN_LR = 3e-4

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "training" / "corpus.txt"
OUT_DIR = ROOT / "ai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

raw = CORPUS_PATH.read_bytes()
if not raw:
    raise RuntimeError("training/corpus.txt is empty")

text = raw.decode("utf-8", errors="replace")

# Train only on complete User/Arti examples.
matches = re.findall(
    r"(?ms)^User:\s*(.*?)\nArti:\s*(.*?)(?=\n\s*\nUser:|\Z)",
    text,
)

examples = []

for user, answer in matches:
    user = " ".join(user.strip().split())
    answer = " ".join(answer.strip().split())

    if not user or not answer:
        continue

    # Match the firmware's input normalization/truncation behavior.
    user_bytes = user.encode("utf-8")
    if len(user_bytes) > 80:
        user_bytes = user_bytes[-80:]

    prompt = b"User: " + user_bytes + b"\nArti:"
    reply = b" " + answer.encode("utf-8") + b"\n"

    # The firmware preserves the beginning of the prompt and then generates
    # inside the remaining context window.
    if len(prompt) >= CONTEXT - 1:
        prompt = prompt[:CONTEXT - 2] + b"Arti:"
        if len(prompt) >= CONTEXT:
            prompt = prompt[:CONTEXT - 1]

    max_reply = CONTEXT - len(prompt)
    if max_reply < 8:
        continue

    reply = reply[:max_reply]
    sequence = prompt + reply

    if len(sequence) >= 4:
        examples.append((prompt, sequence))

if len(examples) < 100:
    raise RuntimeError(
        f"Need at least 100 User/Arti examples, found {len(examples)}"
    )

print(f"conversation examples: {len(examples)}")
print(f"corpus bytes: {len(raw)}")


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

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(c // HEADS)
        mask = torch.triu(
            torch.ones(t, t, device=x.device),
            diagonal=1,
        ).bool()
        scores = scores.masked_fill(mask, -1e9)

        attention = scores.softmax(-1)
        h = (
            attention @ v
        ).transpose(1, 2).contiguous().view(b, t, c)

        x = x + self.o(h)

        # Match the ESP32 runtime's tanh-approximate GELU exactly.
        z = self.ff1(self.ln2(x))
        gelu = 0.5 * z * (
            1.0 + torch.tanh(
                0.79788456 * (z + 0.044715 * z * z * z)
            )
        )
        x = x + self.ff2(gelu)

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

    def forward(self, idx, targets=None, loss_mask=None):
        _, t = idx.shape
        x = self.emb(idx) + self.pos[:t]

        for block in self.blocks:
            x = block(x)

        logits = self.out(self.ln(x))
        loss = None

        if targets is not None:
            per_token = F.cross_entropy(
                logits.reshape(-1, VOCAB),
                targets.reshape(-1),
                reduction="none",
            )

            if loss_mask is None:
                loss = per_token.mean()
            else:
                mask = loss_mask.reshape(-1).float()
                loss = (per_token * mask).sum() / mask.sum().clamp_min(1.0)

        return logits, loss


model = ArtiModel()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.005,
)

for step in range(STEPS):
    progress = step / max(1, STEPS - 1)
    lr = MIN_LR + 0.5 * (LR - MIN_LR) * (
        1.0 + math.cos(math.pi * progress)
    )

    for group in optimizer.param_groups:
        group["lr"] = lr

    batch_x = []
    batch_y = []
    batch_mask = []

    for _ in range(BATCH):
        prompt, sequence = random.choice(examples)
        seq = list(sequence)

        # Input predicts the next byte.
        x = seq[:-1]
        y = seq[1:]

        # Only assistant-response bytes contribute to the loss.
        # The prompt remains visible as conditioning context.
        response_start = len(prompt) - 1
        mask = [0.0] * len(y)

        for i in range(response_start, len(y)):
            mask[i] = 1.0

        # IMPORTANT: right-pad, not left-pad.
        #
        # The ESP32 starts every prompt at positional embedding 0.
        # Left-padding changed the learned positional meaning of every
        # conversation during training and was a major train/runtime mismatch.
        pad = CONTEXT - len(x)

        if pad < 0:
            x = x[:CONTEXT]
            y = y[:CONTEXT]
            mask = mask[:CONTEXT]
            pad = 0

        x = x + [0] * pad
        y = y + [0] * pad
        mask = mask + [0.0] * pad

        batch_x.append(x)
        batch_y.append(y)
        batch_mask.append(mask)

    x = torch.tensor(batch_x, dtype=torch.long)
    y = torch.tensor(batch_y, dtype=torch.long)
    loss_mask = torch.tensor(batch_mask, dtype=torch.float32)

    _, loss = model(x, y, loss_mask)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 500 == 0 or step == STEPS - 1:
        print(f"step={step} loss={loss.item():.4f} lr={lr:.6f}")


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

print(f"wrote {model_path} ({model_path.stat().st_size} bytes)")
