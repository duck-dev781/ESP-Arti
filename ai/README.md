# ESP-Arti AI

This directory contains the project-owned Arti model runtime data.

The model is trained from training/corpus.txt by training/train.py.
It is not a downloaded pretrained AI and it does not call a remote inference API.

## Runtime

The ESP32 firmware is the inference engine.

At boot:
1. ESP32 connects to Wi-Fi.
2. ESP32 downloads the current Arti model from this repository.
3. The model is loaded into PSRAM.
4. Inference runs locally on the ESP32.
5. A reset or power loss clears PSRAM, so the model is downloaded again on the next boot.

Changing the model file on GitHub therefore does not require changing the model weights stored permanently on the ESP32.

The first model is intentionally tiny so it can be used as an embedded starting point. It is expected to be limited compared with large language models.
