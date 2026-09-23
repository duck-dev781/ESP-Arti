# ESP-Arti

A low-cost local neural language model runtime for an ESP32 WROVER-E.

## Architecture

- The permanently flashed `ESP-Arti.ino` is the runtime engine and GitHub loader.
- AI files are downloaded at runtime from this repository; changing them does not require reflashing the firmware.
- The ESP32 runs the neural model locally in PSRAM.
- BLE is the chat transport.
- Internet retrieval is optional and provides temporary current-information context.
- No SD card and no external AI inference API are required.

## Runtime AI files

The firmware expects:

- `ai/config.h`
- `ai/tokenizer.h`
- `ai/weights_00.h`
- additional numbered weight files when configured

The .h extension here means **runtime data**. The ESP32 does not compile downloaded source code.

## Important

A trained weight set is required for useful generation. The firmware deliberately does not contain a fake hard-coded response engine.

## Build

Open `ESP-Arti.ino` in Arduino IDE with an ESP32 board package installed. Select an ESP32 WROVER-compatible board and enable PSRAM if your board menu exposes that option.

Set Wi-Fi credentials in the sketch before flashing.

## GitHub Actions

- `.github/workflows/build-firmware.yml` compiles the firmware on every push.
- `.github/workflows/train-model.yml` is manually triggered to train/export runtime AI files and publish them back into the repository when the training environment is available.
