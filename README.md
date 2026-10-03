# ESP-Arti

ESP-Arti is a Wi-Fi chat interface and offline inference runtime for an ESP32 WROVER-E.

## Current AI

The project no longer trains or ships the old project-owned Arti model.

The current firmware uses the published **TinyTalk 2** Q4 model/runtime from TheREZOR's `cardputer-ai` project. TinyTalk 2 is based on TinyStories-Instruct-8M and was fine-tuned for `User:` / `Bot:` conversations. Its published model card documents the GPT-Neo architecture, Q4 runtime, and chat prompt format.

Source model/runtime:
- GitHub: https://github.com/therezor/cardputer-ai
- Model: https://huggingface.co/TheREZOR/TinyTalk-2

The firmware loads the AI from this repository at boot: `ai/arti.bin` is the TinyTalk 2 Q4 model and `ai/tokenizer.bin` is its tokenizer. The runtime source is fetched from the pinned upstream commit `78c5128672b750977619dc0a6c3b8baed91168ed` during the firmware build.

## ESP32 behavior

- ESP32 WROVER-E runs inference locally.
- `ai/arti.bin` and `ai/tokenizer.bin` are downloaded from this repository into PSRAM at boot.
- After loading, chat generation is local; the model does not call an AI API.
- Wi-Fi is used for the web UI and for the initial model download.
- No SD card is required.
- A role/personality field is available from the web UI.
- Changing the role resets short-term chat history.
- The firmware keeps a small multi-turn history because MCU memory/context are limited.

## Important limitation

TinyTalk 2 is a tiny educational chatbot, not a general ChatGPT-class model. Its published documentation says it is aimed at simple conversation, simple facts, and short context, and it can respond with “I don't know” outside its narrow training range.

## Build

The GitHub Actions workflow `.github/workflows/build-tinytalk.yml` installs Arduino CLI and the ESP32 Arduino core, fetches the pinned TinyTalk runtime source, compiles the WROVER firmware using the scalar path, and publishes `ESP-Arti-TinyTalk2-WROVER.bin` as an Actions artifact.

The permanent sketch is `output/ESP-Arti-WiFi/ESP-Arti-WiFi.ino`.

The build workflow fetches `llm.h` and `llm.cpp` from the pinned upstream runtime at build time, so the large third-party inference engine does not have to be duplicated in this repository.

## Wi-Fi

Edit `WIFI_SSID` and `WIFI_PASSWORD` in the sketch before building.

The model URL points to `https://raw.githubusercontent.com/duck-dev781/ESP-Arti/main/ai/arti.bin`, so replacing that file updates the AI without changing the `.ino`. The tokenizer is loaded from `ai/tokenizer.bin`.

## License

The ESP-Arti project code remains separate from the third-party model/runtime.

TinyTalk 2 is licensed **CC BY-NC-SA 4.0** according to its model card, with attribution and non-commercial/share-alike terms inherited from its training data. Check the upstream license before redistributing the model or using it commercially.