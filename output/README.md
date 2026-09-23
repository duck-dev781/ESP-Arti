# ESP-Arti output

This folder contains the Wi-Fi browser client and generated ESP-Arti runtime outputs.

## Browser client

Open `ESP-Arti.html`, enter the ESP32 IP address shown in Serial Monitor, and chat.

## Firmware

The Wi-Fi firmware is being kept separate from the browser client. It connects to Wi-Fi, downloads the AI runtime files from this repository, loads them into PSRAM, and exposes the HTTP chat endpoint.

The firmware does not use BLE or LittleFS.
