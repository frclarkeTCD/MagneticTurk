# Raspberry Pi setup

1. Flash **Raspberry Pi OS Lite (64-bit)** with Raspberry Pi Imager. Set the hostname
   to `magneticturk`, enable SSH, and add your Wi-Fi.
2. SSH in: `ssh <user>@magneticturk.local`
3. Install system packages (much faster than pip on a Pi Zero 2 W):
   ```bash
   sudo apt update
   sudo apt install -y git python3-venv python3-opencv python3-picamera2 stockfish
   ```
4. Enable I2C: `sudo raspi-config` → Interface Options → I2C.
5. Clone and install:
   ```bash
   git clone https://github.com/frclarkeTCD/MagneticTurk.git
   cd MagneticTurk
   python3 -m venv --system-site-packages .venv   # lets the venv see apt's OpenCV/Picamera2
   source .venv/bin/activate
   pip install -e ".[pi]"
   ```
6. Update later with `git pull`.
