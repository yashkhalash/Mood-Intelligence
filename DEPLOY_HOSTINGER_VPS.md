# Deploy Mood Detector on Hostinger VPS (Ubuntu) — Step by Step

This guide deploys your Flask app (`app.py`) to a **Hostinger VPS** using:
- **Python venv**
- **Gunicorn**
- **systemd** (auto-start on reboot)
- **Nginx** (reverse proxy)
- Optional **HTTPS** via Let’s Encrypt

## Prerequisites
- A Hostinger VPS running **Ubuntu 22.04/24.04**
- A domain (optional, but recommended for HTTPS)
- SSH access to the VPS

## 0) Connect to VPS
From your local machine:

```bash
ssh root@YOUR_VPS_IP
```

## 1) Update OS packages

```bash
sudo apt update && sudo apt upgrade -y
```

## 2) Install system dependencies (critical for MediaPipe/OpenCV)

```bash
sudo apt install -y \
  git curl unzip \
  python3 python3-pip python3-venv \
  build-essential \
  nginx \
  libgl1 \
  libglib2.0-0
```

Notes:
- `libgl1` + `libglib2.0-0` are commonly required for **MediaPipe/OpenCV** on servers.

## 3) Create a non-root user (recommended)

```bash
adduser moodapp
usermod -aG sudo moodapp
su - moodapp
```

## 4) Upload or clone your project

### Option A: Clone from Git (recommended)

```bash
mkdir -p ~/apps
cd ~/apps
git clone YOUR_REPO_URL "Mood-Detector"
cd "Mood-Detector"
```

### Option B: Upload as ZIP (if not using Git)
Upload the project folder to the VPS (SFTP/WinSCP), then:

```bash
cd ~/apps/Mood-Detector
```

## 5) Create a Python virtual environment

```bash
cd ~/apps/Mood-Detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
```

## 6) Install Python requirements

```bash
pip install -r requirements.txt
```

## 7) Ensure model weights exist (required for good predictions)

The app auto-loads model weights from:
- `models/raf_db_emotion_model.pth`

Create the folder if needed:

```bash
mkdir -p models
```

Then **copy your trained** `raf_db_emotion_model.pth` into `models/`.

If you want to train on the VPS (CPU training will be slow), run:

```bash
python3 scripts/clean_rafdb.py
EPOCHS=40 FREEZE_EPOCHS=10 UNFREEZE_LAST_BLOCKS=6 LR_HEAD=0.001 LR_FINETUNE=0.00005 python3 scripts/train_rafdb.py
```

## 8) Smoke test (important)

```bash
python3 -c "import app; print('app import OK')"
```

## 9) Run with Gunicorn (manual test)

```bash
gunicorn --bind 127.0.0.1:8000 --workers 2 --threads 4 app:app
```

In another SSH session:

```bash
curl -I http://127.0.0.1:8000/
```

Stop Gunicorn with `Ctrl+C` after confirming it works.

## 10) Create a systemd service (auto start)

Create the service file:

```bash
sudo nano /etc/systemd/system/mood-detector.service
```

Paste this (edit paths if your folder differs):

```ini
[Unit]
Description=Mood Detector Flask App
After=network.target

[Service]
User=moodapp
Group=www-data
WorkingDirectory=/home/moodapp/apps/Mood-Detector
Environment="PATH=/home/moodapp/apps/Mood-Detector/.venv/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/moodapp/apps/Mood-Detector/.venv/bin/gunicorn \
  --bind 127.0.0.1:8000 \
  --workers 2 \
  --threads 4 \
  --timeout 120 \
  app:app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable + start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mood-detector
sudo systemctl start mood-detector
sudo systemctl status mood-detector --no-pager
```

Logs:

```bash
sudo journalctl -u mood-detector -f
```

## 11) Configure Nginx reverse proxy

Create an Nginx site:

```bash
sudo nano /etc/nginx/sites-available/mood-detector
```

Paste (replace `YOUR_DOMAIN` or use `_` for IP-only):

```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN;

    client_max_body_size 20M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable site and reload:

```bash
sudo ln -sf /etc/nginx/sites-available/mood-detector /etc/nginx/sites-enabled/mood-detector
sudo nginx -t
sudo systemctl reload nginx
```

Now your app should be reachable at:
- `http://YOUR_DOMAIN/` (or `http://YOUR_VPS_IP/`)

## 12) (Optional) Enable HTTPS with Let’s Encrypt

Install Certbot:

```bash
sudo apt install -y certbot python3-certbot-nginx
```

Issue certificate:

```bash
sudo certbot --nginx -d YOUR_DOMAIN
```

Auto-renew check:

```bash
sudo certbot renew --dry-run
```

## 13) Firewall (if needed)

If UFW is enabled:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo ufw status
```

## 14) Verify API endpoint

Open the web UI:
- `http://YOUR_DOMAIN/`

Test analyze endpoint using curl (example; depends on your frontend usage):

```bash
curl -s http://YOUR_DOMAIN/history | head
```

## Troubleshooting

### Gunicorn service won’t start
Check logs:

```bash
sudo journalctl -u mood-detector -n 200 --no-pager
```

### MediaPipe/OpenCV shared library errors
Install missing libs (common fix):

```bash
sudo apt install -y libgl1 libglib2.0-0
sudo systemctl restart mood-detector
```

### Upload size errors
Increase `client_max_body_size` in Nginx (already set to 20M in config above).

---

## What you must deploy (summary checklist)
- **Code**: `app.py`, `mood_detector/`, `templates/`, `static/`, `uploads/`
- **Python deps**: `requirements.txt`
- **Model weights**: `models/raf_db_emotion_model.pth` (recommended)
- **Service**: systemd unit + Nginx config

