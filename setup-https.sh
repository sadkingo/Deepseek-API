#!/usr/bin/env bash
# Obtain a Let's Encrypt certificate for mody.ddns.net and place a readable
# copy in ~/deepseek-certs for the DeepSeek API server.
#
#   sudo ./setup-https.sh
#
# Requires port 80 forwarded to this machine (in addition to 8000).
set -euo pipefail

DOMAIN=mody.ddns.net
TARGET_USER=${SUDO_USER:-$USER}
DEST=$(getent passwd "$TARGET_USER" | cut -d: -f6)/deepseek-certs

[ "$EUID" -eq 0 ] || { echo "Run with sudo: sudo ./setup-https.sh" >&2; exit 1; }

install_certbot() {
  echo "==> installing certbot"
  # Unrelated third-party repos on this box fail to refresh; that must not stop
  # the install, so the update result is deliberately ignored.
  apt-get update || echo "    (apt update reported errors - continuing anyway)"
  if apt-get install -y certbot; then return 0; fi

  echo "==> apt failed, falling back to a self-contained pip install"
  apt-get install -y python3-venv || true
  python3 -m venv /opt/certbot 2>/dev/null || return 1
  /opt/certbot/bin/pip install --upgrade pip certbot || return 1
  ln -sf /opt/certbot/bin/certbot /usr/local/bin/certbot
}

command -v certbot >/dev/null || install_certbot
command -v certbot >/dev/null || { echo "certbot could not be installed" >&2; exit 1; }

# Standalone mode: certbot binds port 80 itself for the ACME challenge. This
# leaves the existing nginx site configs untouched (none of them serve $DOMAIN),
# at the cost of a few seconds of nginx downtime. The hooks do the same on
# automatic renewal.
echo "==> requesting certificate for $DOMAIN (nginx stops for a few seconds)"
systemctl stop nginx
trap 'systemctl start nginx || true' EXIT

certbot certonly --standalone -d "$DOMAIN" \
  --agree-tos --non-interactive --register-unsafely-without-email \
  --pre-hook "systemctl stop nginx" --post-hook "systemctl start nginx"

systemctl start nginx
trap - EXIT

echo "==> copying cert into $DEST (readable by $TARGET_USER)"
install -d -o "$TARGET_USER" -m 700 "$DEST"
install -o "$TARGET_USER" -m 600 \
  "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" \
  "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$DEST/"

echo
echo "Done. Add these two lines to .env, then restart the server:"
echo "  SSL_CERTFILE=$DEST/fullchain.pem"
echo "  SSL_KEYFILE=$DEST/privkey.pem"
