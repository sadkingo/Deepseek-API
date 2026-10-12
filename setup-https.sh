#!/usr/bin/env bash
# Get a Let's Encrypt certificate for the server's domain and switch the
# server to HTTPS. Runs as you, not root: nothing in /etc is touched and
# nginx keeps running.
#
#   ./setup-https.sh            check the router, get the certificate, wire it up
#   ./setup-https.sh --check    only check that the router forwards port 80 here
#
# The API stays on its own port ($PORT, 8010); the certificate is for the name,
# not a port. But Let's Encrypt proves you own the name by fetching a file
# from http://<domain>/ - always port 80 on the outside - and a free No-IP
# name cannot carry the DNS (TXT) record of the other method. So the path
# /.well-known/acme-challenge/ on port 80 must lead to THIS machine, port
# $ACME_PORT (8081 by default): either the machine that owns port 80 proxies
# just that path here (port 80 otherwise unchanged), or the router forwards
# port 80 here. The check below fails fast and prints both options, so no
# Let's Encrypt attempt is wasted (failed ones are rate-limited).
#
# What it does once the check passes:
#   - certbot (standalone, on $ACME_PORT) issues the certificate; its state
#     lives in ~/.local/share/deepseek-certbot
#   - fullchain.pem + privkey.pem are copied to ~/deepseek-certs (mode 600),
#     again by a deploy hook on every renewal
#   - SSL_CERTFILE / SSL_KEYFILE are written to .env
#   - a crontab line renews twice a day (certbot only renews within 30 days
#     of expiry); the running server notices the new files and reloads them
#     without a restart
#
# Settings (environment): DOMAIN, ACME_PORT, CERT_DEST, CERTBOT_DIR, EMAIL.
set -euo pipefail

PROJECT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
DOMAIN="${DOMAIN:-mody.ddns.net}"
ACME_PORT="${ACME_PORT:-8081}"
DEST="${CERT_DEST:-$HOME/deepseek-certs}"
STATE="${CERTBOT_DIR:-$HOME/.local/share/deepseek-certbot}"
EMAIL="${EMAIL:-}"
CRON_MARK="# deepseek-api: renew the HTTPS certificate"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

[ "$EUID" -ne 0 ] || { echo "Run this as yourself, not with sudo." >&2; exit 1; }
if [ "$ACME_PORT" -lt 1024 ]; then
    echo "ACME_PORT=$ACME_PORT needs root; use a port >= 1024 and forward external" >&2
    echo "port 80 to it in the router." >&2
    exit 1
fi

LAN_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<NF;i++) if ($i=="src") {print $(i+1); exit}}')"
LAN_IP="${LAN_IP:-this machine}"
PORT="$(grep -E '^PORT=' "$PROJECT/.env" 2>/dev/null | tail -1 | cut -d= -f2)"
PORT="${PORT:-8010}"

# --- 1. does http://DOMAIN/ (port 80 outside) reach this machine? ----------
check_port_80() {
    local dir token got pid server
    dir="$(mktemp -d)"
    token="probe-$(od -An -N12 -tx1 /dev/urandom | tr -d ' \n')"
    mkdir -p "$dir/.well-known/acme-challenge"
    printf '%s' "$token" > "$dir/.well-known/acme-challenge/$token"
    python3 -m http.server "$ACME_PORT" --bind 0.0.0.0 --directory "$dir" >/dev/null 2>&1 &
    pid=$!
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -rf "$dir"
        echo "Port $ACME_PORT is busy on this machine; pick another: ACME_PORT=8082 $0" >&2
        return 1
    fi
    got="$(curl -s -m 10 "http://$DOMAIN/.well-known/acme-challenge/$token" || true)"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -rf "$dir"
    if [ "$got" = "$token" ]; then
        echo "OK: port 80 on $DOMAIN reaches this machine (port $ACME_PORT)."
        return 0
    fi
    server="$(curl -sI -m 5 "http://$DOMAIN/" 2>/dev/null | tr -d '\r' | awk -F': ' 'tolower($1)=="server"{print $2}')"
    cat >&2 <<EOF
Let's Encrypt cannot reach this machine through http://$DOMAIN/ (port 80)${server:+;
it is answered by "$server" on another machine}.

The API itself stays on https://$DOMAIN:$PORT/; port 80 is only how Let's
Encrypt checks that you own the name. Either of these fixes it:

  A) Keep port 80 where it is, and let the machine that has it pass on just
     the verification path. In that machine's nginx, inside the server block
     that answers on port 80, add:

         location ^~ /.well-known/acme-challenge/ {
             proxy_pass http://$LAN_IP:$ACME_PORT;
         }

     then reload nginx there. Nothing else on port 80 changes.

  B) Forward external TCP port 80 in the router to $LAN_IP port $ACME_PORT.

Then run this again:  $0
EOF
    return 1
}

check_port_80 || exit 1
[ "$CHECK_ONLY" -eq 1 ] && exit 0

# --- 2. certificate -------------------------------------------------------
command -v certbot >/dev/null || { echo "certbot is not installed (sudo apt install certbot)." >&2; exit 1; }
mkdir -p "$STATE"/{config,work,logs}

# Copies the certificate where the server reads it; certbot runs this after
# the first issue and after every renewal.
HOOK="$STATE/deploy-hook.sh"
cat > "$HOOK" <<EOF
#!/bin/sh
set -e
install -d -m 700 "$DEST"
install -m 600 "\$RENEWED_LINEAGE/fullchain.pem" "\$RENEWED_LINEAGE/privkey.pem" "$DEST/"
EOF
chmod 700 "$HOOK"

CERTBOT=(certbot --config-dir "$STATE/config" --work-dir "$STATE/work" --logs-dir "$STATE/logs")
if [ -n "$EMAIL" ]; then contact=(--email "$EMAIL"); else contact=(--register-unsafely-without-email); fi

echo "==> requesting a certificate for $DOMAIN"
"${CERTBOT[@]}" certonly --standalone --http-01-port "$ACME_PORT" -d "$DOMAIN" \
    --agree-tos --non-interactive "${contact[@]}" --keep-until-expiring \
    --deploy-hook "$HOOK"
# --keep-until-expiring skips the deploy hook when the certificate already
# existed, so copy explicitly.
RENEWED_LINEAGE="$STATE/config/live/$DOMAIN" "$HOOK"

# --- 3. point the server at it -------------------------------------------
ENV_FILE="$PROJECT/.env"
touch "$ENV_FILE"
set_env() {
    if grep -qE "^$1=" "$ENV_FILE"; then
        sed -i "s|^$1=.*|$1=$2|" "$ENV_FILE"
    else
        printf '%s=%s\n' "$1" "$2" >> "$ENV_FILE"
    fi
}
set_env SSL_CERTFILE "$DEST/fullchain.pem"
set_env SSL_KEYFILE "$DEST/privkey.pem"
echo "==> .env: SSL_CERTFILE and SSL_KEYFILE set"

# --- 4. renewal ----------------------------------------------------------
RENEW="$(printf '%q ' "${CERTBOT[@]}") renew --quiet"
( crontab -l 2>/dev/null | grep -vF "$CRON_MARK" || true
  echo "17 3,15 * * * $RENEW $CRON_MARK" ) | crontab -
echo "==> crontab: renewal twice a day"

cat <<EOF

Done. Restart the server (Ctrl-C in its terminal, then \`start\`); it will serve
    https://$DOMAIN:$PORT/        to the outside
    http://127.0.0.1:$PORT/       to this machine (extension, register, tunnel)
Renewed certificates are loaded without a restart.

Renewals (about every 60 days) use the same path on port 80 as this run, so
leave the /.well-known/acme-challenge/ route to this machine in place.
EOF
