#!/usr/bin/env bash
set -euo pipefail

##############################################
# CONFIG – adjust if your paths ever change
##############################################

PROJECT_DIR="/Users/derekvantilborg/Dropbox/PD/molml_mcp"
UV_BIN="/Users/derekvantilborg/.local/bin/uv"
CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"

##############################################
# 1. (Optional) sync / ensure deps
##############################################

cd "$PROJECT_DIR"

echo "[deploy] Ensuring dependencies are synced with uv..."
"$UV_BIN" sync

##############################################
# 2. Patch Claude Desktop config JSON
##############################################

# Require jq for safe JSON editing
if ! command -v jq >/dev/null 2>&1; then
  echo "[deploy] ERROR: jq is required but not installed."
  echo "        Install it with: brew install jq"
  exit 1
fi

echo "[deploy] Updating Claude config at:"
echo "         $CLAUDE_CONFIG"

# Ensure config file exists; if not, create a minimal skeleton
if [ ! -f "$CLAUDE_CONFIG" ]; then
  echo '[deploy] Config file not found, creating a new one...'
  mkdir -p "$(dirname "$CLAUDE_CONFIG")"
  cat > "$CLAUDE_CONFIG" <<'EOF'
{
  "mcpServers": {}
}
EOF
fi

# Safely update only the molml-mcp entry and remove any legacy "server" entry
tmpfile="$(mktemp)"
jq --arg uv "$UV_BIN" --arg dir "$PROJECT_DIR" '
  .mcpServers = (.mcpServers // {}) |
  del(.mcpServers["server"]) |
  .mcpServers["molml-mcp"] = {
    "command": $uv,
    "args": [
      "run",
      "--with",
      "mcp[cli]",
      "--directory",
      $dir,
      "mcp",
      "run",
      "./src/molml_mcp/server.py"
    ],
    "enabled": true
  }
' "$CLAUDE_CONFIG" > "$tmpfile"

mv "$tmpfile" "$CLAUDE_CONFIG"

echo "[deploy] Claude config updated."

##############################################
# 3. Restart Claude Desktop
##############################################

echo "[deploy] Restarting Claude Desktop..."

# Kill if running (ignore error if not)
pkill -x "Claude" 2>/dev/null || true

# Small pause to let it die
sleep 1

# Relaunch
open -a "Claude"

echo "[deploy] Done. Your molml-mcp server should now be available in Claude."