#!/usr/bin/env bash
set -euo pipefail

##############################################
# CONFIG – adjust if your paths ever change
##############################################

PROJECT_DIR="/Users/derekvantilborg/Dropbox/PD/molml_mcp"
UV_BIN="/Users/derekvantilborg/.local/bin/uv"
CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
CAIRO_PATH="/opt/homebrew/opt/cairo/lib"

##############################################
# 1. (Optional) sync / ensure deps
##############################################

cd "$PROJECT_DIR"

echo "[deploy] Ensuring dependencies are synced with uv..."
"$UV_BIN" sync

##############################################
# 2. Test server imports and startup
##############################################

echo "[deploy] Testing server.py for import errors..."

# Run server.py with a 2-second timeout to check for import/startup errors
if ! PYTHONPATH="$PROJECT_DIR/src" "$UV_BIN" run --directory "$PROJECT_DIR" timeout 2s python "$PROJECT_DIR/src/molml_mcp/server.py" 2>&1 | head -20 > /tmp/mcp_test.log; then
  # Check if it was a timeout (expected) or a real error
  if grep -q "Traceback\|Error\|ImportError\|ModuleNotFoundError" /tmp/mcp_test.log; then
    echo "[deploy] ERROR: Server startup failed with errors:"
    cat /tmp/mcp_test.log
    echo ""
    echo "[deploy] Fix the errors above before deploying."
    rm -f /tmp/mcp_test.log
    exit 1
  fi
fi

echo "[deploy] Server imports validated successfully."
rm -f /tmp/mcp_test.log

##############################################
# 3. Patch Claude Desktop config JSON
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
jq --arg uv "$UV_BIN" --arg dir "$PROJECT_DIR" --arg cairo "$CAIRO_PATH" '
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
    "env": {
        "DYLD_FALLBACK_LIBRARY_PATH": $cairo
    },
    "enabled": true
  }
' "$CLAUDE_CONFIG" > "$tmpfile"

mv "$tmpfile" "$CLAUDE_CONFIG"

echo "[deploy] Claude config updated."

##############################################
# 4. Restart Claude Desktop
##############################################

echo "[deploy] Restarting Claude Desktop..."

# Kill if running (ignore error if not)
pkill -x "Claude" 2>/dev/null || true

# Small pause to let it die
sleep 1

# Relaunch
open -a "Claude"

echo "[deploy] Done. Your molml-mcp server should now be available in Claude."