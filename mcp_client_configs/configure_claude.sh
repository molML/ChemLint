#!/usr/bin/env bash
set -euo pipefail

##############################################
# Configure ChemLint for Claude Desktop
##############################################
# This script configures Claude Desktop to use
# the ChemLint server
##############################################

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[claude-config]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

##############################################
# Parse arguments
##############################################

if [ $# -lt 2 ]; then
    log_error "Usage: $0 <uv_path> <project_dir> [cairo_path]"
    exit 1
fi

UV_BIN="$1"
PROJECT_DIR="$2"
CAIRO_PATH="${3:-}"

##############################################
# Detect Claude Desktop config location
##############################################

CLAUDE_CONFIG=""

# Detect OS and set Claude config path
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    CLAUDE_CONFIG="$HOME/.config/Claude/claude_desktop_config.json"
else
    log_error "Unsupported OS for automatic Claude configuration"
    echo ""
    echo "For manual setup, add this to your Claude Desktop config:"
    echo ""
    echo '{
  "mcpServers": {
    "molml-mcp": {
      "command": "'$UV_BIN'",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "--directory",
        "'$PROJECT_DIR'",
        "mcp",
        "run",
        "./src/molml_mcp/server.py"
      ],
      "enabled": true
    }
  }
}'
    exit 1
fi

##############################################
# Configure Claude Desktop
##############################################

log_info "Configuring Claude Desktop..."

CLAUDE_DIR="$(dirname "$CLAUDE_CONFIG")"

# Check if Claude config directory exists or can be created
if [ ! -d "$CLAUDE_DIR" ] && ! mkdir -p "$CLAUDE_DIR" 2>/dev/null; then
    log_warning "Claude Desktop not found - skipping configuration"
    echo "            Install Claude Desktop from: https://claude.ai/download"
    exit 0
fi

# Create config if it doesn't exist
if [ ! -f "$CLAUDE_CONFIG" ]; then
    log_info "Creating new Claude Desktop config..."
    cat > "$CLAUDE_CONFIG" <<'EOF'
{
  "mcpServers": {}
}
EOF
fi

# Update config with molml-mcp server using Python
"$UV_BIN" run python3 -c "
import json
import sys

config_path = '$CLAUDE_CONFIG'
uv_bin = '$UV_BIN'
project_dir = '$PROJECT_DIR'
cairo_path = '$CAIRO_PATH'

try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except:
    config = {}

# Ensure mcpServers exists
if 'mcpServers' not in config:
    config['mcpServers'] = {}

# Build server config
server_config = {
    'command': uv_bin,
    'args': [
        'run',
        '--with',
        'mcp[cli]',
        '--directory',
        project_dir,
        'mcp',
        'run',
        './src/chemlint/server.py'
    ],
    'enabled': True
}

# Add Cairo env var if available
if cairo_path:
    server_config['env'] = {
        'DYLD_FALLBACK_LIBRARY_PATH': cairo_path
    }

# Update config
config['mcpServers']['molml-mcp'] = server_config

# Write back
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print('success')
" 2>/dev/null

if [ $? -eq 0 ]; then
    log_success "Claude Desktop configured"
    echo ""
    echo "Config file: $CLAUDE_CONFIG"
    echo ""
    echo "Next steps:"
    echo -e "  1. ${GREEN}Restart Claude Desktop${NC}"
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "     Quick restart (macOS):"
        echo "       pkill -x Claude && sleep 1 && open -a Claude"
    else
        echo "     Close and reopen Claude Desktop from your applications"
    fi
    echo ""
    echo -e "  2. The ${BLUE}molml-mcp${NC} server will appear in your MCP tools"
    echo ""
else
    log_error "Failed to update Claude Desktop config"
    exit 1
fi
