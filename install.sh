#!/usr/bin/env bash
set -euo pipefail

##############################################
# MolML MCP Server - Universal Installer
##############################################
# Automatically installs and configures the
# MolML MCP server for Claude Desktop
##############################################

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[molml-mcp]${NC} $1"
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
# 1. Detect project directory
##############################################

log_info "Installing ChemLint..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

log_success "Found project directory: $PROJECT_DIR"

##############################################
# 2. Find uv
##############################################

UV_BIN=""

# Check if uv is in PATH
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    log_success "Found uv in PATH: $UV_BIN"
# Check common installation locations
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
    log_success "Found uv at: $UV_BIN"
elif [ -f "$HOME/.cargo/bin/uv" ]; then
    UV_BIN="$HOME/.cargo/bin/uv"
    log_success "Found uv at: $UV_BIN"
else
    log_error "uv is not installed!"
    echo ""
    echo "Please install uv first:"
    echo "  • macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  • Or with pip: pip install uv"
    echo ""
    echo "Then run this script again."
    exit 1
fi

##############################################
# 3. Install dependencies
##############################################

log_info "Installing dependencies..."

cd "$PROJECT_DIR"
if "$UV_BIN" sync --quiet; then
    log_success "Dependencies installed"
else
    log_error "Failed to install dependencies"
    exit 1
fi

##############################################
# 5. Run server tests
##############################################

log_info "Running server tests..."
log_warning "First-time installation may take 1-2 minutes (loading RDKit, scikit-learn, etc.)"

if "$UV_BIN" run --directory "$PROJECT_DIR" pytest tests/test_server.py::test_server_imports_and_initializes -v; then
    log_success "Server tests passed"
else
    log_error "Server tests failed!"
    echo ""
    echo "Please fix the errors above before continuing."
    exit 1
fi

##############################################
# 6. Detect Cairo (for plotting support)
##############################################

CAIRO_PATH=""

# Common Cairo locations
if [ -d "/opt/homebrew/opt/cairo/lib" ]; then
    CAIRO_PATH="/opt/homebrew/opt/cairo/lib"
    log_success "Found Cairo at: $CAIRO_PATH"
elif [ -d "/usr/local/opt/cairo/lib" ]; then
    CAIRO_PATH="/usr/local/opt/cairo/lib"
    log_success "Found Cairo at: $CAIRO_PATH"
elif [ -d "/usr/lib/x86_64-linux-gnu" ]; then
    # Linux common location
    CAIRO_PATH="/usr/lib/x86_64-linux-gnu"
    log_success "Found Cairo at: $CAIRO_PATH"
else
    log_warning "Cairo not found - plotting features may not work"
    echo "            Install with: brew install cairo (macOS) or apt-get install libcairo2 (Linux)"
fi

##############################################
# 7. Configure Claude Desktop
##############################################

log_info "Configuring MCP clients..."
echo ""

# Run Claude Desktop configuration script
if [ -f "$PROJECT_DIR/mcp_client_configs/configure_claude.sh" ]; then
    if "$PROJECT_DIR/mcp_client_configs/configure_claude.sh" "$UV_BIN" "$PROJECT_DIR" "$CAIRO_PATH"; then
        CLIENTS_CONFIGURED=1
    else
        CLIENTS_CONFIGURED=0
    fi
else
    log_error "Configuration script not found: mcp_client_configs/configure_claude.sh"
    CLIENTS_CONFIGURED=0
fi

##############################################
# 8. Print success message and next steps
##############################################

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_success "Installation complete! 🎉"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $CLIENTS_CONFIGURED -gt 0 ]; then
    echo "Next steps:"
    echo ""
    
    echo -e "  1. ${GREEN}Restart Claude Desktop${NC} (required for changes to take effect)"
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
    echo "  3. Try it out! Ask Claude to:"
    echo "     • Load a molecular dataset"
    echo "     • Clean SMILES structures"
    echo "     • Calculate molecular descriptors"
    echo "     • Train ML models"
    echo ""
else
    echo "No MCP clients were auto-configured."
    echo ""
    echo "For manual setup, see: README.md"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Documentation: README.md"
echo "Need help? Open an issue at: https://github.com/derekvantilborg/molml_mcp"
echo ""
