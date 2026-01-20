# 🧬 MolML MCP Server

> **Molecular Machine Learning for AI Assistants** — An MCP server that gives LLMs native access to cheminformatics and molecular ML workflows

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-363%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Summary

**MolML MCP** transforms AI assistants into powerful molecular machine learning workbenches. Through the Model Context Protocol (MCP), LLMs gain the ability to manipulate molecular structures, calculate descriptors, train ML models, and generate comprehensive analysis reports — all through natural conversation.

Simply chat with Claude Desktop (or any MCP client) to perform complex cheminformatics workflows that would normally require writing Python scripts and juggling multiple libraries.

---

## ✨ Key Features

### 🧪 Molecular Operations
- **SMILES Processing**: Standardization, canonicalization, validation, 10-step cleaning pipelines
- **Molecular Descriptors**: Simple (MW, LogP, TPSA) and complex (ECFP, MACCS, RDKit fingerprints)
- **Scaffold Analysis**: Bemis-Murcko, generic scaffolds, cyclic skeletons with diversity metrics
- **Similarity & Clustering**: Tanimoto similarity, DBSCAN, hierarchical, k-means, Butina clustering
- **Substructure Matching**: SMARTS pattern detection with 88+ built-in functional groups

### 🤖 Machine Learning
- **33 ML Algorithms**: Classification & regression (RF, GBM, SVM, linear models, ensembles with uncertainty)
- **Cross-Validation**: 6 strategies (k-fold, stratified, Monte Carlo, scaffold, cluster, leave-P-out)
- **Hyperparameter Tuning**: Grid search, random search with customizable parameter spaces
- **Model Evaluation**: 20+ metrics, confusion matrices, ROC curves, calibration plots

### 📊 Quality Reports
- **Data Quality Analysis**: 19-section comprehensive report (PAINS, Lipinski, duplicates, stereochemistry, etc.)
- **Split Quality Analysis**: 8 data leakage checks (duplicates, similarity, scaffolds, stereoisomers)
- **Scaffold Reports**: Diversity metrics (Gini, Shannon entropy), enrichment analysis, structural outliers

### 📈 Visualization & Statistics
- **Interactive Plots**: Scatter plots with molecular tooltips, histograms, density plots, box plots, heatmaps
- **Statistical Tests**: 15+ tests (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis, chi-square, normality tests)
- **Dimensionality Reduction**: PCA, t-SNE for chemical space visualization
- **Outlier Detection**: Z-score, IQR, isolation forest, local outlier factor

### 🔬 Advanced Features
- **Activity Cliff Detection**: Find structurally similar molecules with large activity differences
- **Data Splitting**: Random, stratified, scaffold-based, cluster-based, temporal splits
- **Duplicate Handling**: Activity conflict detection with aggregation strategies
- **Drug-Likeness Filters**: Lipinski, Veber, PAINS, QED, lead-likeness, rule of three

---

## 💬 Example Prompts

### Getting Started
```
"Create a new project and import dataset from /path/to/molecules.csv"

"Show me the summary statistics"

"What columns do I have and what are their data types?"

"What are the steps I need to perform to train a robust ML model on this data?"
```

### Data Quality & Cleaning
```
"Run a comprehensive data quality report on my dataset"

"Standardize all SMILES strings using the default protocol"

"Find and handle duplicate molecules - some might have conflicting labels"

"Remove PAINS patterns and filter by Lipinski's Rule of Five"
```

### Molecular Properties & Features
```
"Calculate molecular weight, logP, TPSA, and number of H-bond donors/acceptors"

"Generate Morgan fingerprints with radius 2"

"Extract Bemis-Murcko scaffolds and analyze the diversity"

"Add columns for all functional groups present in each molecule"
```

### Visualization and Statistical Analysis
```
"Make scatter plot of molecular weight vs logP colored by activity"

"Create a t-SNE visualization colored by pKi"

"Make a correlation heatmap for MW, LogP, TPSA, HBD, HBA, and pKi"

"Box plot of molecular weight for actives vs inactives. Is there a statistical difference?"

"Test if my pKi values are normally distributed"

"Are there any MW outliers in my data?"
```

### Similarity & Clustering
```
"Find the 10 most similar molecules to this SMILES: CC(=O)Oc1ccccc1C(=O)O"

"Cluster my molecules using DBSCAN and visualize the clusters"

"Which test molecules are most similar to my training set?"

"Are there any activity cliffs in my data?"
```

### Machine Learning Workflows
```
"Split my data using scaffold-based splitting 80/20 train/test"

"Check the quality of my train/test split for any data leakage issues"

"Train a Random Forest classifier with 5-fold cross-validation"

"Tune hyperparameters for a gradient boosting model using grid search"

"Evaluate my model on the test set and show me accuracy, precision, recall, and F1"

"Make predictions on the molecules in /path/to/dataset.csv and show uncertainty estimates"
```

### End-to-End Example
```
"Train a ML model for bioactivity prediction using /path/to/dataset.csv"

"Import this dataset /path/to/dataset.csv, standardize SMILES, remove duplicates, 
filter by Lipinski rule of five, calculate ECFP4 fingerprints, do a scaffold-based 
80/20 split, train a Random Forest with cross-validation, evaluate on the test set."

"Use this dataset /path/to/dataset.csv to create a robust model for virtual screening 
for the molecules in /path/to/virtual_screening.csv"
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.13+** (included with uv)
- **uv** - Fast Python package installer ([install instructions](https://docs.astral.sh/uv/))
  ```bash
  # macOS/Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  
  # Or with pip
  pip install uv
  ```
- **Cairo** (optional) - Only needed for plotting features
  ```bash
  # macOS
  brew install cairo
  
  # Linux
  sudo apt-get install libcairo2-dev  # Ubuntu/Debian
  sudo yum install cairo-devel        # RHEL/CentOS
  ```

### Quick Install (Recommended)

Simply run the installer script - it handles everything:

```bash
# Clone repository
git clone https://github.com/derekvantilborg/molml_mcp.git
cd molml_mcp

# Run installer
./install.sh
```

The installer will:
- ✅ Detect and verify uv installation
- ✅ Install Python dependencies
- ✅ Run server tests
- ✅ Detect Cairo (optional, for plotting - will warn if missing)
- ✅ Configure Claude Desktop automatically
- ✅ Provide clear next steps

**After installation, restart Claude Desktop:**
```bash
# macOS
pkill -x Claude && sleep 1 && open -a Claude

# Linux/Windows
# Restart Claude Desktop from your applications menu
```

### Manual Installation

If you prefer manual setup or use a different MCP client:

```bash
# 1. Clone and install dependencies
git clone https://github.com/derekvantilborg/molml_mcp.git
cd molml_mcp
uv sync

# 2. Run tests to verify installation
uv run pytest -m server -q

# 3. Configure your MCP client
./mcp_client_configs/configure_claude.sh $(which uv) $(pwd) [cairo_path]
```

### Other MCP Clients

For clients other than Claude Desktop, add this to your MCP client configuration:

```json
{
  "mcpServers": {
    "molml-mcp": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "--directory",
        "/path/to/molml_mcp",
        "mcp",
        "run",
        "./src/molml_mcp/server.py"
      ],
      "enabled": true
    }
  }
}
```

---

## 🏗️ Architecture

### Core Components

```
molml_mcp/
├── tools/                      # 150+ molecular ML tools
│   ├── cleaning/              # SMILES cleaning, deduplication, label processing
│   ├── core/                  # Dataset ops, filtering, outliers, statistics, dim reduction
│   ├── core_mol/              # Scaffolds, similarity, activity cliffs, complexity
│   ├── featurization/         # Descriptors (simple, complex, SMILES encoding)
│   ├── ml/                    # Training, evaluation, CV, hyperparameter tuning
│   ├── plotting/              # Interactive visualizations (scatter, histogram, heatmaps)
│   └── reports/               # Quality, scaffold, and split analysis reports
├── infrastructure/
│   ├── resources.py           # Manifest-based resource tracking
│   └── supported_resource_types.py  # CSV, model, JSON, PNG handlers
└── server.py                  # FastMCP server registration
```

### Resource Management

All data operations use a **manifest-based tracking system**:

- **Unique IDs**: Files named `{filename}_{8_HEX_ID}.{ext}` (e.g., `cleaned_data_A3F2B1D4.csv`)
- **Manifest JSON**: Tracks all resources with metadata (created_at, created_by, explanation)
- **Type Registry**: Handlers for CSV (pandas), models (joblib), JSON, PNG (matplotlib)
- **Project Isolation**: Each project has its own manifest and resource directory

### Tool Categories

**150+ tools organized by domain:**

1. **Data Management** (15 tools): Import, export, merge, subset, inspect, filter datasets
2. **Molecular Cleaning** (10 tools): SMILES standardization, salt removal, deduplication
3. **Descriptors** (12 tools): Simple properties, fingerprints, encoding methods
4. **Scaffolds** (8 tools): Bemis-Murcko, generic, diversity analysis
5. **Similarity** (6 tools): Pairwise matrices, k-NN, training set similarity
6. **Clustering** (5 tools): DBSCAN, hierarchical, k-means, Butina
7. **Machine Learning** (40 tools): Training, CV, tuning, evaluation, metrics
8. **Statistics** (15 tools): t-tests, ANOVA, correlation, normality tests
9. **Visualization** (8 tools): Scatter, histogram, density, box, heatmap plots
10. **Quality Reports** (5 tools): Data quality, split quality, scaffold analysis
11. **Activity Cliffs** (4 tools): Cliff detection for classification/regression
12. **Outlier Detection** (6 tools): Z-score, IQR, isolation forest, LOF
13. **Dimensionality Reduction** (2 tools): PCA, t-SNE

---

## 🧪 Testing

```bash
# Run all tests (363 tests)
uv run pytest -v

# Run specific test modules
uv run pytest tests/tools/ml/test_training.py -v
uv run pytest tests/tools/reports/test_quality.py -v

# Run server initialization test
uv run pytest -m server -v

# Run with coverage
uv run pytest --cov=molml_mcp --cov-report=html
```

**Test Coverage:**
- ✅ 363 tests across all modules
- ✅ Infrastructure (resource management, manifests)
- ✅ Cleaning & filtering tools
- ✅ Molecular operations (scaffolds, similarity, activity cliffs)
- ✅ ML training, evaluation, and cross-validation
- ✅ Report generation (quality, scaffold, split analysis)
- ✅ Visualization and statistical analysis

### Deployment

```bash
# Quick deploy (updates MCP client config automatically)
./deploy_mcp_server.sh

# Manual deployment
uv mcp install src/molml_mcp/server.py
# Then restart your MCP client (Claude Desktop, Cline, etc.)
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

This project is open source and free to use for academic and commercial purposes.

---

## 🙏 Acknowledgments

Built with:
- **[FastMCP](https://github.com/modelcontextprotocol/fastmcp)** - Model Context Protocol server framework
- **[RDKit](https://www.rdkit.org/)** - Cheminformatics and machine learning toolkit
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning library
- **[pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[Plotly & Dash](https://plotly.com/)** - Interactive visualizations
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager

Special thanks to:
- **Anthropic** for developing the Model Context Protocol and Claude
- The **RDKit community** for maintaining the foundational cheminformatics toolkit
- All contributors to the open-source scientific Python ecosystem

---

## 📬 Contact

**Derek van Tilborg** - [@derekvantilborg](https://github.com/derekvantilborg)

**Project Link**: [https://github.com/derekvantilborg/molml_mcp](https://github.com/derekvantilborg/molml_mcp)

**Issues & Feature Requests**: [GitHub Issues](https://github.com/derekvantilborg/molml_mcp/issues)

---

<div align="center">

**Give your AI assistant superpowers for molecular machine learning** 🚀

</div>
