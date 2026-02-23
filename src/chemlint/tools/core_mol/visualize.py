from typing import Optional, Tuple
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from chemlint.infrastructure.resources import _store_resource, _generate_id

# FastMCP (comes from mcp[cli])
from mcp.server.fastmcp import FastMCP, Image
from chemlint.config import DATA_ROOT

# Optional dependency
try:
    import cairosvg  # type: ignore

    HAS_CAIROSVG = True
except Exception:  # ImportError + any cairo dlopen issues
    cairosvg = None  # type: ignore
    HAS_CAIROSVG = False


def smiles_to_acs1996_png(
    smiles: str,
    project_manifest_path: str,
    filename: str,
    explanation: str,
    base_size: Tuple[int, int] = (200, 200),
    legend: Optional[str] = '',
) -> list:
    """
    Render a SMILES as an ACS1996-style PNG image and store as a resource.

    Returns a FastMCP Image object (displayed inline in chat) and the file path
    where the PNG is stored persistently in the resource directory.

    If CairoSVG is available:
        1) Draw molecule as SVG in ACS1996 mode on a flexicanvas.
        2) Use CairoSVG to rasterize that SVG at 4× resolution for crisp output.

    If CairoSVG is NOT available:
        Fall back to RDKit's MolDraw2DCairo with ACS1996 style at base_size.

    Parameters
    ----------
    smiles:
        Input SMILES string.
    base_size:
        Canvas size in pixels (width, height). This is the final output size
        when using RDKit fallback. With CairoSVG, output will be 4× this size.
    legend:
        Optional legend; defaults to empty string.

    Returns
    -------
    list
        A list containing:
        - Image: FastMCP Image object for inline display in chat
        - str: File path where the PNG is stored (e.g., ~/.chemlint/20251202T143022_png_A3F2B1D4.png)
    
    Examples
    --------
    # Render aspirin at default 300x300 (1200x1200 with CairoSVG)
    image, path = smiles_to_acs1996_png("CC(=O)Oc1ccccc1C(=O)O")
    
    # With custom size and legend
    image, path = smiles_to_acs1996_png("CCO", base_size=(400, 400), legend="Ethanol")
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)

    if HAS_CAIROSVG:
        # --- High-quality SVG → PNG route with 4× scaling ---
        drawer = rdMolDraw2D.MolDraw2DSVG(-1, -1)  # flexicanvas

        opts = drawer.drawOptions()
        mean_bond_len = Draw.MeanBondLength(m) or 1.5
        Draw.SetACS1996Mode(opts, mean_bond_len)

        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            m,
            legend=legend,
        )
        drawer.FinishDrawing()

        svg = drawer.GetDrawingText()

        png_bytes: bytes = cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            scale=4.0,
        )
        width = int(base_size[0] * 4)
        height = int(base_size[1] * 4)

    else:
        # --- Fallback: pure RDKit PNG at base_size ---
        w = base_size[0]
        h = base_size[1]

        drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
        opts = drawer.drawOptions()
        mean_bond_len = Draw.MeanBondLength(m) or 1.5
        Draw.SetACS1996Mode(opts, mean_bond_len)

        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            m,
            legend=legend,
        )
        drawer.FinishDrawing()

        png_bytes: bytes = drawer.GetDrawingText()
        width = w
        height = h
    
    # Store the PNG as a resource
    resource_id = _store_resource(png_bytes, project_manifest_path, filename, explanation, "png")
    path = str(DATA_ROOT / f"{resource_id}")
    img = Image(data=png_bytes, format="png")
    
    # return {
    #     "resource_id": resource_id,
    #     "smiles": smiles,
    #     "img": img.to_image_content(),
    # }
    return [img, path]


def smiles_grid_to_acs1996_png(
    smiles_list: list[str],
    project_manifest_path: str,
    filename: str,
    explanation: str,
    legends: Optional[list[str]] = None,
    mols_per_row: int = 4,
    sub_img_size: Tuple[int, int] = (250, 200),
) -> list:
    """
    Render a grid of molecules as an ACS1996-style PNG image and store as a resource.

    Returns a FastMCP Image object (displayed inline in chat) and the file path
    where the PNG is stored persistently in the resource directory.

    If CairoSVG is available:
        1) Draw molecule grid as SVG in ACS1996 mode.
        2) Use CairoSVG to rasterize that SVG at 4× resolution for crisp output.

    If CairoSVG is NOT available:
        Fall back to RDKit's MolDraw2DCairo with ACS1996 style at sub_img_size.

    Parameters
    ----------
    smiles_list:
        List of SMILES strings to visualize.
    legends:
        Optional list of legends for each molecule. If None, uses SMILES as legends.
        Must match length of smiles_list if provided.
    mols_per_row:
        Number of molecules per row in the grid.
    sub_img_size:
        Size of each individual molecule image (width, height) in pixels.
        This is the final output size when using RDKit fallback.
        With CairoSVG, output will be 4× this size.

    Returns
    -------
    list
        A list containing:
        - Image: FastMCP Image object for inline display in chat
        - str: File path where the PNG is stored (e.g., ~/.chemlint/20251202T143022_png_A3F2B1D4.png)

    Raises
    ------
    ValueError
        If any SMILES is invalid or if legends length doesn't match smiles_list.

    Examples
    --------
    # Create a 2x2 grid of molecules
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CN"]
    image, path = smiles_grid_to_acs1996_png(smiles, mols_per_row=2)
    
    # With custom legends and larger sub-images
    legends = ["Ethanol", "Benzene", "Acetic Acid", "Methylamine"]
    image, path = smiles_grid_to_acs1996_png(smiles, legends=legends, mols_per_row=2, sub_img_size=(400, 400))
    """
    if not smiles_list:
        raise ValueError("smiles_list cannot be empty")
    
    if legends is not None and len(legends) != len(smiles_list):
        raise ValueError(f"legends length ({len(legends)}) must match smiles_list length ({len(smiles_list)})")
    
    # Use SMILES as legends if not provided
    if legends is None:
        legends = smiles_list
    
    # Convert SMILES to molecule objects
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi!r}")
        m = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(m)
        mols.append(m)
    
    if HAS_CAIROSVG:
        # --- High-quality SVG → PNG route with 4× scaling ---
        n_rows = (len(mols) + mols_per_row - 1) // mols_per_row
        drawer = rdMolDraw2D.MolDraw2DSVG(
            sub_img_size[0] * mols_per_row,
            sub_img_size[1] * n_rows,
            sub_img_size[0],
            sub_img_size[1],
        )
        
        opts = drawer.drawOptions()
        # Use first valid molecule to get mean bond length
        mean_bond_len = Draw.MeanBondLength(mols[0]) or 1.5
        Draw.SetACS1996Mode(opts, mean_bond_len)
        
        drawer.DrawMolecules(mols, legends=legends)
        drawer.FinishDrawing()
        
        svg = drawer.GetDrawingText()
        
        png_bytes: bytes = cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            scale=4.0,
        )
        width = int(sub_img_size[0] * mols_per_row * 4)
        height = int(sub_img_size[1] * n_rows * 4)
    
    else:
        # --- Fallback: pure RDKit PNG at sub_img_size ---
        n_rows = (len(mols) + mols_per_row - 1) // mols_per_row
        w = sub_img_size[0] * mols_per_row
        h = sub_img_size[1] * n_rows
        
        drawer = rdMolDraw2D.MolDraw2DCairo(
            w,
            h,
            sub_img_size[0],
            sub_img_size[1],
        )
        
        opts = drawer.drawOptions()
        mean_bond_len = Draw.MeanBondLength(mols[0]) or 1.5
        Draw.SetACS1996Mode(opts, mean_bond_len)
        
        drawer.DrawMolecules(mols, legends=legends)
        drawer.FinishDrawing()
        
        png_bytes: bytes = drawer.GetDrawingText()
        width = w
        height = h
    
    # Store the PNG as a resource
    resource_id = _store_resource(png_bytes, project_manifest_path, filename, explanation, "png")
    path = str(DATA_ROOT / f"{resource_id}")
    img = Image(data=png_bytes, format="png")
    
    # return {
    #     "resource_id": resource_id,
    #     "n_molecules": len(smiles_list),
    #     "mols_per_row": mols_per_row,
    #     "img": img.to_image_content(),
    # }
    return [img, path]



# if __name__ == "__main__":
#     from chemlint.infrastructure.resources import _load_resource
#     from chemlint.config import DATA_ROOT
    
#     print(f"DATA_ROOT: {DATA_ROOT}")
    
#     # Test single molecule
#     smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin

#     result = smiles_to_acs1996_png(
#         smi,
#         base_size=(300, 300),
#     )

#     print(f"Single molecule: {result['resource_id']}")
#     print(f"  Size: {result['width']}x{result['height']}")
#     print(f"  SMILES: {result['smiles']}")
#     print(f"  File: {DATA_ROOT / result['resource_id']}")
    
#     # Test grid
#     smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "CN", "CC(=O)Oc1ccccc1C(=O)O", "CCN"]
#     legends = ["Ethanol", "Benzene", "Acetic Acid", "Methylamine", "Aspirin", "Ethylamine"]
    
#     grid_result = smiles_grid_to_acs1996_png(
#         smiles_list,
#         legends=legends,
#         mols_per_row=3,
#         sub_img_size=(300, 300),
#     )
    
#     print(f"\nGrid: {grid_result['resource_id']}")
#     print(f"  Molecules: {grid_result['n_molecules']}")
#     print(f"  Per row: {grid_result['mols_per_row']}")
#     print(f"  Size: {grid_result['width']}x{grid_result['height']}")
#     print(f"  File: {DATA_ROOT / grid_result['resource_id']}")
    
#     print(f"\nHAS_CAIROSVG = {HAS_CAIROSVG}")