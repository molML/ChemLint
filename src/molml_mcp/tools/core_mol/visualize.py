from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# Optional dependency
try:
    import cairosvg  # type: ignore

    HAS_CAIROSVG = True
except Exception:  # ImportError + any cairo dlopen issues
    cairosvg = None  # type: ignore
    HAS_CAIROSVG = False


def smiles_to_acs1996_png_hd(
    smiles: str,
    base_size: Tuple[int, int] = (300, 300),
    legend: Optional[str] = '',
    scale: float = 4.0,
) -> bytes:
    """
    Render a SMILES as an ACS1996-style PNG.

    If CairoSVG is available:
        1) Draw molecule as SVG in ACS1996 mode on a flexicanvas.
        2) Use CairoSVG to rasterize that SVG at `scale`× resolution.

    If CairoSVG is NOT available:
        Fall back to RDKit's MolDraw2DCairo with ACS1996 style.
        (The `scale` argument will still scale the canvas size, but
        the visual scaling is less "true DPI" than the SVG route.)

    Parameters
    ----------
    smiles:
        Input SMILES string.
    base_size:
        Base canvas size in pixels (width, height).
    legend:
        Optional legend; defaults to the SMILES.
    scale:
        How much to scale the output resolution.
        - SVG path: multiplies rendering DPI.
        - Fallback path: multiplies canvas width/height.

    Returns
    -------
    PNG image bytes (ACS1996 style).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)

    if HAS_CAIROSVG:
        # --- High-quality SVG → PNG route ---
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
            scale=scale,
        )
        return png_bytes

    else:
        # --- Fallback: pure RDKit PNG ---
        # Use scale to bump canvas size, even though molecule scale is less ideal
        w = int(base_size[0] * scale)
        h = int(base_size[1] * scale)

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
        return png_bytes



if __name__ == "__main__":
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin

    png_bytes = smiles_to_acs1996_png_hd(
        smi,
        base_size=(300, 300),
        scale=6.0,  # 6x, nice and crisp with CairoSVG
    )

    out_path = "aspirin_acs1996_hd.png"
    with open(out_path, "wb") as f:
        f.write(png_bytes)

    print("Wrote", out_path, "| HAS_CAIROSVG =", HAS_CAIROSVG)