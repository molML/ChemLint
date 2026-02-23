# PAINS detection using 480 PAINS patterns from Baell & Holloway 2010 

from typing import Dict
from rdkit.Chem import MolFromSmiles
import csv
import os
import re
from chemlint.tools.core_mol.substructure_matching import _mol_has_pattern


def get_pains_smarts() -> Dict[str, str]:
    """Load all PAINS (Pan-Assay INterference compoundS) filter patterns.
    
    PAINS are substructures known to give false positive results in biological assays
    through non-specific binding or interfering with assay readouts. Returns the complete
    PAINS database (480 patterns) from the Baell & Holloway 2010 publication.
    
    Returns:
        dict: Mapping of pattern name to SMARTS string. Keys are pattern names like
              "anil_di_alk_F(14)", "ene_six_het_A(483)", etc. Values are SMARTS patterns.
        
    Reference:
        Baell JB, Holloway GA. New Substructure Filters for Removal of Pan Assay 
        Interference Compounds (PAINS) from Screening Libraries and for Their Exclusion 
        in Bioassays. J Med Chem 53 (2010) 2719-2740. doi:10.1021/jm901137j
    """
    import rdkit
    rdkit_path = os.path.dirname(rdkit.__file__)
    pains_csv = os.path.join(rdkit_path, "Data", "Pains", "wehi_pains.csv")
    
    patterns = {}
    if not os.path.exists(pains_csv):
        return patterns
    
    with open(pains_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                smarts = row[0]
                # Extract name from regId=name format
                name_match = re.search(r'<regId=([^>]+)>', row[1])
                if name_match:
                    name = name_match.group(1)
                    patterns[name] = smarts
    
    return patterns



# "why PAINS" by *prefix* of the name
PREFIX_EXPLANATIONS: dict[str, str] = {
    # very generic fallback
    "": "frequent hitter / PAINS motif",

    # core heterocycles / pharmacofores
    "catechol": "catechol; redox-active metal chelator",
    "quinone": "quinone; redox cycling electrophile",
    "hydroquin": "hydroquinone; easily oxidized redox pair",
    "indole": "indole; flat π-stacking frequent hitter",
    "indol_3yl": "3-substituted indole; π-stacking binder",
    "imidazole": "imidazole; metal binder / hinge mimic",
    "pyrazole": "pyrazole; hinge-binding heterocycle",
    "pyrrole": "pyrrole; electron-rich heteroaromatic",
    "thiophene": "thiophene; flat sulfur heteroaromatic",
    "furan": "furan; oxidizable heteroaromatic",
    "thiazole": "thiazole; metal / hinge binder",
    "tetrazole": "tetrazole; acidic, multidentate chelator",
    "melamine": "melamine; multi-H-bond chelator",
    "acridine": "acridine; DNA intercalator scaffold",

    # anilines / anilides
    "anil_di_alk": "di-alkyl aniline; redox-active, sticky",
    "anil_alk": "alkyl aniline; redox-active aryl amine",
    "anil_NH": "arylamide / aniline; strong H-bonding",
    "anil_OC": "anilide / acetanilide; flat H-bond donor/acceptor",
    "anil_OH": "p-hydroxy aniline; redox-active phenol",
    "anil_no_alk": "unsubstituted aniline; reactive aryl amine",

    # hydrazones / azo / “HZONE”
    "hzone": "hydrazone / azo-like linker; redox-active",
    "azo_": "azo (N=N) linker; chromophore, redox-active",
    "anthranil": "anthranilic / isatoic motif; reactive acyl-amide",

    # conjugated enones, enals, Michael acceptors
    "ene_one_one": "conjugated 1,3-dicarbonyl; Michael acceptor",
    "ene_one_ene": "conjugated enone diene; Michael acceptor",
    "ene_one_amide": "α,β-unsaturated amide; Michael acceptor",
    "ene_one_ester": "α,β-unsaturated keto-ester; Michael acceptor",
    "ene_one_hal": "α,β-unsaturated carbonyl-halide; electrophile",
    "ene_one_yne": "ynone; very strong Michael acceptor",
    "ene_one_A": "α,β-unsaturated ketone; covalent Michael acceptor",
    "ene_one_B": "conjugated enone; Michael acceptor",
    "ene_one_C": "α,β-unsaturated imide; Michael acceptor",
    "ene_one_D": "extended enone; Michael acceptor",

    # generic “ene_…” conjugated systems
    "ene_five_het": "enone fused to 5-membered heterocycle",
    "ene_six_het": "enone fused to 6-membered heterocycle",
    "ene_five_one": "benzo-fused enone; Michael acceptor",
    "ene_quin_methide": "quinone-methide-like; strong electrophile",
    "ene_cyano": "conjugated nitrile; strong electron-withdrawing enone",
    "ene_rhod": "rhodamine-like conjugated dye; sticky cation",
    "ene_misc": "extended conjugated enone / dye",
    "ene_misc_A": "rigid conjugated scaffold; aggregation risk",
    "ene_misc_B": "extended conjugated enone; PAINS dye",
    "ene_misc_C": "highly conjugated enone; autofluorescent",
    "ene_misc_D": "extended enone; aggregation / fluorescence",
    "ene_misc_E": "aryl enone; Michael acceptor",

    # rhodamine / xanthene dyes
    "rhod_sat": "rhodamine-type cationic dye; sticky binder",
    "ene_rhod_A": "rhodamine/xanthene-like dye core",
    "ene_rhod_B": "halogenated rhodamine-type dye",
    "ene_rhod_C": "rhodamine-like dye with thioamide",
    "ene_rhod_D": "rhodamine-like amidine dye",
    "ene_rhod_E": "sulfur-containing rhodamine analogue",
    "ene_rhod_F": "rhodamine-like conjugated imide",
    "ene_rhod_G": "extended rhodamine-like scaffold",
    "ene_rhod_H": "rhodamine-like thioether dye",
    "ene_rhod_I": "brominated rhodamine-like dye",
    "ene_rhod_J": "complex rhodamine-like conjugated dye",

    # cyano-containing electrophiles
    "cyano_ene_amine": "conjugated nitrile enamine; Michael acceptor",
    "cyano_imine": "nitrile-imine conjugate; 1,3-dipole / electrophile",
    "cyano_pyridone": "cyano-pyridone; chelating H-bond donor/acceptor",
    "cyano_keto": "cyano-ketone; electrophile / Michael acceptor",
    "cyano_cyano": "poly-cyano scaffold; strong e-withdrawer",
    "cyano_misc": "poly-cyano aromatic; chelating / reactive",
    "cyano_amino_het": "amino-nitrile heterocycle; strong binder",

    # 1,3-dicarbonyls and chelators
    "keto_keto_beta": "β-diketone; enolizable metal chelator",
    "keto_keto_gamma": "1,3-dicarbonyl; metal chelator",
    "keto_keto_beta_zone": "bis-diketone; strong chelator",
    "keto_keto_beta_E": "highly conjugated diketone; chelator",
    "keto_keto_beta_F": "diketone with alkyne; chelator",
    "keto_naphthol": "naphthol-diketone; redox / chelator",
    "keto_phenone": "aryl-ketone-imide; flat H-bonding",
    "keto_phenone_zone": "bis-aryl-ketone; aggregation risk",
    "keto_thiophene": "thiophene-aryl ketone; conjugated binder",

    # sulfur-rich (thio-) motifs
    "thio_ketone": "thioenone; very strong Michael acceptor",
    "thio_amide": "thioamide; nucleophilic, redox-active",
    "thio_ester": "thioester; acyl-transfer electrophile",
    "thio_ester_A": "aryl thioester; reactive acyl donor",
    "thio_ester_B": "thioester with aromatic; reactive",
    "thio_ester_C": "bis-thioester motif; covalent risk",
    "thio_aldehyd": "thioaldehyde; highly reactive electrophile",
    "thio_carbonate": "thiocarbonate; redox-active leaving group",
    "thio_carbam": "thiocarbamate; covalent modifier",
    "thio_carbam_ene": "conjugated thiocarbamoyl; Michael acceptor",
    "thio_cyano": "thio-nitrile; soft electrophile / chelator",
    "thio_imide": "thioimide; conjugated electrophile",
    "thio_imide_A": "aryl thioimide; Michael acceptor",
    "thio_imine_ium": "thiourea-like iminium; reactive",
    "thio_keto_het": "heteroaryl thioenone; Michael acceptor",
    "thio_amide_A": "thioamide on aryl; sticky binder",
    "thio_amide_B": "extended thioamide; redox / covalent",
    "thio_amide_C": "bis-aryl thioamide; PAINS binder",
    "thio_amide_D": "anilide-thioamide; redox-active",
    "thio_amide_E": "thioamide-ether; sticky heteroaromatic",
    "thio_amide_F": "bis-amide thioamide; crosslink risk",

    "thio_urea": "thiourea; strong H-bonding, chelating",
    "thio_urea_A": "bulky diaryl thiourea; aggregator",
    "thio_urea_B": "diaryl thiourea; strong H-bond donor",
    "thio_urea_C": "thiourea-urea hybrid; chelating",
    "thio_urea_D": "linker thiourea; promiscuous binder",
    "thio_urea_E": "thiourea with N-heterocycle; sticky",
    "thio_urea_F": "cyclic thiourea; metal binder",
    "thio_urea_G": "thiourea with guanidine; multidentate",
    "thio_urea_H": "conjugated thiourea; Michael acceptor",
    "thio_urea_I": "thiourea-imine system; covalent risk",
    "thio_urea_J": "phenoxy thiourea; redox / binder",
    "thio_urea_K": "poly-nitrogen thiourea; chelator",
    "thio_urea_L": "thiourea with enone; covalent PAINS",
    "thio_urea_M": "bulky thiourea; aggregation / chelation",
    "thio_urea_N": "thiourea fused to phenoxy; sticky",
    "thio_urea_O": "imidazole-thiourea; strong binder",
    "thio_urea_P": "benzimidazole thiourea; chelator",
    "thio_urea_Q": "polyaryl thiourea; aggregator",
    "thio_urea_R": "conjugated thiourea-enone; PAINS",
    "thio_urea_I(3)": "thiourea-imine; electrophilic chelator",

    # “het_thio_…”: sulfur-rich heterocycles
    "het_thio": "sulfur-rich heterocycle; soft chelator / redox",
    "thio_thiomorph": "thiomorpholine-thioamide; sticky, chelating",
    "thio_thiomorph_Z": "thiomorpholine-thioamide; poly-heteroatom binder",

    # sulfonamides
    "sulfonamide_A": "aryl sulfonamide; strong H-bonding, sticky",
    "sulfonamide_B": "aryl sulfonamide; frequent hitter",
    "sulfonamide_C": "bis-sulfonamide; multidentate binder",
    "sulfonamide_D": "disulfonamide; strong chelator / binder",
    "sulfonamide_E": "bis-aryl sulfonamide; sticky, PAINS",
    "sulfonamide_F": "sulfonamide on thiazole; chelating",
    "sulfonamide_G": "highly substituted sulfonamide; aggregator",
    "sulfonamide_H": "bis-sulfonamide heterocycle; sticky",
    "sulfonamide_I": "sulfonamide on N-heterocycles; strong binder",
    "sulfonamide_J": "polyaryl sulfonamide; aggregation risk",

    # Mannich, aminals, ureas
    "mannich_A": "Mannich phenol; unstable, covalent risk",
    "mannich_B": "Mannich anilide; redox / covalent",
    "mannich_catechol": "Mannich catechol; redox / chelator",
    "misc_urea": "bulky diaryl urea; aggregator",
    "misc_anilide": "bulky anilide; flat sticky binder",

    # dyes and poly-conjugated aromatics
    "dyes": "highly conjugated dye-like chromophore",
    "coumarin": "coumarin dye scaffold; autofluorescent",
    "stilbene": "stilbene; planar π-system, aggregator",
    "misc_stilbene": "heavily substituted stilbene; aggregator",
    "thio_dibenzo": "thio-dibenzoquinone; redox-active",
    "diazox": "diazoxide-like N-oxide; redox / chelation",
    "diazox_sulfon": "diazoxide-sulfonamide hybrid; sticky",
    "het_6_tetrazine": "tetrazine; very strong π-acceptor, quencher",
    "het_pyridiniums": "pyridinium salt; cationic dye / binder",

    # hydroquinone/catechol-like phenols
    "hzone_phenone": "hydrazone-phenone; redox and chelation",
    "hzone_phenol": "hydrazone-phenol; redox cycling PAINS",
    "hzone_furan": "hydrazone-furan; conjugated redox motif",
    "hzone_acid": "hydrazide acid; conjugated hydrazone",
    "hzone_acyl": "acyl hydrazone; classic PAINS linker",

    # thiophene-amino families
    "thiophene_amino": "amino-thiophene; hinge-binder frequent hitter",
    "thiophene_hydroxy": "hydroxy-thiophene; redox / binder",
    "thiophene_C": "conjugated thiophene amide; PAINS",
    "thiophene_D": "thiophene amide/ester; reactive",
    "thiophene_E": "thiophene amide-nitrile; binder",
    "thiophene_F": "sulfonyl thiophene; sticky",
    "thiophene_G": "di-carbonyl thiophene; metal binder",

    # furan / naphthalene / coumarin etc
    "furan_A": "arylfuran with electron-rich tail; PAINS",
    "furan_acid": "furan-carboxylic acid; conjugated binder",
    "naphth_amino": "amino-naphthalene; flat aromatic binder",
    "naphth_ene_one": "naphthoquinone-like enone; redox",
    "naphthimidazole": "naphthalene-imidazole; DNA-like binder",

    # steroids / bulky hydrophobes
    "steroid": "steroid-like fused rings; detergent-like binder",
    "tert_butyl": "heavily tert-butylated phenyl; greasy aggregator",
    "misc_trityl": "trityl-like aryls; very hydrophobic",

    # generic “het_5”, “het_6”, “het_65/66/666…”
    "het_5_": "five-membered N/S heteroaromatic PAINS",
    "het_6_": "six-membered N-rich heteroaromatic PAINS",
    "het_65": "fused 5-6 heteroaromatics; chelating",
    "het_66": "fused 6-6 heteroaromatics; flat chelator",
    "het_666": "extended N-rich fused heterocycles; chelator",
    "het_6666": "large fused heterocycles; π-stacking binder",
    "het_565": "5-6-5 fused heterocycles; sticky",
    "het_565_indole": "indole-like 5-6-5; intercalator",
    "het_76": "7-6 sulfur heterocycles; redox PAINS",

    # “misc_*” buckets - keep super short & generic
    "misc_anisole": "heavily substituted anisole; hydrophobic binder",
    "misc_aminoacid": "bulky amino-acid-like scaffold; sticky",
    "misc_anilide": "bulky diaryl anilide; aggregator",
    "misc_cyclopropane": "sulfonyl-cyclopropane; strained electrophile",
    "misc_furan": "furan-rich, heteroatom-rich scaffold; PAINS",
    "misc_imidazole": "poly-imidazole, poly-aryl scaffold; binder",
    "misc_naphthimidazole": "naphthalene-imidazole; intercalator-like",
    "misc_pyridine_OC": "poly-alkoxy pyridine; sticky heterocycle",
    "misc_pyrrole_thiaz": "pyrrole-thiazole hybrid; hinge-binder",
    "misc_pyrrole_benz": "pyrrole-benzamide; flat PAINS binder",
    "misc_anilide_A": "di-anilide; flat aggregating binder",
    "misc_anilide_B": "halogenated anilide; redox/aggregation",
    "misc_pyridine_OC": "alkoxy-pyridine; sticky hydrogen-bonder",
    "misc_aminal_acid": "aminal-amide motif; unstable / reactive",
}

def _short_explanation(name: str) -> str:
    """Return a very short PAINS explanation based on the name prefix."""
    # longest prefixes first so specific beats generic
    for prefix in sorted(PREFIX_EXPLANATIONS, key=len, reverse=True):
        if prefix and name.startswith(prefix):
            return PREFIX_EXPLANATIONS[prefix]
    return PREFIX_EXPLANATIONS[""]

# Pre-compile PAINS patterns for performance (avoid recompiling 480 patterns per molecule)
_PAINS_PATTERNS_CACHE = None

def _get_compiled_pains_patterns():
    """Get pre-compiled PAINS patterns (cached for performance)."""
    global _PAINS_PATTERNS_CACHE
    if _PAINS_PATTERNS_CACHE is None:
        from rdkit.Chem import MolFromSmarts
        _PAINS_PATTERNS_CACHE = {}
        for name, smarts in get_pains_smarts().items():
            pattern = MolFromSmarts(smarts, mergeHs=True)
            if pattern is not None:  # Skip invalid patterns
                _PAINS_PATTERNS_CACHE[name] = {
                    "pattern": pattern,
                    "explanation": _short_explanation(name.split("(")[0])
                }
    return _PAINS_PATTERNS_CACHE


def _check_smiles_for_pains(smiles: str) -> str:
    """Screen a molecule for PAINS (Pan-Assay INterference compoundS) substructures.
    
    Checks input SMILES against 480 PAINS patterns from Baell & Holloway 2010. PAINS are 
    substructures that cause false positives in high-throughput screening through non-specific 
    binding, aggregation, redox activity, or assay interference.
    
    Args:
        smiles: SMILES string of molecule to screen (e.g., 'O=C1C=CC(=O)C=C1')
        
    Returns:
        str: Screening result in one of three formats:
             - 'Passed' if no PAINS patterns detected (molecule is clean)
             - 'PAINS: <reasons>' if PAINS detected, with comma-separated explanations
               (e.g., 'PAINS: quinone; redox cycling electrophile')
             - 'Failed: <error>' if input is invalid or cannot be parsed
             
    Example:
        _check_smiles_for_pains('CCO')  # Returns 'Passed' (ethanol is clean)
        _check_smiles_for_pains('O=C1C=CC(=O)C=C1')  
        # Returns 'PAINS: quinone; redox cycling electrophile' (benzoquinone flagged)
        _check_smiles_for_pains('Oc1ccccc1CN(C)C')
        # Returns 'PAINS: Mannich phenol; unstable, covalent risk' (ortho-Mannich base)
        
    Note:
        PAINS patterns are highly specific to problematic scaffolds observed in screening,
        not generic functional group filters. A molecule may contain a motif like thiourea
        or enone but still pass if the specific structural context doesn't match PAINS patterns.
        
    Reference:
        Baell JB, Holloway GA. J Med Chem 53 (2010) 2719-2740. doi:10.1021/jm901137j
    """
    # Handle invalid input types
    if not isinstance(smiles, str):
        return "Failed: Invalid SMILES string"
    
    mol = MolFromSmiles(smiles)
    if mol is None:
        return "Failed: Invalid SMILES string"
    
    try:
        matched_patterns = []
        pains_dict = _get_compiled_pains_patterns()
        
        for name, info in pains_dict.items():
            compiled_pattern = info["pattern"]
            description = info["explanation"]

            # Use pre-compiled pattern directly (much faster than _mol_has_pattern)
            if mol.HasSubstructMatch(compiled_pattern):
                matched_patterns.append(description)

        if not matched_patterns:
            return "Passed"

        return 'PAINS: ' + ', '.join(matched_patterns)
    
    except Exception as e:
        return f"Failed: {str(e)}"

