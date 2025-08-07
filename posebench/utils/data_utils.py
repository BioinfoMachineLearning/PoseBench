# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
import re
import shutil
import subprocess  # nosec
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import rootutils
from beartype import beartype
from beartype.typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Structure import Structure
from biopandas.pdb import PandasPdb
from prody import parsePDB, writePDB, writePDBStream
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench.data.components.protein_apo_to_holo_alignment import read_molecule

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants.
AMINO_ACID_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

ID_TO_HHBLITS_AA = {
    0: "A",
    1: "C",  # Also U.
    2: "D",  # Also B.
    3: "E",  # Also Z.
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",  # Includes J and O.
    21: "-",
}

MODIFIED_TO_NATURAL_AMINO_ACID_RESNAME_MAP = {
    "004": "GLY",
    "02K": "ALA",
    "02L": "ASN",
    "02O": "ALA",
    "02Y": "LEU",
    "03Y": "CYS",
    "04Q": "GLY",
    "04U": "PRO",
    "04V": "PRO",
    "05N": "PRO",
    "05O": "TYR",
    "07O": "CYS",
    "0A0": "ASP",
    "0A1": "TYR",
    "0A2": "LYS",
    "0A8": "CYS",
    "0A9": "ALA",
    "0AA": "VAL",
    "0AB": "VAL",
    "0AC": "GLY",
    "0AF": "TRP",
    "0AH": "SER",
    "0AK": "ASP",
    "0AR": "ARG",
    "0BN": "PHE",
    "0CS": "ALA",
    "0E5": "THR",
    "0EA": "TYR",
    "0FL": "ALA",
    "0LF": "PRO",
    "0PR": "TYR",
    "0QL": "ALA",
    "0RJ": "ALA",
    "0WZ": "TYR",
    "0X9": "ARG",
    "0Y8": "PRO",
    "11Q": "PRO",
    "12X": "PRO",
    "12Y": "PRO",
    "13E": "MET",
    "143": "CYS",
    "19W": "VAL",
    "1C3": "PRO",
    "1IP": "ASN",
    "1MH": "ALA",
    "1OP": "TYR",
    "1PA": "PHE",
    "1PI": "ALA",
    "1TQ": "TRP",
    "1TY": "TYR",
    "1VR": "VAL",
    "1X6": "SER",
    "200": "PHE",
    "22G": "TYR",
    "23F": "PHE",
    "23P": "ALA",
    "28X": "THR",
    "2AG": "ALA",
    "2CO": "CYS",
    "2FM": "MET",
    "2GX": "ALA",
    "2HF": "HIS",
    "2JC": "GLY",
    "2JH": "ALA",
    "2KK": "LYS",
    "2KP": "LYS",
    "2L5": "ALA",
    "2LT": "TYR",
    "2LU": "LEU",
    "2ML": "LEU",
    "2MR": "ARG",
    "2MT": "PRO",
    "2OR": "ARG",
    "2P0": "PRO",
    "2QZ": "THR",
    "2R3": "TYR",
    "2RX": "SER",
    "2SO": "ALA",
    "2TY": "TYR",
    "2VA": "VAL",
    "2XA": "CYS",
    "2YC": "LYS",
    "2YF": "LYS",
    "2YG": "LYS",
    "2ZC": "SER",
    "30V": "CYS",
    "31Q": "CYS",
    "33S": "ALA",
    "33W": "ALA",
    "34E": "VAL",
    "35Y": "ALA",
    "3AH": "HIS",
    "3AR": "ARG",
    "3BY": "PRO",
    "3CF": "PHE",
    "3CT": "TYR",
    "3GA": "ALA",
    "3GL": "GLU",
    "3MD": "ASP",
    "3NF": "TYR",
    "3PM": "SER",
    "3PX": "PRO",
    "3QN": "LYS",
    "3TY": "ALA",
    "3WS": "ALA",
    "3X9": "ALA",
    "3XH": "GLY",
    "3YM": "TYR",
    "3ZL": "LYS",
    "41H": "ALA",
    "41Q": "SER",
    "432": "SER",
    "4AF": "ALA",
    "4AK": "LYS",
    "4AW": "TRP",
    "4BF": "TYR",
    "4CF": "PHE",
    "4CY": "MET",
    "4DP": "TRP",
    "4FB": "PRO",
    "4FW": "TRP",
    "4GJ": "CYS",
    "4HH": "SER",
    "4HJ": "SER",
    "4HT": "TRP",
    "4II": "PHE",
    "4IN": "TRP",
    "4J4": "CYS",
    "4KY": "PRO",
    "4L0": "PRO",
    "4LZ": "TYR",
    "4OG": "TRP",
    "4OU": "ALA",
    "4PH": "PHE",
    "4PQ": "TRP",
    "4QK": "LEU",
    "4SJ": "ALA",
    "4U7": "ALA",
    "51T": "TYR",
    "54C": "TRP",
    "55I": "PHE",
    "56A": "HIS",
    "5AB": "ALA",
    "5CR": "PHE",
    "5CS": "CYS",
    "5CT": "LYS",
    "5CW": "TRP",
    "5DW": "ALA",
    "5FQ": "ALA",
    "5GM": "LEU",
    "5HP": "GLU",
    "5OL": "LYS",
    "5OW": "LYS",
    "5PG": "GLY",
    "5R5": "SER",
    "5T3": "LYS",
    "5VV": "ASN",
    "5X8": "CYS",
    "66C": "ALA",
    "66D": "LEU",
    "66E": "VAL",
    "6BR": "THR",
    "6CL": "LYS",
    "6CV": "ALA",
    "6CW": "TRP",
    "6DU": "ALA",
    "6FL": "LEU",
    "6G4": "LYS",
    "6GL": "ALA",
    "6HN": "LYS",
    "6KM": "CYS",
    "6M6": "CYS",
    "6V1": "CYS",
    "6WK": "CYS",
    "6ZS": "VAL",
    "73N": "ARG",
    "73O": "TYR",
    "73P": "LYS",
    "74P": "LYS",
    "7C9": "SER",
    "7CC": "ASN",
    "7HA": "GLY",
    "7ID": "ASP",
    "7J4": "ASN",
    "7JA": "ILE",
    "7N8": "PHE",
    "7O5": "ALA",
    "7QK": "LYS",
    "7T2": "PHE",
    "7TK": "ASP",
    "7VN": "ALA",
    "7VU": "PHE",
    "7W2": "PHE",
    "81R": "VAL",
    "81S": "VAL",
    "823": "ASN",
    "85G": "GLN",
    "85J": "GLN",
    "85L": "CYS",
    "8RE": "LYS",
    "8SP": "SER",
    "8WY": "LEU",
    "99Y": "ALA",
    "9AT": "THR",
    "9DN": "ASN",
    "9E7": "LYS",
    "9EV": "LEU",
    "9KK": "LEU",
    "9KP": "LYS",
    "9NE": "GLU",
    "9NF": "PHE",
    "9NR": "ARG",
    "9NV": "VAL",
    "9OW": "ALA",
    "9R4": "ALA",
    "9R7": "GLY",
    "9TR": "LYS",
    "9TU": "LYS",
    "9TX": "LYS",
    "9U0": "LYS",
    "9U6": "ALA",
    "9U9": "LYS",
    "9UC": "LYS",
    "9UF": "LYS",
    "9WV": "ALA",
    "A30": "TYR",
    "A3U": "ALA",
    "A5N": "ASN",
    "A66": "LYS",
    "A9D": "PRO",
    "AA3": "ALA",
    "AA4": "ALA",
    "AAR": "ARG",
    "ABA": "ALA",
    "AC5": "LEU",
    "ACB": "ASP",
    "ACL": "ARG",
    "AEA": "CYS",
    "AEI": "ASP",
    "AFA": "ASN",
    "AGM": "ARG",
    "AGQ": "ALA",
    "AGT": "CYS",
    "AHB": "ASN",
    "AHO": "ALA",
    "AHP": "ALA",
    "AIB": "ALA",
    "AKL": "ASP",
    "ALA": "ALA",
    "ALC": "ALA",
    "ALM": "ALA",
    "ALN": "ALA",
    "ALO": "THR",
    "ALS": "ALA",
    "ALT": "ALA",
    "ALY": "LYS",
    "AN8": "ALA",
    "API": "LYS",
    "APK": "LYS",
    "APM": "ALA",
    "AR2": "ARG",
    "AR4": "GLU",
    "ARG": "ARG",
    "ARM": "ARG",
    "ARO": "ARG",
    "ARV": "ARG",
    "AS7": "ASN",
    "ASA": "ASP",
    "ASB": "ASP",
    "ASI": "ASP",
    "ASK": "ASP",
    "ASL": "ASP",
    "ASN": "ASN",
    "ASP": "ASP",
    "ASQ": "ASP",
    "AYA": "ALA",
    "AZH": "ALA",
    "AZK": "LYS",
    "AZS": "SER",
    "AZY": "TYR",
    "B1F": "PHE",
    "B3A": "ALA",
    "B3E": "GLU",
    "B3K": "LYS",
    "B3L": "LEU",
    "B3Q": "GLN",
    "B3U": "HIS",
    "B3X": "ASN",
    "B3Y": "TYR",
    "BB6": "CYS",
    "BB7": "CYS",
    "BB8": "PHE",
    "BB9": "CYS",
    "BBC": "CYS",
    "BCS": "CYS",
    "BFD": "ASP",
    "BG1": "SER",
    "BH2": "ASP",
    "BHD": "ASP",
    "BIF": "PHE",
    "BIL": "ILE",
    "BIU": "ILE",
    "BL2": "LEU",
    "BMT": "THR",
    "BNN": "PHE",
    "BP5": "ALA",
    "BPE": "CYS",
    "BSE": "SER",
    "BTA": "LEU",
    "BTC": "CYS",
    "BTK": "LYS",
    "BTR": "TRP",
    "BUC": "CYS",
    "BUG": "VAL",
    "BWB": "SER",
    "BXT": "SER",
    "BYR": "TYR",
    "C1T": "CYS",
    "C1X": "LYS",
    "C22": "ALA",
    "C3Y": "CYS",
    "C4R": "CYS",
    "C5C": "CYS",
    "C66": "LYS",
    "C6C": "CYS",
    "CAB": "VAL",
    "CAF": "CYS",
    "CAS": "CYS",
    "CAY": "CYS",
    "CCS": "CYS",
    "CDV": "VAL",
    "CEA": "CYS",
    "CG6": "CYS",
    "CGA": "GLU",
    "CGH": "GLY",
    "CGU": "GLU",
    "CGV": "CYS",
    "CHG": "GLY",
    "CHP": "GLY",
    "CIR": "ARG",
    "CLE": "LEU",
    "CLG": "LYS",
    "CLH": "LYS",
    "CME": "CYS",
    "CMH": "CYS",
    "CML": "CYS",
    "CMT": "CYS",
    "CNG": "GLY",
    "CR5": "GLY",
    "CS0": "CYS",
    "CS1": "CYS",
    "CS3": "CYS",
    "CS4": "CYS",
    "CSA": "CYS",
    "CSB": "CYS",
    "CSD": "CYS",
    "CSE": "CYS",
    "CSJ": "CYS",
    "CSK": "CYS",
    "CSO": "CYS",
    "CSP": "CYS",
    "CSR": "CYS",
    "CSS": "CYS",
    "CSU": "CYS",
    "CSW": "CYS",
    "CSX": "CYS",
    "CSZ": "CYS",
    "CTE": "TRP",
    "CTH": "THR",
    "CWD": "ALA",
    "CWR": "SER",
    "CXM": "MET",
    "CY0": "CYS",
    "CY1": "CYS",
    "CY3": "CYS",
    "CY4": "CYS",
    "CYA": "CYS",
    "CYF": "CYS",
    "CYG": "CYS",
    "CYJ": "LYS",
    "CYM": "CYS",
    "CYQ": "CYS",
    "CYR": "CYS",
    "CYS": "CYS",
    "CYW": "CYS",
    "CZ2": "CYS",
    "CZS": "ALA",
    "CZZ": "CYS",
    "D0Q": "TRP",
    "DA2": "ARG",
    "DAB": "ALA",
    "DAH": "PHE",
    "DAM": "ALA",
    "DBS": "SER",
    "DBU": "THR",
    "DBY": "TYR",
    "DBZ": "ALA",
    "DC2": "CYS",
    "DDE": "HIS",
    "DDZ": "ALA",
    "DHA": "SER",
    "DHN": "VAL",
    "DI7": "TYR",
    "DIR": "ARG",
    "DJD": "ALA",
    "DLS": "LYS",
    "DM0": "LYS",
    "DMH": "ASN",
    "DMK": "ASP",
    "DNL": "LYS",
    "DNP": "ALA",
    "DNS": "LYS",
    "DNW": "ALA",
    "DO2": "LEU",
    "DOH": "ASP",
    "DON": "LEU",
    "DPL": "PRO",
    "DPP": "ALA",
    "DPQ": "TYR",
    "DV7": "GLY",
    "DV9": "GLU",
    "DYA": "ASP",
    "DYJ": "PRO",
    "DYS": "CYS",
    "E0Y": "PRO",
    "E95": "TRP",
    "E9C": "TYR",
    "E9M": "TRP",
    "E9V": "HIS",
    "ECC": "GLN",
    "ECX": "CYS",
    "EFC": "CYS",
    "EHP": "PHE",
    "EI4": "ARG",
    "EJA": "CYS",
    "ELY": "LYS",
    "EO2": "LEU",
    "EOE": "PRO",
    "ESB": "TYR",
    "ESC": "MET",
    "EU0": "VAL",
    "EUP": "THR",
    "EW6": "SER",
    "EXA": "LYS",
    "EXL": "TRP",
    "EXY": "LEU",
    "EYS": "CYS",
    "F0G": "ALA",
    "F2F": "PHE",
    "F2Y": "TYR",
    "F7P": "TRP",
    "F7Q": "TYR",
    "F7S": "LEU",
    "F7W": "TRP",
    "FAK": "LYS",
    "FB5": "ALA",
    "FB6": "ALA",
    "FCL": "PHE",
    "FDL": "LYS",
    "FF9": "LYS",
    "FFM": "CYS",
    "FGA": "GLU",
    "FGL": "GLY",
    "FGP": "SER",
    "FH7": "LYS",
    "FHL": "LYS",
    "FHO": "LYS",
    "FL6": "ASP",
    "FLA": "ALA",
    "FLE": "LEU",
    "FLT": "TYR",
    "FME": "MET",
    "FOE": "CYS",
    "FP9": "PRO",
    "FPK": "PRO",
    "FQA": "LYS",
    "FT6": "TRP",
    "FTR": "TRP",
    "FTY": "TYR",
    "FVA": "VAL",
    "FY2": "TYR",
    "FY3": "TYR",
    "FZN": "LYS",
    "G1X": "TYR",
    "G3M": "LYS",
    "G8M": "GLY",
    "GAU": "GLU",
    "GEE": "GLY",
    "GFT": "SER",
    "GGL": "GLU",
    "GHG": "GLN",
    "GL3": "GLY",
    "GLH": "GLN",
    "GLJ": "GLU",
    "GLN": "GLN",
    "GLQ": "GLU",
    "GLU": "GLU",
    "GLY": "GLY",
    "GLZ": "GLY",
    "GMA": "GLU",
    "GNC": "GLN",
    "GPL": "LYS",
    "GQI": "CYS",
    "GSC": "GLY",
    "GSU": "GLU",
    "GT9": "CYS",
    "GVL": "SER",
    "H14": "PHE",
    "H5M": "PRO",
    "HAC": "ALA",
    "HAR": "ARG",
    "HBN": "HIS",
    "HCM": "CYS",
    "HGY": "GLY",
    "HHI": "HIS",
    "HIA": "HIS",
    "HIC": "HIS",
    "HIP": "HIS",
    "HIQ": "HIS",
    "HIS": "HIS",
    "HIX": "ALA",
    "HL2": "LEU",
    "HL5": "LEU",
    "HLU": "LEU",
    "HLX": "LEU",
    "HLY": "LYS",
    "HMR": "ARG",
    "HNC": "CYS",
    "HOO": "HIS",
    "HOX": "ALA",
    "HPC": "PHE",
    "HPE": "PHE",
    "HPQ": "PHE",
    "HQA": "ALA",
    "HR7": "ARG",
    "HRG": "ARG",
    "HRP": "TRP",
    "HS8": "HIS",
    "HS9": "HIS",
    "HSE": "SER",
    "HSK": "HIS",
    "HSL": "SER",
    "HSO": "HIS",
    "HSV": "HIS",
    "HT7": "TRP",
    "HTI": "CYS",
    "HTR": "TRP",
    "HV5": "ALA",
    "HVA": "VAL",
    "HY3": "PRO",
    "HYP": "PRO",
    "HZP": "PRO",
    "I2F": "LYS",
    "I2M": "ILE",
    "I3L": "CYS",
    "I4G": "GLY",
    "I4O": "HIS",
    "I58": "LYS",
    "I7F": "SER",
    "IAE": "PHE",
    "IAM": "ALA",
    "IAR": "ARG",
    "IAS": "ASP",
    "IB9": "TYR",
    "IC0": "GLY",
    "ICY": "CYS",
    "IEL": "LYS",
    "IGL": "GLY",
    "IIL": "ILE",
    "ILE": "ILE",
    "ILG": "GLU",
    "ILX": "ILE",
    "IML": "ILE",
    "IPG": "GLY",
    "IT1": "LYS",
    "IYR": "TYR",
    "IZO": "MET",
    "J2F": "TYR",
    "J3D": "CYS",
    "J8W": "SER",
    "J9A": "GLN",
    "JJJ": "CYS",
    "JJK": "CYS",
    "JJL": "CYS",
    "JKH": "PRO",
    "JLP": "LYS",
    "K1R": "CYS",
    "K5H": "CYS",
    "K5L": "SER",
    "K7K": "SER",
    "KBE": "LYS",
    "KCJ": "ALA",
    "KCR": "LYS",
    "KCX": "LYS",
    "KEO": "LYS",
    "KGC": "LYS",
    "KHB": "LYS",
    "KNB": "ALA",
    "KOR": "MET",
    "KPF": "LYS",
    "KPI": "LYS",
    "KPY": "LYS",
    "KR3": "LYS",
    "KST": "LYS",
    "KYN": "TRP",
    "KYQ": "LYS",
    "L3O": "LEU",
    "L4R": "ALA",
    "LA2": "LYS",
    "LAA": "ASP",
    "LAL": "ALA",
    "LAY": "LEU",
    "LBY": "LYS",
    "LBZ": "LYS",
    "LCK": "LYS",
    "LCX": "LYS",
    "LDH": "LYS",
    "LE1": "VAL",
    "LED": "LEU",
    "LEF": "LEU",
    "LEH": "LEU",
    "LEM": "LEU",
    "LET": "LYS",
    "LEU": "LEU",
    "LGY": "LYS",
    "LLO": "LYS",
    "LLP": "LYS",
    "LLY": "LYS",
    "LLZ": "LYS",
    "LME": "GLU",
    "LMF": "LYS",
    "LMQ": "GLN",
    "LNE": "LEU",
    "LNM": "LEU",
    "LOU": "ALA",
    "LP6": "LYS",
    "LPD": "PRO",
    "LPG": "GLY",
    "LPH": "GLY",
    "LPS": "SER",
    "LSO": "LYS",
    "LT0": "SER",
    "LTR": "TRP",
    "LVG": "GLY",
    "LVN": "VAL",
    "LWI": "PHE",
    "LYF": "LYS",
    "LYH": "LYS",
    "LYM": "LYS",
    "LYN": "LYS",
    "LYO": "LYS",
    "LYR": "LYS",
    "LYS": "LYS",
    "LYU": "LYS",
    "LYV": "GLY",
    "LYX": "LYS",
    "LYZ": "LYS",
    "M0H": "CYS",
    "M2L": "LYS",
    "M2S": "MET",
    "M30": "GLY",
    "M3L": "LYS",
    "MAA": "ALA",
    "MAI": "ARG",
    "MBQ": "TYR",
    "MC1": "SER",
    "MCG": "GLY",
    "MCL": "LYS",
    "MCS": "CYS",
    "MD3": "CYS",
    "MD5": "CYS",
    "MD6": "GLY",
    "MDF": "TYR",
    "ME0": "MET",
    "MEA": "PHE",
    "MEG": "GLU",
    "MEN": "ASN",
    "MEQ": "GLN",
    "MET": "MET",
    "MEU": "GLY",
    "MF3": "MET",
    "MGG": "ARG",
    "MGN": "GLN",
    "MGY": "GLY",
    "MH1": "HIS",
    "MH6": "SER",
    "MHL": "LEU",
    "MHO": "MET",
    "MHS": "HIS",
    "MHU": "PHE",
    "MIR": "SER",
    "MIS": "SER",
    "MK8": "LEU",
    "ML3": "LYS",
    "MLE": "LEU",
    "MLL": "LEU",
    "MLY": "LYS",
    "MLZ": "LYS",
    "MME": "MET",
    "MMO": "ARG",
    "MNL": "LEU",
    "MNV": "VAL",
    "MP8": "PRO",
    "MPH": "MET",
    "MPJ": "MET",
    "MPQ": "GLY",
    "MSA": "GLY",
    "MSE": "MET",
    "MSL": "MET",
    "MSO": "MET",
    "MT2": "MET",
    "MTY": "TYR",
    "MVA": "VAL",
    "MYK": "LYS",
    "N0A": "ALA",
    "N10": "SER",
    "N2C": "CYS",
    "N65": "LYS",
    "N7P": "PRO",
    "N80": "PRO",
    "N9P": "ALA",
    "NA8": "ALA",
    "NAL": "ALA",
    "NAM": "ALA",
    "NB8": "ASN",
    "NBQ": "TYR",
    "NC1": "SER",
    "NCB": "ALA",
    "NCY": "CYS",
    "NEM": "HIS",
    "NEP": "HIS",
    "NFA": "PHE",
    "NIY": "TYR",
    "NKS": "GLY",
    "NLB": "LEU",
    "NLE": "LEU",
    "NLN": "LEU",
    "NLO": "LEU",
    "NLP": "LEU",
    "NLQ": "GLN",
    "NLW": "LEU",
    "NLY": "GLY",
    "NMC": "GLY",
    "NMM": "ARG",
    "NNH": "ARG",
    "NOT": "LEU",
    "NPH": "CYS",
    "NPI": "ALA",
    "NRG": "ARG",
    "NSK": "LYS",
    "NTR": "TYR",
    "NTY": "TYR",
    "NVA": "VAL",
    "NWD": "ALA",
    "NYB": "CYS",
    "NYS": "CYS",
    "NZC": "THR",
    "NZH": "HIS",
    "O2E": "SER",
    "O6H": "TRP",
    "O7A": "THR",
    "O7D": "TRP",
    "OAS": "SER",
    "OBS": "LYS",
    "OCS": "CYS",
    "OCY": "CYS",
    "OGC": "ALA",
    "OHI": "HIS",
    "OHS": "ASP",
    "OJY": "VAL",
    "OLD": "HIS",
    "OLT": "THR",
    "OLZ": "SER",
    "OMH": "SER",
    "OMT": "MET",
    "OMX": "TYR",
    "OMY": "TYR",
    "ONH": "ALA",
    "ONL": "LEU",
    "ORN": "ALA",
    "ORQ": "ARG",
    "OSE": "SER",
    "OTH": "THR",
    "OTY": "TYR",
    "OXX": "ASP",
    "OYL": "HIS",
    "OZW": "PHE",
    "P1L": "CYS",
    "P2Q": "TYR",
    "P2Y": "PRO",
    "P3Q": "TYR",
    "P9S": "CYS",
    "PAQ": "TYR",
    "PAS": "ASP",
    "PAT": "TRP",
    "PBB": "CYS",
    "PBF": "PHE",
    "PCA": "GLN",
    "PCC": "PRO",
    "PCS": "PHE",
    "PDL": "ALA",
    "PE1": "LYS",
    "PEC": "CYS",
    "PF5": "PHE",
    "PFF": "PHE",
    "PG1": "SER",
    "PGY": "GLY",
    "PH6": "PRO",
    "PH8": "VAL",
    "PHA": "PHE",
    "PHD": "ASP",
    "PHE": "PHE",
    "PHI": "PHE",
    "PHL": "PHE",
    "PJ3": "HIS",
    "PLJ": "PRO",
    "PM3": "PHE",
    "POK": "ARG",
    "POM": "PRO",
    "PPN": "PHE",
    "PR3": "CYS",
    "PR4": "PRO",
    "PR7": "PRO",
    "PRK": "LYS",
    "PRO": "PRO",
    "PRR": "ALA",
    "PRS": "PRO",
    "PRV": "GLY",
    "PSH": "HIS",
    "PSW": "ALA",
    "PTH": "TYR",
    "PTM": "TYR",
    "PTR": "TYR",
    "PVH": "HIS",
    "PXU": "PRO",
    "PYA": "ALA",
    "PYH": "LYS",
    "PYL": "LYS",
    "PYX": "CYS",
    "Q2E": "TRP",
    "Q3P": "LYS",
    "Q75": "MET",
    "Q78": "PHE",
    "Q8X": "CYS",
    "QAC": "ALA",
    "QCI": "GLN",
    "QCS": "CYS",
    "QIL": "ILE",
    "QM8": "LEU",
    "QMB": "VAL",
    "QMM": "GLN",
    "QNQ": "CYS",
    "QNT": "CYS",
    "QNW": "CYS",
    "QNY": "THR",
    "QO2": "CYS",
    "QO5": "CYS",
    "QO8": "CYS",
    "QPA": "CYS",
    "QPH": "PHE",
    "QQ8": "GLN",
    "QVA": "CYS",
    "QX7": "ALA",
    "R0K": "GLU",
    "R1A": "CYS",
    "R2T": "GLN",
    "R4K": "TRP",
    "RE0": "TRP",
    "RE3": "TRP",
    "RF9": "LYS",
    "RGL": "ARG",
    "RGP": "GLU",
    "RON": "VAL",
    "RPI": "ARG",
    "RT0": "PRO",
    "RVJ": "ALA",
    "RVX": "SER",
    "RX9": "ILE",
    "RXL": "VAL",
    "RZ4": "SER",
    "S0R": "LEU",
    "S12": "SER",
    "S1H": "SER",
    "S2C": "CYS",
    "S2P": "ALA",
    "SAC": "SER",
    "SAH": "CYS",
    "SAO": "CYS",
    "SAR": "GLY",
    "SBL": "SER",
    "SCH": "CYS",
    "SCS": "CYS",
    "SCY": "CYS",
    "SD2": "MET",
    "SD4": "ASN",
    "SDP": "SER",
    "SEB": "SER",
    "SEC": "CYS",
    "SEE": "SER",
    "SEG": "ALA",
    "SEL": "SER",
    "SEM": "SER",
    "SEN": "SER",
    "SEP": "SER",
    "SER": "SER",
    "SET": "SER",
    "SFE": "ALA",
    "SGB": "SER",
    "SHC": "CYS",
    "SHP": "GLY",
    "SHR": "LYS",
    "SIB": "CYS",
    "SKG": "ILE",
    "SKH": "LYS",
    "SKJ": "LEU",
    "SLL": "LYS",
    "SLZ": "LYS",
    "SMC": "CYS",
    "SME": "MET",
    "SMF": "PHE",
    "SNC": "CYS",
    "SNK": "HIS",
    "SNN": "ASN",
    "SOC": "CYS",
    "SOY": "SER",
    "SRZ": "SER",
    "STY": "TYR",
    "SUB": "ALA",
    "SUN": "SER",
    "SVA": "SER",
    "SVV": "SER",
    "SVW": "SER",
    "SVX": "SER",
    "SVY": "SER",
    "SVZ": "SER",
    "SWW": "SER",
    "SXE": "SER",
    "SYS": "ALA",
    "SZF": "TRP",
    "T0I": "TYR",
    "T11": "PHE",
    "T3R": "LEU",
    "T66": "LYS",
    "T8L": "THR",
    "T9E": "THR",
    "TAV": "ASP",
    "TBG": "VAL",
    "TBM": "THR",
    "TCQ": "TYR",
    "TCR": "TRP",
    "TEF": "ALA",
    "TFQ": "PHE",
    "TGH": "TRP",
    "TH5": "THR",
    "TH6": "THR",
    "THC": "THR",
    "THO": "THR",
    "THR": "THR",
    "THZ": "ARG",
    "TIH": "ALA",
    "TIS": "SER",
    "TLY": "LYS",
    "TMB": "THR",
    "TMD": "THR",
    "TNB": "CYS",
    "TNQ": "TRP",
    "TNR": "SER",
    "TOQ": "TRP",
    "TOX": "TRP",
    "TPJ": "PRO",
    "TPK": "PRO",
    "TPL": "TRP",
    "TPO": "THR",
    "TPQ": "TYR",
    "TQI": "TRP",
    "TQQ": "TRP",
    "TQZ": "CYS",
    "TRF": "TRP",
    "TRG": "LYS",
    "TRN": "TRP",
    "TRO": "TRP",
    "TRP": "TRP",
    "TRQ": "TRP",
    "TRW": "TRP",
    "TRX": "TRP",
    "TRY": "TRP",
    "TS9": "ILE",
    "TSQ": "ALA",
    "TST": "LEU",
    "TSY": "CYS",
    "TTQ": "TRP",
    "TTS": "TYR",
    "TXY": "TYR",
    "TY1": "TYR",
    "TY2": "TYR",
    "TY3": "TYR",
    "TY5": "TYR",
    "TY8": "ALA",
    "TY9": "ALA",
    "TYB": "TYR",
    "TYE": "TYR",
    "TYI": "TYR",
    "TYJ": "TYR",
    "TYN": "TYR",
    "TYO": "TYR",
    "TYQ": "TYR",
    "TYR": "TYR",
    "TYS": "TYR",
    "TYT": "TYR",
    "TYX": "CYS",
    "TYY": "TYR",
    "U2X": "TYR",
    "U3X": "ALA",
    "UF0": "SER",
    "UGY": "GLY",
    "UKD": "ALA",
    "UKY": "ALA",
    "UM2": "ALA",
    "UMA": "ALA",
    "UOX": "ALA",
    "UXY": "LYS",
    "V44": "CYS",
    "V61": "PHE",
    "V6W": "TRP",
    "VAD": "VAL",
    "VAF": "VAL",
    "VAH": "VAL",
    "VAL": "VAL",
    "VB1": "LYS",
    "VH0": "PRO",
    "VHF": "GLU",
    "VI3": "CYS",
    "VLM": "VAL",
    "VOL": "VAL",
    "VPV": "LYS",
    "VVK": "ALA",
    "WFP": "ALA",
    "WLU": "LEU",
    "WPA": "PHE",
    "WRP": "TRP",
    "WVL": "VAL",
    "WYK": "ARG",
    "WZJ": "ILE",
    "X2W": "GLU",
    "X60": "VAL",
    "XA6": "PHE",
    "XCN": "CYS",
    "XPL": "LYS",
    "XPR": "PRO",
    "XSN": "ASN",
    "XX1": "LYS",
    "Y1V": "LEU",
    "Y57": "LYS",
    "YCM": "CYS",
    "YHA": "LYS",
    "YNM": "TYR",
    "YOF": "TYR",
    "YPR": "PRO",
    "YPZ": "ALA",
    "YTH": "THR",
    "Z01": "ALA",
    "Z3E": "THR",
    "Z70": "HIS",
    "ZAI": "LYS",
    "ZBZ": "CYS",
    "ZCL": "PHE",
    "ZDJ": "TYR",
    "ZIQ": "TRP",
    "ZNY": "PRO",
    "ZT6": "TYR",
    "ZU0": "THR",
    "ZV4": "PHE",
    "ZYJ": "PRO",
    "ZYK": "PRO",
    "ZZD": "CYS",
    "ZZJ": "ALA",
    "ZZU": "ARG",
}


@beartype
def extract_remarks_from_pdb(pdb_file: str, remark_number: Optional[int] = None) -> List[str]:
    """Extract REMARK statements from a PDB file.

    :param pdb_file: Path to the PDB file.
    :param remark_number: Specific REMARK number to filter. If None,
        extracts all REMARKs.
    :return list: List of REMARK statements.
    """
    remarks = []
    with open(pdb_file) as file:
        for line in file:
            if line.startswith("REMARK"):
                if remark_number is not None:
                    if line[7:10].strip() == str(remark_number):
                        remarks.append(line.strip())
                else:
                    remarks.append(line.strip())
    return remarks


@beartype
def parse_inference_inputs_from_dir(
    input_data_dir: Union[str, Path], pdb_ids: Optional[Set[Any]] = None
) -> List[Tuple[str, str]]:
    """Parse a data directory containing subdirectories of protein-ligand
    complexes and return corresponding SMILES strings and PDB IDs.

    :param input_data_dir: Path to the input data directory.
    :param pdb_ids: Optional set of IDs by which to filter processing.
    :return: A list of tuples each containing a SMILES string and a PDB
        ID.
    """
    smiles_and_pdb_id_list = []
    casp_dataset_requested = os.path.basename(input_data_dir) == "targets"

    parser = PDBParser()

    num_skipped_inputs = 0
    if casp_dataset_requested:
        # parse CASP inputs uniquely
        smiles_filepaths = list(glob.glob(os.path.join(input_data_dir, "*.smiles.txt")))
        for smiles_filepath in smiles_filepaths:
            pdb_id = os.path.basename(smiles_filepath).split(".")[0]
            smiles_df = pd.read_csv(smiles_filepath, delim_whitespace=True)
            assert smiles_df.columns.tolist() == [
                "ID",
                "Name",
                "SMILES",
                "Relevant",
            ], "SMILES DataFrame must have columns ['ID', 'Name', 'SMILES', 'Relevant']."
            mol_smiles = ".".join(smiles_df["SMILES"].tolist())
            assert len(mol_smiles) > 0, f"SMILES string for {pdb_id} cannot be empty."
            smiles_and_pdb_id_list.append((mol_smiles, pdb_id))

    else:
        ligand_expo_mapping = read_ligand_expo()
        input_data_dirs = [
            item
            for item in os.listdir(input_data_dir)
            # e.g., skip sequence files and predicted structure directories
            if not any(substr in item.lower() for substr in ["sequence", "structure"])
            # e.g., skip PoseBusters Benchmark PDBs that contain crystal contacts
            # reference: https://github.com/maabuu/posebusters/issues/26
            and not (pdb_ids is not None and item not in pdb_ids)
        ]
        for pdb_name in tqdm(input_data_dirs, desc="Parsing input data directory"):
            pdb_dir = os.path.join(input_data_dir, pdb_name)
            if os.path.isdir(pdb_dir):
                mol = None
                pdb_id = os.path.split(pdb_dir)[-1]

                # NOTE: the Astex Diverse and PoseBusters Benchmark datasets use `.sdf` files to store their primary
                # ligands, but we want to extract all cofactors as well to enhance model context for predictions
                if os.path.exists(os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf")):
                    supplier = Chem.SDMolSupplier(
                        os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf"),
                        sanitize=True,
                        removeHs=True,
                    )
                    pdb_mol = extract_protein_and_ligands_with_prody(
                        os.path.join(pdb_dir, f"{pdb_id}_protein.pdb"),
                        protein_output_pdb_file=None,
                        ligands_output_sdf_file=None,
                        write_output_files=False,
                        ligand_expo_mapping=ligand_expo_mapping,
                    )

                    assert len(supplier) == 1, f"Expected one ligand molecule for PDB ID {pdb_id}"
                    pdb_mols = [pdb_mol] if pdb_mol is not None else []
                    mol = combine_molecules([*supplier, *pdb_mols])

                # NOTE: DockGen uses `.pdb` files to store its ligands
                if mol is None and os.path.exists(
                    os.path.join(pdb_dir, f"{pdb_id.split('_')[0]}_processed.pdb")
                ):
                    # ensure the primary ligand exists in the crystal structure of the first bioassembly, and then
                    # extract all cofactors in the protein-ligand complex to enhance model context for predictions
                    assert os.path.exists(
                        os.path.join(pdb_dir, f"{pdb_id}_ligand.pdb")
                    ), f"Could not find primary ligand file for PDB ID {pdb_id}"
                    remarks = extract_remarks_from_pdb(
                        os.path.join(pdb_dir, f"{pdb_id}_ligand.pdb")
                    )

                    primary_ligand_structure = parser.get_structure(
                        "protein", os.path.join(pdb_dir, f"{pdb_id}_ligand.pdb")
                    )
                    assert (
                        primary_ligand_structure is not None
                    ), f"Could not parse primary ligand structure for PDB ID {pdb_id}"

                    primary_ligand_resnames = [
                        res.resname for res in primary_ligand_structure.get_residues()
                    ]
                    assert (
                        len(primary_ligand_resnames) > 0
                    ), f"Expected at least one primary ligand residue for PDB ID {pdb_id}"

                    assert (
                        len(remarks) > 0
                    ), f"No REMARK statements found in ligand PDB file for PDB ID {pdb_id}"
                    remark = remarks[0]
                    primary_ligand_chainid = remark.split("segment ")[-1].split(")")[0]
                    primary_ligand_resnum = remark.split("num ")[-1].split(")")[0].replace("`", "")
                    if "to" in primary_ligand_resnum:
                        primary_ligand_chain_res_ids = []
                        start_resnum, end_resnum = primary_ligand_resnum.split("to")
                        num_res = min(
                            int(end_resnum) - int(start_resnum) + 1, len(primary_ligand_resnames)
                        )
                        for resnum in range(int(start_resnum), int(start_resnum) + num_res):
                            primary_ligand_chain_res_ids.append(
                                f"{primary_ligand_chainid}_{resnum}"
                            )
                    else:
                        primary_ligand_chain_res_ids = [
                            f"{primary_ligand_chainid}_{primary_ligand_resnum}"
                        ]
                    assert len(primary_ligand_resnames) == len(
                        primary_ligand_chain_res_ids
                    ), f"Expected primary ligand residues to match chain-residue IDs for PDB ID {pdb_id}"

                    structure = parser.get_structure(
                        "protein", os.path.join(pdb_dir, f"{pdb_id.split('_')[0]}_processed.pdb")
                    )
                    assert (
                        structure is not None
                    ), f"Could not parse protein structure for PDB ID {pdb_id}"

                    detected_missing_input = False
                    chain_res_id_to_name = {
                        f"{res.parent.id}_{res.id[1]}": res.resname
                        for res in structure.get_residues()
                    }
                    for ligand_idx, chain_res_id in enumerate(primary_ligand_chain_res_ids):
                        if (
                            chain_res_id not in chain_res_id_to_name
                            or chain_res_id_to_name[chain_res_id]
                            != primary_ligand_resnames[ligand_idx]
                        ):
                            logger.warning(
                                f"Primary ligand residue {primary_ligand_resnames[ligand_idx]} at chain-residue ID {chain_res_id.replace('_', '-')} not found in protein structure for PDB ID {pdb_id}. Skipping this complex."
                            )
                            detected_missing_input = True
                            break

                    if detected_missing_input:
                        num_skipped_inputs += 1
                        continue

                    mol = extract_protein_and_ligands_with_prody(
                        os.path.join(pdb_dir, f"{pdb_id.split('_')[0]}_processed.pdb"),
                        protein_output_pdb_file=None,
                        ligands_output_sdf_file=None,
                        write_output_files=False,
                        load_hetatms_as_ligands=True,
                        ligand_expo_mapping=ligand_expo_mapping,
                    )

                    if mol is None:
                        num_skipped_inputs += 1
                        logger.warning(f"Could not extract ligand molecule for PDB ID {pdb_id}")
                        continue

                if mol is None:
                    num_skipped_inputs += 1
                    logger.warning(f"Could not extract ligand molecule for PDB ID {pdb_id}")
                    continue

                mol_smiles = Chem.MolToSmiles(mol)
                if mol_smiles is None:
                    raise ValueError(
                        f"Could not convert ligand molecule to SMILES for PDB ID {pdb_id}"
                    )

                smiles_and_pdb_id_list.append((mol_smiles, pdb_id))

    if num_skipped_inputs > 0:
        logger.warning(
            f"Skipped {num_skipped_inputs} complexes with missing or unmappable primary ligand residues."
        )

    return smiles_and_pdb_id_list


@beartype
def extract_sequences_from_protein_structure_file(
    protein_filepath: Union[str, Path],
    structure: Optional[Structure] = None,
    exclude_hetero: bool = False,
) -> List[str]:
    """Extract the protein chain sequences from a protein structure file.

    :param protein_filepath: Path to the protein structure file.
    :param structure: Optional BioPython structure object to use
        instead.
    :param exclude_hetero: Whether to exclude hetero (e.g., water)
        residues.
    :return: A list of protein sequences.
    """
    if structure is None:
        # load the first model of the PDB file
        biopython_parser = PDBParser(QUIET=True)
        models = biopython_parser.get_structure("random_id", protein_filepath)
        structure = models[0]

    sequences = []
    for chain in structure:
        aa_residues = [residue for residue in chain if is_aa(residue)]
        if exclude_hetero:
            # NOTE: here, we exclude modified (hetero) amino acid residues serving as ligands
            aa_residues = [residue for residue in aa_residues if residue.id[0] == " "]

        aa_residue_names = [
            MODIFIED_TO_NATURAL_AMINO_ACID_RESNAME_MAP[residue.resname]
            for residue in aa_residues
            if residue.resname in MODIFIED_TO_NATURAL_AMINO_ACID_RESNAME_MAP
        ]
        seq = "".join(AMINO_ACID_THREE_TO_ONE.get(resname, "X") for resname in aa_residue_names)

        # skip if not a protein chain
        if not seq:
            continue

        sequences.append(seq)

    return sequences


@beartype
def combine_molecules(molecule_list: List[Chem.Mol]) -> Chem.Mol:
    """Combine a list of RDKit molecules into a single molecule.

    :param molecule_list: A list of RDKit molecules.
    :return: A single RDKit molecule.
    """
    # Initialize the combined molecule with the first molecule in the list
    new_mol = molecule_list[0]

    # Iterate through the remaining molecules and combine them pairwise
    for mol in molecule_list[1:]:
        new_mol = Chem.CombineMols(new_mol, mol)

    return new_mol


@beartype
def renumber_pdb_df_residues(input_pdb_file: str, output_pdb_file: str):
    """Renumber residues in a PDB file starting from 1 for each chain.

    :param input_pdb_file: Path to the input PDB file.
    """
    # Load the PDB file
    pdb = PandasPdb().read_pdb(input_pdb_file)

    # Iterate through each chain
    for _, chain_df in pdb.df["ATOM"].groupby("chain_id"):
        # Get the minimum residue index for the current chain
        min_residue_index = chain_df["residue_number"].min()

        # Reindex the residues starting from 1
        chain_df["residue_number"] -= min_residue_index - 1

        # Update the PDB dataframe with the reindexed chain
        pdb.df["ATOM"].loc[chain_df.index] = chain_df

    # Write the modified PDB file
    pdb.to_pdb(output_pdb_file)


@beartype
def renumber_biopython_structure_residues(
    structure: Structure, gap_insertion_point: Optional[str] = None
) -> Structure:
    """Renumber residues in a PDB file using BioPython starting from 1 for each
    chain.

    :param structure: BioPython structure object.
    :param gap_insertion_point: Optional `:`-separated string
        representing the chain-residue pair index of the residue at
        which to insert a single index gap.
    :return: BioPython structure object with renumbered residues.
    """
    # Iterate through each model in the structure
    if gap_insertion_point is not None:
        assert (
            len(gap_insertion_point.split(":")) == 2
        ), "When provided, gap insertion point must be in the format 'chain_id:residue_index'."
    gap_insertion_chain_id = (
        gap_insertion_point.split(":")[0] if gap_insertion_point is not None else None
    )
    gap_insertion_residue_index = (
        int(gap_insertion_point.split(":")[1]) if gap_insertion_point is not None else None
    )
    for model in structure:
        # Iterate through each chain in the model
        for chain in model:
            # Get the minimum residue index for the current chain
            min_residue_index = min(residue.id[1] for residue in chain)

            # Reindex the residues starting from 1
            gap_insertion_counter = 0
            for residue in chain:
                new_residue_index = residue.id[1] - min_residue_index + 1
                gap_index_found = (
                    gap_insertion_chain_id is not None
                    and gap_insertion_residue_index is not None
                    and chain.id == gap_insertion_chain_id
                    and new_residue_index == gap_insertion_residue_index
                )
                if gap_index_found:
                    gap_insertion_counter = 1
                residue.id = (" ", new_residue_index + gap_insertion_counter, residue.id[2])
                for atom in residue:
                    atom.serial_number = None  # Reset atom serial number

    return structure


def get_pdb_components_with_prody(
    input_pdb_file: str, load_hetatms_as_ligands: bool = False
) -> tuple:
    """Split a protein-ligand pdb into protein and ligand components using
    ProDy.

    :param input_pdb_file: Path to the input PDB file.
    :param load_hetatms_as_ligands: Whether to load HETATM records as
        ligands if no ligands are initially found.
    :return: Tuple of protein and ligand components.
    """
    pdb = parsePDB(input_pdb_file)
    protein = pdb.select("protein")
    ligand = pdb.select("not (protein or nucleotide or water)")

    if ligand is None and load_hetatms_as_ligands:
        ligand = pdb.select("hetatm and not water")

    return protein, ligand


def read_ligand_expo(
    ligand_expo_url: str = "http://ligand-expo.rcsb.org/dictionaries",
    ligand_expo_filename: str = "Components-smiles-stereo-oe.smi",
) -> Dict[str, Any]:
    """Read Ligand Expo data, first trying to find a file called `Components-
    smiles-stereo-oe.smi` in the current directory. If the file can't be found,
    grab it from the RCSB.

    :param ligand_expo_url: URL to Ligand Expo.
    :param ligand_expo_filename: Name of the Ligand Expo file.
    :return: Ligand Expo as a dictionary with ligand id as the key
    """
    try:
        df = pd.read_csv(
            ligand_expo_filename, sep="\t", header=None, names=["SMILES", "ID", "Name"]
        )

    except FileNotFoundError:
        r = requests.get(
            f"{ligand_expo_url}/{ligand_expo_filename}", allow_redirects=True
        )  # nosec

        with open("Components-smiles-stereo-oe.smi", "wb") as f:
            f.write(r.content)

        df = pd.read_csv(
            ligand_expo_filename, sep="\t", header=None, names=["SMILES", "ID", "Name"]
        )

    assert (
        len(df) > 0
    ), "Downloaded Ligand Expo file is empty. Please consult the Ligand Expo website for troubleshooting."

    df.set_index("ID", inplace=True)
    return df.to_dict("index")


def write_pdb_with_prody(atoms, pdb_name, add_element_types=False):
    """Write atoms to a pdb file using ProDy.

    :param atoms: atoms object from prody
    :param pdb_name: base name for the pdb file
    :param add_element_types: whether to add element types to the pdb
        file
    """
    writePDB(pdb_name, atoms)
    if add_element_types:
        with open(pdb_name.replace(".pdb", "_elem.pdb"), "w") as f:
            subprocess.run(  # nosec
                f"pdb_element {pdb_name}",
                shell=True,
                check=True,
                stdout=f,
            )
        shutil.move(pdb_name.replace(".pdb", "_elem.pdb"), pdb_name)
    logger.info(f"Wrote {pdb_name}")


def process_ligand_with_prody(
    ligand,
    res_name,
    chain,
    resnum,
    sanitize: bool = True,
    generify_resnames: bool = False,
    sub_smiles: Optional[str] = None,
    ligand_expo_mapping: Optional[Dict[str, Any]] = None,
) -> Chem.Mol:
    """
    Add bond orders to a pdb ligand using ProDy.
    1. Select the ligand component with name "res_name"
    2. Get the corresponding SMILES from pypdb
    3. Create a template molecule from the SMILES in step 2
    4. Write the PDB file to a stream
    5. Read the stream into an RDKit molecule
    6. Assign the bond orders from the template from step 3

    :param ligand: ligand as generated by prody
    :param res_name: residue name of ligand to extract
    :param chain: chain of ligand to extract
    :param resnum: residue number of ligand to extract
    :param sanitize: whether to sanitize the molecule
    :param generify_resnames: whether to generify the residue names
    :param sub_smiles: optional SMILES string of the ligand molecule
    :param ligand_expo_mapping: optional Ligand Expo mapping
    :return: molecule with bond orders assigned
    """
    sub_smiles_provided = sub_smiles is not None
    sub_mol = ligand.select(f"resname {res_name} and chain {chain} and resnum {resnum}")

    ligand_expo_mapping = ligand_expo_mapping or read_ligand_expo()
    chem_desc = ligand_expo_mapping.get(res_name)

    if chem_desc is not None and not sub_smiles_provided:
        sub_smiles = None
        if "SMILES" in chem_desc:
            sub_smiles = chem_desc["SMILES"]

    if sub_smiles is not None:
        template = AllChem.MolFromSmiles(sub_smiles)
    else:
        template = None

    if generify_resnames:
        sub_mol.setResnames("LIG")

    output = StringIO()
    writePDBStream(output, sub_mol)
    pdb_string = output.getvalue()
    rd_mol = AllChem.MolFromPDBBlock(pdb_string, sanitize=sanitize)

    if sanitize and rd_mol is None:
        logger.warning(
            f"Could not sanitize ligand {res_name} in chain {chain} at residue number {resnum}. Skipping its sanitization..."
        )
        rd_mol = AllChem.MolFromPDBBlock(pdb_string, sanitize=False)
    elif rd_mol is None:
        raise ValueError(
            f"Could not convert ligand {res_name} in chain {chain} at residue number {resnum} to RDKit molecule."
        )

    if sub_smiles_provided and template is not None:
        # Ensure the input ligand perfectly matches the template ligand
        if rd_mol.GetNumAtoms() != template.GetNumAtoms():
            template = None
            logger.warning(
                f"Number of atoms in both molecules is different ({rd_mol.GetNumAtoms()} vs. {template.GetNumAtoms()}). Cannot assign bond orders from template."
            )

    try:
        if template is not None:
            new_mol = AllChem.AssignBondOrdersFromTemplate(template, rd_mol)
        else:
            new_mol = rd_mol
    except ValueError:
        new_mol = rd_mol

    return new_mol


def write_sdf(new_mol: Chem.Mol, pdb_name: str):
    """Write an RDKit molecule to an SD file.

    :param new_mol: RDKit molecule
    :param pdb_name: name of the output file
    """
    writer = Chem.SDWriter(pdb_name)
    writer.write(new_mol)
    logger.info(f"Wrote {pdb_name}")


def extract_protein_and_ligands_with_prody(
    input_pdb_file: str,
    protein_output_pdb_file: Optional[str],
    ligands_output_sdf_file: Optional[str],
    sanitize: bool = True,
    add_element_types: bool = False,
    generify_resnames: bool = False,
    clear_ligand_segnames: bool = False,
    write_output_files: bool = True,
    load_hetatms_as_ligands: bool = False,
    ligand_smiles: Optional[str] = None,
    ligand_expo_mapping: Optional[Dict[str, Any]] = None,
    permute_ligand_smiles: bool = False,
) -> Optional[Chem.Mol]:
    """Using ProDy, extract protein atoms and ligand molecules from a PDB file
    and write them to separate files.

    :param input_pdb_file: The input PDB file.
    :param protein_output_pdb_file: The output PDB file for the protein
        atoms.
    :param ligands_output_sdf_file: The output SDF file for the ligand
        molecules.
    :param sanitize: Whether to sanitize the ligand molecules.
    :param add_element_types: Whether to add element types to the
        protein atoms.
    :param generify_resnames: Whether to generify the residue names of
        the ligand molecules (e.g., for Boltz).
    :param clear_ligand_segnames: Whether to clear the segment names of
        the ligand atoms (e.g., for Boltz).
    :param write_output_files: Whether to write the output files.
    :param load_hetatms_as_ligands: Whether to load HETATM records as
        ligands if no ligands are initially found.
    :param ligand_smiles: The SMILES string of the ligand molecule.
    :param ligand_expo_mapping: The Ligand Expo mapping.
    :param permute_ligand_smiles: Whether to permute the ligand SMILES
        string's fragment components if necessary.
    :return: The combined final ligand molecule(s) as an RDKit molecule.
    """
    protein, ligand = get_pdb_components_with_prody(
        input_pdb_file, load_hetatms_as_ligands=load_hetatms_as_ligands
    )

    if ligand is None:
        logger.info(f"No ligand found in {input_pdb_file}. Returning None.")
        return None

    if clear_ligand_segnames:
        ligand.setSegnames("")

    if write_output_files:
        assert protein_output_pdb_file is not None, "Protein output PDB file must be provided."
        write_pdb_with_prody(protein, protein_output_pdb_file, add_element_types=add_element_types)

    ligand_resnames = ligand.getResnames()
    ligand_chids = ligand.getChids()
    ligand_resnums = ligand.getResnums()

    seen = set()
    resname_chain_resnum_list = []
    for resname, chid, resnum in zip(ligand_resnames, ligand_chids, ligand_resnums):
        if (resname, chid, resnum) not in seen:
            seen.add((resname, chid, resnum))
            resname_chain_resnum_list.append((resname, chid, resnum))

    new_mol = None
    new_mol_list = []
    ligand_smiles_component_indices_seen = set()
    ligand_smiles_components = ligand_smiles.split(".") if ligand_smiles is not None else None
    for i, resname_chain_resnum in enumerate(resname_chain_resnum_list, start=1):
        resname, chain, resnum = resname_chain_resnum
        sub_smiles = (
            ligand_smiles_components[i - 1]
            if ligand_smiles_components is not None
            and len(ligand_smiles_components) == len(resname_chain_resnum_list)
            else None
        )

        if permute_ligand_smiles and sub_smiles is not None:
            # E.g., for RFAA with the DockGen dataset, find the matching template SMILES string
            sub_mol = ligand.select(f"resname {resname} and chain {chain} and resnum {resnum}")
            template = AllChem.MolFromSmiles(sub_smiles)

            if len(sub_mol) != len(template.GetAtoms()):
                sub_smiles = None
                for comp_idx, comp in enumerate(ligand_smiles_components):
                    if (
                        len(sub_mol) == AllChem.MolFromSmiles(comp).GetNumAtoms()
                        and comp_idx not in ligand_smiles_component_indices_seen
                    ):
                        sub_smiles = comp
                        ligand_smiles_component_indices_seen.add(comp_idx)
                        break

        new_mol = process_ligand_with_prody(
            ligand,
            resname,
            chain,
            resnum,
            sanitize=sanitize,
            generify_resnames=generify_resnames,
            sub_smiles=sub_smiles,
            ligand_expo_mapping=ligand_expo_mapping,
        )
        if new_mol is not None:
            new_mol_list.append(new_mol)

            if write_output_files:
                assert (
                    ligands_output_sdf_file is not None
                ), "Ligands output SDF file must be provided."
                write_sdf(
                    new_mol,
                    os.path.join(
                        os.path.dirname(ligands_output_sdf_file),
                        f"{Path(ligands_output_sdf_file).stem}_{resname}_{i}.sdf",
                    ),
                )

    if len(new_mol_list):
        new_mol = combine_molecules(new_mol_list)

        if write_output_files:
            assert ligands_output_sdf_file is not None, "Ligands output SDF file must be provided."
            write_sdf(new_mol, ligands_output_sdf_file)

    return new_mol


@beartype
def create_sdf_file_from_smiles(smiles: str, output_sdf_file: str) -> str:
    """Create an SDF file from a SMILES string.

    :param smiles: SMILES string of the molecule.
    :param output_sdf_file: Path to the output SDF file.
    :return: Path to the output SDF file.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    writer = Chem.SDWriter(output_sdf_file)
    writer.write(mol)
    return output_sdf_file


@beartype
def count_num_residues_in_pdb_file(pdb_filepath: str) -> int:
    """Count the number of Ca atoms (i.e., residues) in a PDB file.

    :param pdb_filepath: Path to PDB file.
    :return: Number of Ca atoms (i.e., residues) in the PDB file.
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_filepath)
    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    count += 1
    return count


@beartype
def count_pdb_inter_residue_clashes(pdb_filepath: str, clash_cutoff: float = 0.63) -> int:
    """
    Count the number of inter-residue clashes in a protein PDB file.
    From: https://www.blopig.com/blog/2023/05/checking-your-pdb-file-for-clashing-atoms/

    :param pdb_filepath: Path to the PDB file.
    :param clash_cutoff: The cutoff for what is considered a clash.
    :return: The number of inter-residue clashes in the structure.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_filepath)

    # Atomic radii for various atom types.
    # You can comment out the ones you don't care about or add new ones
    atom_radii = {
        #    "H": 1.20,  # Who cares about hydrogen??
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "F": 1.47,
        "P": 1.80,
        "CL": 1.75,
        "MG": 1.73,
    }

    # Set what we count as a clash for each pair of atoms
    clash_cutoffs = {
        i + "_" + j: (clash_cutoff * (atom_radii[i] + atom_radii[j]))
        for i in atom_radii
        for j in atom_radii
    }

    # Extract atoms for which we have a radii
    atoms = [x for x in structure.get_atoms() if x.element in atom_radii]
    coords = np.array([a.coord for a in atoms], dtype="d")

    # Build a KDTree (speedy!!!)
    kdt = PDB.kdtrees.KDTree(coords)

    # Initialize a list to hold clashes
    clashes = []

    # Iterate through all atoms
    for atom_1 in atoms:
        # Find atoms that could be clashing
        kdt_search = kdt.search(np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values()))

        # Get index and distance of potential clashes
        potential_clash = [(a.index, a.radius) for a in kdt_search]

        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]

            # Exclude clashes from atoms in the same residue
            if atom_1.parent.id == atom_2.parent.id:
                continue

            # Exclude clashes from peptide bonds
            elif (atom_2.name == "C" and atom_1.name == "N") or (
                atom_2.name == "N" and atom_1.name == "C"
            ):
                continue

            # Exclude clashes from disulphide bridges
            elif (atom_2.name == "SG" and atom_1.name == "SG") and atom_distance > 1.88:
                continue

            if atom_distance < clash_cutoffs[atom_2.element + "_" + atom_1.element]:
                clashes.append((atom_1, atom_2))

    return len(clashes) // 2


@beartype
def parse_fasta(
    file_path: str,
    only_mols: List[Literal["protein", "na"]] | None = None,
    collate_by_pdb_id: bool = False,
) -> Dict[str, str]:
    """Parses a FASTA file into a dictionary and optionally filters by molecule
    type.

    :param file_path: Path to the .txt FASTA file.
    :param only_mols: List of molecule types to filter (e.g.,
        ['protein', 'na']).
    :param collate_by_pdb_id: Whether to group sequences by PDB ID.
    :return: A dictionary where keys are sequence IDs and values are
        tuples (description, sequence).
    """
    fasta_dict = {}
    current_id = None
    current_desc = None
    current_seq = []

    # Read the file line by line
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Save the previous sequence if applicable
                if current_id:
                    fasta_dict[current_id] = (current_desc, "".join(current_seq))

                # Parse the new header
                header_parts = line[1:].split(maxsplit=1)
                current_id = header_parts[0]
                current_desc = header_parts[1] if len(header_parts) > 1 else ""
                current_seq = []
            else:
                # Append sequence lines
                current_seq.append(line)

        # Save the last sequence
        if current_id:
            fasta_dict[current_id] = (current_desc, "".join(current_seq))

    # Filter by molecule type if only_mols is provided
    if only_mols:
        only_mols_set = {mol.lower() for mol in only_mols}
        fasta_dict = {
            seq_id: (desc, seq)
            for seq_id, (desc, seq) in fasta_dict.items()
            if any(f"mol:{mol}" in desc.lower() for mol in only_mols_set)
        }

    # Group sequences by PDB ID as requested
    if collate_by_pdb_id:
        collated_fasta_dict = defaultdict(list)
        for seq_id, (desc, seq) in fasta_dict.items():
            pdb_id, chain_id = seq_id.split("_")
            collated_fasta_dict[pdb_id].append((chain_id, desc, seq))
        fasta_dict = dict(collated_fasta_dict)

    return fasta_dict
