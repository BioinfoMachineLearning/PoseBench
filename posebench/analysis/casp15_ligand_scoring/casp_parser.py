"""Parsing of CASP predictions.

From: https://git.scicore.unibas.ch/schwede/casp15_ligand
"""

import functools
import logging
import os
import re
import sys

# Custom modules imports
import networkx
import ost
import ost.conop
import ost.io
from openbabel import openbabel

# Local imports
from posebench.analysis.casp15_ligand_scoring import helper, openbabel_helper

try:
    cached_property = functools.cached_property
except AttributeError:
    cached_property = helper.cached_property

logger = logging.getLogger("casp_parser." + __name__)


# This is needed if the script is run with `python` instead of `ost`.
def _InitRuleBasedProcessor():
    compound_lib_path = os.path.join(ost.GetSharedDataPath(), "compounds.chemlib")
    if os.path.exists(compound_lib_path):
        compound_lib = ost.conop.CompoundLib.Load(compound_lib_path)
        ost.conop.SetDefaultLib(compound_lib)


_InitRuleBasedProcessor()


class TargetSmiles:
    """A target ligand representation from only the SMILES."""

    def __init__(self, ligand_id, compound_id, smiles, relevant, target):
        self.ligand_id = ligand_id
        self.compound_id = compound_id
        self.smiles = smiles
        self.target = target
        if relevant.lower() == "yes":
            self.relevant = True
        elif relevant.lower() == "no":
            self.relevant = False
        else:
            raise ValueError("Invalid value for relevant: %s" % relevant)

    @cached_property
    def smiles_obmol(self):
        try:
            return openbabel_helper.read_ligands_from_string(self.smiles, ["smi"])[0]
        except openbabel_helper.OpenBabelReadError as err:
            logger.error("Can't read SMILES string '%s'", self.smiles)
            raise


class TargetLigand(TargetSmiles):
    """A target ligand representation including the structure."""

    def __init__(
        self,
        ligand_id,
        compound_id,
        smiles,
        relevant,
        chain_name,
        residue_number,
        substructure_match,
        target,
    ):
        super().__init__(ligand_id, compound_id, smiles, relevant, target)
        self.chain_name = chain_name
        self.residue_number = int(residue_number)
        if substructure_match.lower() == "yes":
            self.substructure_match = True
        elif substructure_match.lower() == "no":
            self.substructure_match = False
        else:
            raise ValueError("Invalid value for substructure_match: %s" % substructure_match)

    def __repr__(self):
        return "TargetLigand<{} {}:{}:{}>".format(
            self.ligand_id,
            self.compound_id,
            self.chain_name,
            self.residue_number,
        )

    @cached_property
    def struct_ost_ent(self):
        """Return an  ost EntityView of the ligand."""
        selection = "cname=%s and rnum=%d" % (self.chain_name, self.residue_number)
        ent_view = self.target.struct_ost_ent.Select(selection)
        assert len(ent_view.residues) == 1
        return ent_view

    @cached_property
    def struct_ost_residue(self):
        """Return an ost ResidueView of the ligand."""
        return self.struct_ost_ent.residues[0]

    @cached_property
    def custom_compound(self):
        """Return the residue as an ost CustomCompound."""
        return ost.mol.alg.lddt.CustomCompound.FromResidue(self.struct_ost_residue)

    @cached_property
    def custom_compound_dict(self):
        """Return the residue sas a dictionary of CustomCompounds for lDDT."""
        return {self.struct_ost_residue.name: self.custom_compound}

    @cached_property
    def struct_obmol_from_ost(self):
        """Return the structure ligand as an openbabel molecule, passing
        through OST."""
        pdb_str = ost.io.EntityToPDBStr(self.struct_ost_ent)
        return openbabel_helper.read_ligands_from_string(pdb_str, ["pdb"])[0]

    @cached_property
    def struct_spyrmsd_mol(self):
        """Return the structure ligand as a syprmsd.molecule.Molecule, passing
        through OST, and stripped of hydrogens.

        :rtype: :class:`spyrmsd.molecule.Molecule`
        """
        import spyrmsd.io

        pdb_str = ost.io.EntityToPDBStr(self.struct_ost_ent)
        with helper.temporary_file(suffix=".sdf", delete=False) as tmpfile:
            tmpfile.close()
            ost.io.SaveEntity(self.struct_ost_ent, tmpfile.name, format="sdf")
            mol = spyrmsd.io.to_molecule(spyrmsd.io.load(tmpfile.name))
            mol.strip()
            return mol

    @cached_property
    def struct_obmol(self):
        """Return he structure ligand as an openbabel molecule, without OST."""
        trg_obmol = openbabel_helper.read_ligands_from_file(self.target.target_pdb_file, ["pdb"])[
            0
        ]
        for residue_id in range(trg_obmol.NumResidues()):
            residue = trg_obmol.GetResidue(residue_id)
            if residue.GetChain() == self.chain_name and residue.GetNum() == self.residue_number:
                # Optionally check residue name?
                if residue.GetName() != self.compound_id:
                    raise ValueError("Invalid residue name %s" % residue.GetName())
                else:
                    # TODO: find out how to return a Mol (we have a Residue)
                    raise NotImplementedError()
                    # Here's a try...
                    mol = openbabel.OBMol()
                    mol.AddResidue(residue)
                    atom_iter = residue.BeginAtoms()
                    next_atom = residue.NextAtom(atom_iter)
                    while next_atom:
                        mol.AddAtom(next_atom)
                        next_atom = residue.NextAtom(atom_iter)
                    # # How to transfer bonds?
                    # import ipdb

                    # ipdb.set_trace()
                    # for bond in residue.GetBonds():
                    #     mol.AddBond(bond)

    @cached_property
    def formula(self):
        """Get the chemical formula of this ligand."""
        return self.struct_obmol_from_ost.GetFormula()

    @cached_property
    def simplified_formula(self):
        """Get a simplified chemical formula of this ligand, with hydrogens and
        charge removed."""
        formula = self.formula
        formula = re.sub(r"H\d*", "", formula)
        formula = re.sub(r"(\+|-)+$", "", formula)
        return formula

    @cached_property
    def graph(self):
        """Return a NetworkX graph representation of the ligand."""
        nxg = networkx.Graph()
        nxg.add_nodes_from(
            [a.name for a in self.struct_ost_ent.atoms],
            element=[a.element for a in self.struct_ost_ent.atoms],
        )
        nxg.add_edges_from(
            [(b.first.name, b.second.name) for b in self.struct_ost_ent.GetBondList()]
        )
        return nxg

    @cached_property
    def is_connected(self):
        """Check if the ligand is a connected graph."""
        return networkx.is_connected(self.graph)


class Target:
    def __init__(self, target_id, targets_path):
        self.target_id = target_id
        self.targets_path = targets_path

    def __repr__(self):
        return "Target<%s>" % self.target_id

    @cached_property
    def target_pdb_file(self):
        return os.path.join(self.targets_path, "%s_lig.pdb" % self.target_id)

    @cached_property
    def target_seq_file(self):
        return os.path.join(self.targets_path, "%s.seq.txt" % self.target_id)

    @cached_property
    def target_smiles_file(self):
        return os.path.join(self.targets_path, "%s.smiles.txt" % self.target_id)

    @cached_property
    def target_ligands_file(self):
        return os.path.join(self.targets_path, "%s.ligands.txt" % self.target_id)

    @cached_property
    def target_seq(self):
        """OST seq of the target."""
        return ost.io.LoadSequenceList(self.target_seq_file, format="fasta")

    @cached_property
    def target_smiles(self):
        ligands = []
        with open(self.target_smiles_file) as fd:
            titles = fd.readline().split()
            assert titles == ["ID", "Name", "SMILES", "Relevant"]
            for line in fd:
                if line.strip() != "":
                    ligands.append(TargetSmiles(*(line.split()), target=self))
        return ligands

    @cached_property
    def target_ligands(self):
        ligands = []
        with open(self.target_ligands_file) as fd:
            titles = fd.readline().split()
            assert titles == [
                "ID",
                "Name",
                "SMILES",
                "Relevant",
                "Chain",
                "Resnum",
                "SubstructureMatch",
            ]
            for line in fd:
                if line.strip() != "":
                    ligands.append(TargetLigand(*(line.split()), target=self))
        return ligands

    def validate(self):
        """Validate the target."""
        # Can we read the ligands file and is it not empty?
        assert self.target_ligands
        # Check the target sequence
        self.validate_entity_sequence(self.struct_ost_ent.Select("ishetatm=false"))

        # Check ligand connectivity
        # For now this only check that they are connected, not whether the
        # connectivity is correct.
        for ligand in self.target_ligands:
            if not ligand.is_connected:
                raise CaspValidationError("Disconnected graph for ligand %s" % ligand)

    def validate_entity_sequence(self, entity):
        """Validate that the entity matches one of the target sequences.

        Raises a CaspValidationError upon error.
        """
        for chain in entity.chains:
            # Validate chain name
            if not re.match(r"^[A-Za-z0-9_]$", chain.name):
                raise CaspValidationError("Invalid chain name: %s" % chain.name)

            for seq in self.target_seq:
                logger.debug("Trying match chain %s to sequence %s", chain.name, seq.name)
                for residue in chain.residues:
                    try:
                        if seq[residue.number.num - 1] != residue.one_letter_code:
                            if residue.one_letter_code == "?":
                                logger.debug("Unknown residue: %s. Ignoring", str(residue))
                            else:
                                logger.debug(
                                    "Chain mismatch: %s != %s",
                                    seq[residue.number.num - 1],
                                    residue,
                                )
                                break  # for residue in chain.residues
                    except IndexError as err:
                        if "Position is not covered in sequence" in str(err):
                            logger.debug(str(err))
                            break
                        else:
                            raise  # not sure what this index error would be - so just raise
                else:
                    logger.info("Aligned chain %s to %s", chain.name, seq.name)
                    break  # for seq in self.target_seq
            else:
                # We failed to align anything
                raise CaspValidationError(
                    "Chain %s couldn't be aligned to target sequence" % chain.name
                )

    @cached_property
    def struct_ost_ent(self):
        entity = ost.io.LoadPDB(self.target_pdb_file)
        # Remove Hydrogens
        entity_view = entity.Select("ele != H")
        return ost.mol.CreateEntityFromView(entity_view, False)

    def save_images(self, skip_large=200):
        """Save a PNG representation of the target.

        :param skip_large: skip saving images of ligands with more than
            this many atoms.
        """
        for ligand in self.target_smiles:
            if ligand.smiles_obmol.NumAtoms() <= skip_large:
                out_fn = "{}_{}_{}.svg".format(
                    self.target_id, ligand.ligand_id, ligand.compound_id
                )
                out_file = os.path.join(self.targets_path, out_fn)
                openbabel_helper.write_mol(ligand.smiles_obmol, out_file, "svg")
                logger.debug("Saved %s", out_file)

    def save_images(self, skip_large=200):
        """Save a PNG representation of the target.

        :param skip_large: skip saving images of ligands with more than
            this many atoms.
        """
        for ligand in self.target_ligands:
            # SMILES
            if ligand.smiles_obmol.NumAtoms() <= skip_large:
                out_fn = "{}_{}_{}_smiles.svg".format(
                    self.target_id,
                    ligand.ligand_id,
                    ligand.compound_id,
                )
                out_file = os.path.join(self.targets_path, out_fn)
                openbabel_helper.write_mol(ligand.smiles_obmol, out_file, "svg")
                logger.debug("Saved SMILES image %s", out_file)
                # Structure
                out_fn = "{}_{}_{}.svg".format(
                    self.target_id, ligand.ligand_id, ligand.compound_id
                )
                out_file = os.path.join(self.targets_path, out_fn)
                openbabel_helper.write_mol(ligand.struct_obmol_from_ost, out_file, "svg")
                logger.debug("Saved structure image %s", out_file)


class ModelLigand:
    """A ligand in a model is define by the MDL data part and its ID."""

    def __init__(self, name, data, mol):
        self.name = name
        self.data = data
        self.mol = mol
        self.ligand_id, self.compound_id = name.split()

    def __repr__(self):
        return "ModelLigand<%s>" % (self.name)

    @cached_property
    def obmol(self):
        return self.mol

    @property
    def obmol_fresh(self):
        """A "fresh", not cached obMol object for operations with a side
        effect, such as saving the molecule to SVG which flattens the
        coordinates."""
        return openbabel_helper.read_ligands_from_string(self.data, ["sdf"])[0]

    @cached_property
    def ost_ent(self):
        with helper.temporary_file() as tmpfile:
            sdf_data = openbabel_helper.mol_to_string(self.obmol, format="sdf")
            tmpfile.write(sdf_data.encode())
            tmpfile.flush()
            entity = ost.io.LoadEntity(tmpfile.name, format="sdf")
            # Remove Hydrogens
            entity_view = entity.Select("ele != H")
            entity = ost.mol.CreateEntityFromView(entity_view, False)
            # Rename chain
            ed = entity.EditXCS()
            ed.RenameChain(entity.chains[0], self.chain_name)
            ed.UpdateICS()
            return entity

    @cached_property
    def chain_name(self):
        return "%03d_%s" % (int(self.ligand_id), self.compound_id)

    @cached_property
    def resnum_tuple(self):
        return self.ost_residue.number.num, self.ost_residue.number.inscode

    @cached_property
    def ost_residue(self):
        """Return an ost ResidueView of the ligand."""
        return self.ost_ent.residues[0]

    @cached_property
    def spyrmsd_mol(self):
        """Return the ligand as a spyrmsd Molecule, stripped of hydrogens.

        :rtype: :class:`spyrmsd.molecule.Molecule`
        """
        import spyrmsd.io

        with helper.temporary_file(suffix=".sdf") as tmpfile:
            tmpfile.write(self.data.encode())
            tmpfile.flush()
            mol = spyrmsd.io.to_molecule(spyrmsd.io.load(tmpfile.name))
            mol.strip()
            return mol

    @cached_property
    def formula(self):
        """Get the chemical formula of this ligand."""
        return self.obmol.GetFormula()

    @cached_property
    def simplified_formula(self):
        """Get a simplified chemical formula of this ligand, with hydrogens and
        charge removed."""
        formula = self.formula
        formula = re.sub(r"H\d*", "", formula)
        formula = re.sub(r"(\+|-)+$", "", formula)
        return formula

    @cached_property
    def graph(self):
        """Return a NetworkX graph representation of the ligand."""
        nxg = networkx.Graph()
        nxg.add_nodes_from(
            [a.name for a in self.ost_ent.atoms], element=[a.element for a in self.ost_ent.atoms]
        )
        nxg.add_edges_from([(b.first.name, b.second.name) for b in self.ost_ent.GetBondList()])
        return nxg

    @cached_property
    def is_connected(self):
        """Check if the ligand is a connected graph."""
        return networkx.is_connected(self.graph)


class CaspParserError(ValueError):
    """Raised when the parsing of the CASP data failed.

    This allows catching formatting errors in submissions.
    """


class CaspValidationError(AssertionError):
    """Raised when the validation of the CASP data failed.

    This allows catching more subtle errors in submissions.
    """


class CaspParser:
    """Parse a potential CASP prediction."""

    _IGNORE_REC = (
        "PFRMAT",
        "TARGET",
        "AUTHOR",
        "REMARK",
        "METHOD",
        "PARENT",
        "SCORE",
        "QSCORE",
        "QMODE",
    )
    _PROCESS_REC = (
        "MODEL",
        "END",
        "ENDMDL",
        "LIGAND",
        "POSE",
    )
    _ACCUMULATE_REC = (
        "HEADER",
        "OBSLTE",
        "TITLE",
        "SPLT",
        "CAVEAT",
        "COMPND",
        "SOURCE",
        "KEYWDS",
        "EXPDTA",
        "NUMMDL",
        "MDLTYP",
        "AUTHOR",
        "REVDAT",
        "SPRSDE",
        "JRNL",
        "REMARKS",
        "DBREF",
        "DBREF1",
        "DBREF2",
        "SEQADV",
        "SEQRES",
        "MODRES",
        "HET",
        "FORMUL",
        "HETNAM",
        "HETSYN",
        "HELIX",
        "SHEET",
        "SSBOND",
        "LINK",
        "CISPEP",
        "SITE",
        "ATOM",
        "ANISOU",
        "TER",
        "HETATM",
        "CONECT",
        "MASTER",
    )
    _LIGAND_END_REC = ("M",)
    _LIGAND_FORMAT = "mdl"

    def __init__(self, data, fault_tolerant=False, allow_sdf=True):
        """Instantiate a CaspParser.

        :param data: the string of the part/file to process.
        :type data: :class:`str`
        :param fault_tolerant: whether to parse with a fault_tolerant
          IOProfile.
        :type fault_tolerant: :class:`bool`
        :param allow_sdf: whether to allow SDF submissions. SDF data can
          contain an extra property block and end with a line with four
          dollar signs (`$$$$`).
        :type allow_sdf: :class:`bool`
        """
        self.data = data
        # Main output data
        self.models = {}
        self.ligands = {}
        # Set some status bits
        self._line_num = -1
        self._model_num = None
        self._pose_num = None
        self._ligand_name = None
        self._ligand_count = 0
        # Keep track of whether we're looking at a ligand
        # Turned on with LIGAND, off with M END (end ligand)
        self._accumulating_ligand = False
        self._ligand_accumulator = ""
        self._model_accumulator = ""
        if fault_tolerant:
            self._profile = ost.io.profiles["SLOPPY"]
        else:
            self._profile = ost.io.profiles["STRICT"]

        # Consider switching to SDF format?
        if allow_sdf and "\n$$$$\n" in data:
            logger.info("Switching to SDF format")
            self._LIGAND_END_REC = ("$$$$",)
            self._LIGAND_FORMAT = "sdf"

    def validate(self, target):
        """Validate the data."""
        if len(self.models) == 0:
            raise CaspValidationError("No PDB models found")
        if len(self.ligands) == 0:
            raise CaspValidationError("No ligands found")
        if len(self.models) != len(self.ligands):
            raise CaspValidationError(
                "%d PDB MODELs != %d ligands MODELs" % (len(self.models), len(self.ligands))
            )

        # Look at the ligands
        for model_num, poses in self.ligands.items():
            delete_poses = []
            for pose_num, pose in poses.items():
                seen_lig_id = set()
                delete_ligs = []
                for ligand in pose:
                    logger.info("Ligand %s, pose %d, model %d", ligand.name, pose_num, model_num)
                    ligand_id, compound_id = ligand.name.split()
                    ligand_id = int(ligand_id)
                    # Check ligand ID
                    if len(target.target_smiles) < ligand_id:
                        raise CaspValidationError("Ligand id %s not in target" % ligand.name)
                    # Get ligand target
                    target_ligand = target.target_smiles[ligand_id - 1]
                    # check compound (only warning)
                    if compound_id != target_ligand.compound_id:
                        logger.warning(
                            "Compound %s doesn't match target compound id (%s)"
                            % (compound_id, target_ligand.compound_id)
                        )
                    # Check if we've seen the ligand ID already
                    if ligand_id in seen_lig_id:
                        raise CaspValidationError(
                            f"Already seen a ligand with id {ligand_id} ({ligand.name})"
                        )
                    else:
                        seen_lig_id.add(ligand_id)

                    # Check if the ligand is connected
                    if not ligand.is_connected:
                        # raise CaspValidationError("Disconnected graph for ligand %s" % ligand)
                        logger.error("Disconnected graph for ligand %s" % ligand)
                        delete_ligs.append(ligand)

                    trg_mol = target_ligand.smiles_obmol
                    mdl_mol = ligand.obmol

                    # Remove Hydrogens for comparison
                    if not mdl_mol.DeleteHydrogens():
                        raise RuntimeError("Failed to delete hydrogens from model molecule")
                    target_ligand.smiles_obmol.DeleteHydrogens()

                    # Check smiles
                    trg_smiles = openbabel_helper.get_canonical_smiles(trg_mol)
                    mdl_smiles = openbabel_helper.get_canonical_smiles(mdl_mol)
                    if trg_smiles != mdl_smiles:
                        logger.warning(f"SMILES {mdl_smiles} doesn't match target ({trg_smiles})")
                        trg_formula = trg_mol.GetFormula()
                        mdl_formula = mdl_mol.GetFormula()
                        # if mdl_formula.endswith("+"):
                        #    import ipdb; ipdb.set_trace()
                        # Formulas still have H - need to remove
                        trg_formula_fixed = re.sub(r"H\d*", "", trg_formula)
                        mdl_formula_fixed = re.sub(r"H\d*", "", mdl_formula)
                        trg_formula_fixed = re.sub(r"(\+|-)+$", "", trg_formula_fixed)
                        mdl_formula_fixed = re.sub(r"(\+|-)+$", "", mdl_formula_fixed)
                        if trg_formula_fixed != mdl_formula_fixed:
                            logger.warning(
                                "Ligand %s has the wrong formula (%s, expected %s)",
                                ligand.name,
                                mdl_formula,
                                trg_formula,
                            )
                        else:
                            logger.info("Formula matches target")
                # Remove disconnected ligands
                for delete_lig in delete_ligs:
                    pose.remove(delete_lig)
                    logger.info("Removed %s from pose %s", delete_lig, pose_num)
                if len(pose) == 0:
                    delete_poses.append(pose_num)
            for pose_num in delete_poses:
                logger.error("No more ligands in pose %s, removing", pose_num)
                del poses[pose_num]

        # Check chain against target sequences
        for model_id, model in self.models.items():
            target.validate_entity_sequence(model)

    def parse(self):
        """Parse the provided data."""
        part_str = self.data
        for i, line in enumerate(part_str.splitlines(), 1):
            self._line_num = i
            # logger.debug("Line %d: %s", self.line_num, line)
            self._process_line(line)

    def _process_line(self, line):
        try:
            # Split with the 1st space
            # Note: in PDB format we could write
            #     content = line[:6]
            #     record = line[7:]
            # but this is not PDB format
            record, content = line.split(" ", 1)
        except ValueError:
            # Bare record with no content
            record = line
            content = None
        # logger.debug(record)
        if record in self._IGNORE_REC:
            logger.debug("Ignoring %s", record)
            pass
        elif record in self._PROCESS_REC:
            logger.debug("Processing %s", record)
            getattr(self, "_process_" + record)(content, line)
        elif record in self._ACCUMULATE_REC:
            self._accumulate_model(line)
        elif record in self._LIGAND_END_REC and record == "M" and content.strip() == "END":
            logger.debug("M END of a ligand")
            self._process_end_ligand(line)
        elif record in self._LIGAND_END_REC and record == "$$$$":
            logger.debug("$$$$ (end) of a ligand")
            self._process_end_ligand(line)
        elif self._accumulating_ligand:
            self._accumulate_ligand(line)
        else:
            raise CaspParserError(f"Unknown record on line {str(self._line_num)}: {record}")

    def _process_MODEL(self, content, line):
        new_model_num = int(content.strip())
        if self._model_num != new_model_num:
            self._model_num = int(content.strip())
            logger.debug("Model number: %d", self._model_num)
            self._reset_model_accumulator()
            self._reset_ligand_accumulator()
        else:
            logger.debug("Ignoring repeated MODEL number %d", new_model_num)

    def _process_LIGAND(self, content, line):
        self._ligand_name = content.strip()
        logger.debug("Ligand: %s", self._ligand_name)
        self._reset_ligand_accumulator()
        self._accumulating_ligand = True

    def _process_POSE(self, content, line):
        self._pose_num = int(content.strip())
        logger.debug("Pose number: %d", self._pose_num)
        self._reset_ligand_accumulator()
        self._accumulating_ligand = True

    def _process_ENDMDL(self, content, line):
        """We reached the end of a model.

        Read in the model data that is saved in the model accumulator.
        """
        self._process_end_model(line)

    def _process_END(self, content, line):
        """We reached an END.

        This can be either the end of a model, or can be an extra END
        after ligands, which should have been processed already.
        """
        if self._model_accumulator != "":
            # We got a model!
            logging.debug("END with a model")
            self._process_end_model(line)
        elif self._ligand_accumulator != "":
            self._reset_ligand_accumulator()
        else:
            logging.debug("Normal END")

    def _process_end_model(self, line):
        # Add the last line before parsing
        self._accumulate_model(line)

        if self._model_num in self.models:
            msg = "Model %d already exists on line %d" % (self._model_num, self._line_num)
            raise CaspParserError(msg)
        else:
            logger.debug("Model %s", str(self._model_num))
            try:
                entity = ost.io.PDBStrToEntity(
                    self._model_accumulator, profile=self._profile, process=True
                )
                processor = ost.conop.RuleBasedProcessor(ost.conop.GetDefaultLib())
                processor.Process(entity)
                # Remove Hydrogens
                entity_view = entity.Select("ele != H")
                entity = ost.mol.CreateEntityFromView(entity_view, False)
            except Exception as exc:
                msg = "Error parsing model %d: %s" % (self._model_num, str(exc))
                raise CaspParserError(msg)
            else:
                # Fix blank chain names - common in CASP predictions
                if len(entity.chains) == 1 and entity.chains[0].name == " ":
                    editor = entity.EditXCS()
                    editor.RenameChain(entity.chains[0], "A")
                    editor.UpdateICS()
                    logger.warning("Blank chain name renamed to A")
                self.models[self._model_num] = entity
                self._model_accumulator = ""

    def _process_end_ligand(self, content):
        """We reached the end of the ligand part (M END or $$$$ depending on
        the format).

        Attempt to read it with OpenBabel.
        """
        # Add the last line before parsing
        self._accumulate_ligand(content)

        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat(self._LIGAND_FORMAT)
        mol = openbabel.OBMol()
        out = openbabel_helper.OutputGrabber()
        with out:
            success = obConversion.ReadString(mol, self._ligand_accumulator)
        if success and mol.NumAtoms() > 0:
            if self._model_num not in self.ligands:
                self.ligands[self._model_num] = {}
            if self._pose_num not in self.ligands[self._model_num]:
                self.ligands[self._model_num][self._pose_num] = []
            model_ligand = ModelLigand(
                name=self._ligand_name, data=self._ligand_accumulator, mol=mol
            )
            self.ligands[self._model_num][self._pose_num].append(model_ligand)
            self._ligand_accumulator = ""
        else:
            ligand_id = "ligand %s, pose %d" % (self._ligand_name, self._pose_num)
            logger.error("Cannot read %s with OpenBabel", ligand_id)
            if len(self._ligand_accumulator) > 1000:
                logger.error("Beginning of the data:")
                logger.error(self._ligand_accumulator[:200])
            else:
                logger.error(self._ligand_accumulator)
            if out.capturedtext:
                msg = f"OpenBabel error for {ligand_id}:\n{out.capturedtext}"
                logger.error(msg)
                raise CaspParserError(out.capturedtext)
            elif success and mol.NumAtoms() == 0:
                msg = "No atoms could be read with OpenBabel for %s" % ligand_id
                logger.error(msg)
                raise CaspParserError(msg)
            else:
                msg = "Unknown OpenBabel error for %s" % ligand_id
                logger.error(msg)
                raise CaspParserError()

    def _accumulate_model(self, line):
        self._model_accumulator += line + os.linesep

    def _accumulate_ligand(self, line):
        self._ligand_accumulator += line + os.linesep

    def _reset_ligand_accumulator(self):
        """This resets the ligand accumulator, processing the content again
        with self._accumulating_ligand = False if the accumulator is not empty.
        This ensures we process all the tags.

        Typically, this happens when there was content between the
        LIGAND and POSE tags, or if those tags were misspelled. It
        enables a cleaner error message than the "OpenBabel failed to
        read the counts line" that we would get otherwise.
        """
        # Skip if accumulator contains only blank
        if self._ligand_accumulator.strip() == "":
            self._ligand_accumulator = ""
        else:
            # Reprocess the old lines
            reprocess_lines = self._ligand_accumulator.splitlines()
            logger.debug("Reprocessing some lines: %s", reprocess_lines)
            self._ligand_accumulator = ""

            # Adjust line number
            old_line_num = self._line_num
            logger.debug("Current line number: %d", old_line_num)
            lines_new_start = self._line_num - len(reprocess_lines)
            logger.debug("New line number: %d", lines_new_start)
            # Process again without _accumulating_ligand
            self._accumulating_ligand = False
            for i, line in enumerate(reprocess_lines, lines_new_start):
                self._line_num = i
                self._process_line(line)

            # Restore the status
            self._line_num = old_line_num
            self._accumulating_ligand = True

    def _reset_model_accumulator(self):
        """This resets the model accumulator.

        Really it should be empty
        """
        # Skip if accumulator contains only blank
        if self._model_accumulator.strip() == "":
            self._model_accumulator = ""
        else:
            raise CaspParserError("Model accumulator not empty on line %s", self._line_num)

    def save_images(self, dest_prefix, skip_large=200):
        """Save a PNG representation of the ligands in the prediction.

        :param skip_large: skip saving images of ligands with more than
            this many atoms.
        """
        for model_num, poses in self.ligands.items():
            for pose_num, pose in poses.items():
                for ligand in pose:
                    if ligand.obmol.NumAtoms() <= skip_large:
                        ligand_num, compound_id = ligand.name.split()
                        out_suffix = "_{}_{}_{}_{}.svg".format(
                            model_num,
                            pose_num,
                            ligand_num,
                            compound_id,
                        )
                        out_path = dest_prefix + out_suffix
                        openbabel_helper.write_mol(ligand.obmol_fresh, out_path, "svg")
                        logger.debug("Saved %s for inspection", out_path)

    def save_pdb(self, dest_prefix):
        """Save a PNG representation of the ligands in the prediction.

        Ligands are save in the _ chain.
        """
        for model_num, poses in self.ligands.items():
            for pose_num, pose in poses.items():
                save_entity = self.models[model_num].Copy()
                for ligand in pose:
                    ligand_num, compound_id = ligand.name.split()
                    helper.add_ligand_to_entity(
                        save_entity, ligand.ost_ent, ost.mol.ResNum(int(ligand_num)), compound_id
                    )
                out_suffix = f"_{model_num}_{pose_num}.pdb"
                out_path = dest_prefix + out_suffix
                ost.io.SavePDB(save_entity, out_path)
                logger.debug("Saved %s for inspection", out_path)
