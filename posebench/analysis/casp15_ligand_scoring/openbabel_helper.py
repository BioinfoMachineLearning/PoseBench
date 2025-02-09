"""Module to handle OpenBabel.

From: https://git.scicore.unibas.ch/schwede/casp15_ligand
"""

import logging
import os
import sys

from openbabel import openbabel

logger = logging.getLogger("casp." + __name__)


class OpenBabelReadError(ValueError):
    """Raised when OpenBabel failed to read a ligand."""

    pass


def read_ligands_from_file(file, formats=["sdf", "mol2"]):
    """Read one or more ligands from file. Return them as a list.

    This function either returns a list with at least one molecule, or
    raises and OpenBabelReadError.
    """

    # Read in the file.
    # We could use obConversion.ReadFile but that would prevent us from raising
    # clear IOErrors
    with open(file) as fd:
        ligand_string = fd.read()
    return read_ligands_from_string(ligand_string, formats)


def read_ligands_from_string(string, formats=["sdf", "mol2"]):
    """Read one or more ligands from string. Return them as a list.

    This function either returns a list with at least one molecule, or
    raises and OpenBabelReadError.
    """
    ligands = []
    obConversion = openbabel.OBConversion()
    grab_out = OutputGrabber()
    for format in formats:
        logger.debug("Trying to read file as %s", format)
        obConversion.SetInFormat(format)
        mol = openbabel.OBMol()
        with grab_out:
            success = obConversion.ReadString(mol, string)
        if grab_out.capturedtext:
            logger.debug("OpenBabel output:")
            logger.debug(grab_out.capturedtext)
        if success and mol.NumAtoms() > 0:
            while success:
                logger.debug("Read ligand %s with %d atoms" % (mol.GetTitle(), mol.NumAtoms()))
                ligands.append(mol)
                mol = openbabel.OBMol()
                logger.debug("Trying to read more ligands from file")
                with grab_out:
                    success = obConversion.Read(mol)
                if grab_out.capturedtext:
                    logger.debug("OpenBabel output:")
                    logger.debug(grab_out.capturedtext)
            return ligands
        else:
            logger.debug("Can't read ligand as %s", format)
    raise OpenBabelReadError("No ligand read")


def write_mol(mol, dest, format):
    """Write the OpenBabel molecule `mol` to `dest` in `format`."""
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat(format)
    obConversion.WriteFile(mol, dest)


def mol_to_string(mol, format):
    """Return a string representation of the OpenBabel molecule `mol` in
    `format`."""
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat(format)
    return obConversion.WriteString(mol)


def get_canonical_smiles(mol):
    """Get the SMILES (canonical) of an openbabel molecule."""
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat("can")
    return obConversion.WriteString(mol).split("\t")[0]


class OutputGrabber:
    """Class used to grab standard error from OpenBabel into a string.

    This code was shamelessly copied from Stack Overflow:
    https://stackoverflow.com/a/29834357/333599
    """

    escape_char = "\b"

    def __init__(self):
        self.origstream = sys.stderr
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """Start capturing the stream data."""
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)

    def stop(self):
        """Stop capturing the stream data and save the text in
        `capturedtext`."""
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """Read the stream data (one byte at a time) and save the text in
        `capturedtext`."""
        while True:
            char = os.read(self.pipe_out, 1).decode(self.origstream.encoding)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char
