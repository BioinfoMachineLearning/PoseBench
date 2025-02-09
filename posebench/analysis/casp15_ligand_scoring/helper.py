"""General helper functions.

From: https://git.scicore.unibas.ch/schwede/casp15_ligand
"""

import contextlib
import logging
import math
import os
import tempfile

import ost

logger = logging.getLogger("helper." + __name__)


class OSTLogSink(ost.LogSink):
    """An OST log sink that redirects all OST log messages to the CAMEO
    logger."""

    def __init__(self):
        ost.LogSink.__init__(self)
        self._logger = logging.getLogger("helper." + __name__)

    def LogMessage(self, message, severity):
        # map ost severity levels to logging
        levels = [
            logging.ERROR,
            logging.WARNING,
            logging.INFO,
            logging.INFO,
            logging.DEBUG,
            logging.DEBUG,
            logging.DEBUG,
        ]
        self._logger.log(levels[severity], message.rstrip())


@contextlib.contextmanager
def capture_ost_logs():
    """Redirect OST logs to the Python logger.

    This context manager allows OST logs to be handled by the Python logger
    used in CAMEO instead of being printed to the standard error stream.
    This is particularly useful in combination with :func:`tmp_file_handler`.
    """
    ost.PushLogSink(OSTLogSink())
    yield
    ost.PopLogSink()


def set_verbosity(level):
    """Set verbosity for the logger.

    Useful in actions where we can set the verbosity from the script argument.

    :param level: Level name to set eg. DEBUG
    :type level: :class:`str`
    """
    # Map of logging levels to OST numeric levels.
    OST_LOGGING_LEVEL_MAP = {
        "DEBUG": ost.LogLevel.Debug,
        "INFO": ost.LogLevel.Info,
        "WARN": ost.LogLevel.Warning,
        "ERROR": ost.LogLevel.Error,
        "CRITICAL": -1,
    }

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.debug("Setting %s logger verbosity to %s", logger.name, level)
        logger.setLevel(level)

    ost.PushVerbosityLevel(OST_LOGGING_LEVEL_MAP[level])


# def squared_distance(coordsA, coordsB):
#     """Find the squared distance between two 3-tuples.

#     Sourced from https://gist.github.com/baoilleach/974477
#     """
#     sqrdist = sum( (a-b)**2 for a, b in zip(coordsA, coordsB) )
#     return sqrdist


# def rmsd(allcoordsA, allcoordsB):
#     """Find the RMSD between two lists of 3-tuples.

#     Sourced from https://gist.github.com/baoilleach/974477
#     """
#     deviation = sum(squared_distance(atomA, atomB) for
#                     (atomA, atomB) in zip(allcoordsA, allcoordsB))
#     return math.sqrt(deviation / float(len(allcoordsA)))


@contextlib.contextmanager
def temporary_file(yield_name=False, **kwargs):
    """Context manager to create temporary file and close and delete it after
    use.

    :param yield_name: whether to yield the file object (False) or the file
      name (True).
    :type yield_name: :class:`bool`

    Keyword arguments:
        delete (bool): do not delete the file upon closing. The file will
        still be deleted upon exit of the context manager, though.
        Any other argument to :class:`~tempfile.NamedTemporaryFile`.

    Sometimes you want to close the file so that an other process can safely
    read or write to it, without having it deleted. In such a case, you can
    use the following pattern, or use the :func:`temporary_file_name` manager
    instead:

    .. code:: python

        with temporary_file(yield_name=False, delete=False) as tf:
            tf.close()  # doesn't delete
            ... do things with tf
        # tf will be deleted once upon exiting the context
    """
    tf = tempfile.NamedTemporaryFile(**kwargs)
    try:
        logger.debug("Making temporary file %s" % tf.name)
        if yield_name:
            yield tf.name
        else:
            yield tf
    finally:
        logger.debug("Closing temporary file %s" % tf.name)
        tf.close()
        if not kwargs.get("delete", True):
            logger.debug("Deleting temporary file %s" % tf.name)
            try:
                os.remove(tf.name)
            except Exception as e:
                logger.debug(f"Couldn't remove file {tf.name}: {str(e)}")


def add_ligand_to_entity(entity, ligand, resnum=None, compound_id=None):
    """Add the given ligand to the OST entity. Modifies the entity in place
    - use entity.Copy() beforehand if you want to keep the original entity clean.

    :param entity: the entity to be modified.
    :param ligand: the entity containing the ligand as a single residue.
    :type ligand: :class:`ost.mol.EntityHandle` or :class:`ost.mol.EntityHandle`
    :type resnum: :class:`~ost.mol.ResNum`
    :type resnum: :class:`str`
    :rtype: None
    """
    ed = entity.EditXCS()
    # Get or create the ligand chain
    ligand_chain = entity.FindChain("_")
    if not ligand_chain.IsValid():
        ligand_chain = ed.InsertChain("_")
        # Get residue number
        if resnum is None:
            resnum = ost.mol.ResNum(1)
    else:
        resnum = ligand_chain.residues[-1].number + 1
    new_ligand_res = ed.AppendResidue(ligand_chain, ligand.residues[0], deep=True)
    ed.SetResidueNumber(new_ligand_res, resnum)
    ed.RenameResidue(new_ligand_res, compound_id)
    ed.UpdateICS()


def flat_mapping(mdl_residues, ref_residues):
    """LDDT requires a flat mapping with mdl_ch as key and ref_ch as value.

    This function takes the residues in the model and reference (must
    have the same length) and generates this map.
    """
    flat_mapping = dict()
    for mdl_residue, ref_residue in zip(mdl_residues, ref_residues):
        mdl_cname = mdl_residue.chain.name
        if mdl_cname not in flat_mapping:
            ref_cname = ref_residue.chain.name
            flat_mapping[mdl_cname] = ref_cname
    return flat_mapping


class cached_property:
    """A read-only property that is cached - code from CAMEO.
    In python 3.8 this is available directly in functools.

    Decorator/descriptor that converts a method with a single self argument
    into a property cached on the instance.
    """

    def __init__(self, method):
        self.method = method
        self.__doc__ = getattr(method, "__doc__")
        self.name = method.__name__

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.method(instance)
        return res
