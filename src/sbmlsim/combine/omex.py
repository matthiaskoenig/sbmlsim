"""
COMBINE Archive helper functions and classes based on libCOMBINE.

Here common operations with COMBINE archives are implemented, like
extracting archives, creating archives from entries or directories,
adding metadata, listing content of archives.

When working with COMBINE archives these wrapper functions should be used.

"""
# FIXME: handle the adding of metadata

import os
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import List, Iterator, Iterable

import libcombine


import pprint

class Creator:
    """Helper class to store the creator information.

    FIXME: reuse sbmlutils creators
    """

    def __init__(self, given_name: str, family_name: str,
                 organization: str, email: str):
        self.given_name = given_name
        self.family_name = family_name
        self.organization = organization
        self.email = email


class Entry:
    """ Helper class to store content to create an OmexEntry."""

    def __init__(
        self,
        location: str,
        format: str = None,
        format_key: str = None,
        master: bool = False,
        description: str = None,
        creators: Iterator[Creator] = None,
    ):
        """Create entry from information.

        If format and formatKey are provided the format is used.

        :param location: location of the entry
        :param format: full format string
        :param format_key: short formatKey string
        :param master: master attribute
        :param description: description
        :param creators: iterator over Creator objects
        """
        if (format_key is None) and (format is None):
            raise ValueError(
                "Either 'formatKey' or 'format' must be specified for Entry."
            )
        if format is None:
            format = libcombine.KnownFormats.lookupFormat(formatKey=format_key)

        self.format: str = format
        self.location: str = location
        self.master: bool = master
        self.description: str = description
        self.creators: Iterator[Creator] = creators

    def __str__(self) -> str:
        """String representation of entry."""
        if self.master:
            return f"<*master* Entry {self.location} | {self.format}>"
        else:
            return f"<Entry {self.location} | {self.format}>"


class Omex:
    """Combine archive class"""

    MANIFEST_PATTERN = "manifest.xml"
    METADATA_PATTERN = "metadata.*"

    def __init__(self, omex_path: Path, working_dir: Path):
        """Create combine archive."""
        self.omex_path: Path = omex_path
        self.working_dir: Path

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        omex_path: Path,
        creators=None,
    ):
        """Creates a COMBINE archive from a given folder.

        The file types are inferred,
        in case of existing manifest or metadata information this should be reused.

        For all SED-ML files in the directory the master attribute is set to True.

        :param directory: Directory to compress
        :param omex_path: Output path for omex directory
        :param creators: List of creators
        :return:
        """

        manifest_path: Path = directory / cls.MANIFEST_PATTERN

        if manifest_path.exists():
            warnings.warn(
                f"Manifest file exists in directory, but not used in COMBINE "
                f"archive creation: {manifest_path}"
            )
            # FIXME: reuse existing manifest

        # add the base entry
        entries = [
            Entry(
                location=".",
                format="http://identifiers.org/combine.specifications/omex",
                master=False,
            )
        ]

        # iterate over all locations & guess format
        for root, dirs, files in os.walk(str(directory)):
            for file in files:
                file_path = os.path.join(root, file)
                location = os.path.relpath(file_path, directory)
                # guess the format
                format = libcombine.KnownFormats.guessFormat(file_path)
                master = False
                if libcombine.KnownFormats.isFormat(formatKey="sed-ml", format=format):
                    master = True

                entries.append(
                    Entry(
                        location=location, format=format, master=master, creators=creators
                    )
                )

        # create additional metadata if available

        # write all the entries
        return cls.from_entries(entries=entries, omex_path=omex_path, working_dir=directory)

    @classmethod
    def from_entries(cls, omex_path: str, entries: Iterable[Entry], workingDir):
        """Creates combine archive from given entries.

        Overwrites existing combine archive at omexPath.

        :param entries:
        :param workingDir:
        :return:
        """
        _addEntriesToArchive(omex_path, entries, workingDir=workingDir, add_entries=False)
        print("*" * 80)
        print("Archive created:\n\t", omex_path)
        print("*" * 80)


    @staticmetho
    def _addEntriesToArchive(entries, workingDir, add_entries: bool):
        """

        :param archive:
        :param entries:
        :param workingDir:
        :return:
        """
        omexPath = os.path.abspath(omexPath)
        print("omexPath:", omexPath)
        print("workingDir:", workingDir)

        if not os.path.exists(workingDir):
            raise IOError("Working directory does not exist: {}".format(workingDir))

        if add_entries is False:
            if os.path.exists(omexPath):
                # delete the old omex file
                warnings.warn("Combine archive is overwritten: {}".format(omexPath))
                os.remove(omexPath)

        archive = libcombine.CombineArchive()

        if add_entries is True:
            # use existing entries
            if os.path.exists(omexPath):
                # init archive from existing content
                if archive.initializeFromArchive(omexPath) is None:
                    raise IOError("Combine Archive is invalid: ", omexPath)

        # timestamp
        time_now = libcombine.OmexDescription.getCurrentDateAndTime()

        print("*" * 80)
        for entry in entries:
            print(entry)
            location = entry.location
            path = os.path.join(workingDir, location)
            if not os.path.exists(path):
                raise IOError("File does not exist at given location: {}".format(path))

            archive.addFile(path, location, entry.format, entry.master)

            if entry.description or entry.creators:
                omex_d = libcombine.OmexDescription()
                omex_d.setAbout(location)
                omex_d.setCreated(time_now)

                if entry.description:
                    omex_d.setDescription(entry.description)

                if entry.creators:
                    for c in entry.creators:
                        creator = libcombine.VCard()
                        creator.setFamilyName(c.family_name)
                        creator.setGivenName(c.given_name)
                        creator.setEmail(c.email)
                        creator.setOrganization(c.organization)
                        omex_d.addCreator(creator)

                archive.addMetadata(location, omex_d)

        archive.writeToFile(omexPath)
        archive.cleanUp()


    def extract(self, working_dir: Path=None, method: str = "zip") -> None:
        """Extracts combine archive to working directory.

        The zip method extracts all entries in the zip, the omex method
        only extracts the entries listed in the manifest.
        In some archives not all content is listed in the manifest.

        :param omex_path: COMBINE archive
        :param output_dir: output directory
        :param method: method to extract content, either 'zip' or 'omex'
        :return:
        """
        if working_dir is None:
            output_dir = self.working_dir

        if method == "zip":
            zip_ref = zipfile.ZipFile(self.omex_path, "r")
            zip_ref.extractall(output_dir)
            zip_ref.close()

        elif method == "omex":
            omex = libcombine.CombineArchive()
            if omex.initializeFromArchive(str(omex_path)) is None:
                raise IOError(f"Invalid COMBINE archive: {omex_path}")

            for i in range(omex.getNumEntries()):
                entry = omex.getEntry(i)
                location = entry.getLocation()
                filename = os.path.join(output_dir, location)
                omex.extractEntry(location, filename)

            omex.cleanUp()
        else:
            raise ValueError(f"Method is not supported '{method}'")


def get_locations_by_format(omex_path: Path, format_key=None, method="omex"):
    """Returns locations to files with given format in the archive.

    Uses the libcombine KnownFormats for formatKey, e.g., 'sed-ml' or 'sbml'.
    Files which have a master=True have higher priority and are listed first.

    :param omex_path:
    :param format_key:
    :param method:
    :return:
    """
    if not format_key:
        raise ValueError("Format must be specified.")

    locations_master: List[str] = []
    locations: List[str] = []

    if method == "omex":
        omex: libcombine.CombineArchive = libcombine.CombineArchive()
        if omex.initializeFromArchive(str(omex_path)) is None:
            raise IOError(f"Invalid COMBINE Archive: {omex_path}")

        for i in range(omex.getNumEntries()):
            entry: libcombine.CaContent = omex.getEntry(i)
            format: str = entry.getFormat()
            master: bool = entry.getMaster()
            if libcombine.KnownFormats.isFormat(format_key, format):
                loc: str = entry.getLocation()
                if (master is None) or (master is False):
                    locations.append(loc)
                else:
                    locations_master.append(loc)
        omex.cleanUp()

    elif method == "zip":
        # extract to tmpfile and guess format
        tmp_dir = tempfile.mkdtemp()

        try:
            extract_combine_archive(omex_path, output_dir=tmp_dir, method="zip")

            # iterate over all locations & guess format
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    location = os.path.relpath(file_path, tmp_dir)
                    # guess the format
                    format = libcombine.KnownFormats.guessFormat(file_path)
                    if libcombine.KnownFormats.isFormat(
                        formatKey=format_key, format=format
                    ):
                        locations.append(location)
                    # print(format, "\t", location)

        finally:
            shutil.rmtree(tmp_dir)

    else:
        raise ValueError(f"Method is not supported '{method}'")

    return locations_master + locations


def listContents(omexPath, method="omex"):
    """Returns list of contents of the combine archive.

    :param omexPath:
    :param method: method to extract content, only 'omex' supported
    :return: list of contents
    """
    if method not in ["omex"]:
        raise ValueError("Method is not supported: {}".format(method))

    contents = []
    omex = libcombine.CombineArchive()
    if omex.initializeFromArchive(omexPath) is None:
        raise IOError("Invalid Combine Archive: {}", omexPath)

    for i in range(omex.getNumEntries()):
        entry = omex.getEntry(i)
        location = entry.getLocation()
        format = entry.getFormat()
        master = entry.getMaster()
        info = None
        try:
            for formatKey in ["sed-ml", "sbml", "sbgn", "cellml"]:
                if libcombine.KnownFormats_isFormat(formatKey, format):
                    info = omex.extractEntryToString(location)
        except:
            pass

        contents.append([i, location, format, master, info])

    omex.cleanUp()

    return contents


def print_contents(omexPath):
    """Prints contents of archive.

    :param omexPath:
    :return:
    """
    pprint.pprint(listContents(omexPath))
