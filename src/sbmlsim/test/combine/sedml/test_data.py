"""
Testing of SED-ML data support, i.e., DataDescription.
"""
import importlib
import os
from pathlib import Path

import libsedml

from sbmlsim.combine.sedml.data import DataDescriptionParser
from sbmlsim.combine.sedml.io import check_sedml, read_sedml
from sbmlsim.test import DATA_DIR


# ---------------------------------------------------------------------------------
BASE_DIR = DATA_DIR / "sedml" / "data"

SOURCE_CSV = BASE_DIR / "oscli.csv"
SOURCE_TSV = BASE_DIR / "oscli.tsv"

SEDML_READ_CSV = BASE_DIR / "reading-oscli-csv.xml"
SEDML_READ_TSV = BASE_DIR / "reading-oscli-tsv.xml"
SEDML_READ_NUML = BASE_DIR / "reading-oscli-numl.xml"
SEDML_READ_NUML_1D = BASE_DIR / "reading-numlData1D.xml"
SEDML_READ_NUML_2D = BASE_DIR / "reading-numlData2D.xml"
SEDML_READ_NUML_2DRC = BASE_DIR / "reading-numlData2DRC.xml"

OMEX_PLOT_CSV = BASE_DIR / "omex" / "plot_csv.omex"
OMEX_PLOT_CSV_WITH_MODEL = BASE_DIR / "omex", "plot_csv_with_model.omex"
OMEX_PLOT_NUML = BASE_DIR / "omex" / "plot_numl.omex"
OMEX_PLOT_NUML_WITH_MODEL = BASE_DIR / "omex", "plot_numl_with_model.omex"

SOURCE_CSV_PARAMETERS = BASE_DIR / "parameters.csv"
SEDML_CSV_PARAMETERS = BASE_DIR / "parameter-from-data-csv.xml"
OMEX_CSV_PARAMETERS = BASE_DIR / "omex", "parameter_from_data_csv.omex"

OMEX_CSV_JWS_ADLUNG2017_FIG2G = BASE_DIR / "omex" / "jws_adlung2017_fig2g.omex"
# ---------------------------------------------------------------------------------


def test_load_csv():
    data = DataDescriptionParser._load_csv(SOURCE_CSV)
    assert data is not None
    assert data.shape[0] == 200
    assert data.shape[1] == 3


def test_load_tsv():
    data = DataDescriptionParser._load_tsv(SOURCE_TSV)
    assert data is not None
    assert data.shape[0] == 200
    assert data.shape[1] == 3


def test_load_csv_parameters():
    data = DataDescriptionParser._load_csv(SOURCE_CSV_PARAMETERS)
    assert data is not None
    assert data.shape[0] == 10
    assert data.shape[1] == 1


def _parseDataDescriptions(sedml_path):
    """Test helper functions.

    Tries to parse all DataDescriptions in the SED-ML file.
    """
    importlib.reload(libsedml)
    print("parseDataDescriptions:", sedml_path)

    # load sedml document
    sedml_path_str = sedml_path
    if isinstance(sedml_path, Path):
        sedml_path_str = str(sedml_path)
    assert os.path.exists(sedml_path_str)

    doc_sedml = libsedml.readSedMLFromFile(sedml_path_str)
    check_sedml(doc_sedml)

    # parse DataDescriptions
    list_dd = doc_sedml.getListOfDataDescriptions()
    # print(list_dd)
    # print(len(list_dd))

    assert len(list_dd) > 0

    for dd in list_dd:
        print(type(dd))
        data_sources = DataDescriptionParser.parse(dd, workingDir=BASE_DIR)
        assert data_sources is not None
        assert type(data_sources) == dict
        assert len(data_sources) > 0
    return data_sources


def test_parse_csv():
    data_sources = _parseDataDescriptions(SEDML_READ_CSV)
    assert "dataTime" in data_sources
    assert "dataS1" in data_sources
    assert len(data_sources["dataTime"]) == 200
    assert len(data_sources["dataS1"]) == 200


def test_parse_csv_parameters():
    data_sources = _parseDataDescriptions(SEDML_CSV_PARAMETERS)
    assert "dataIndex" in data_sources
    assert "dataMu" in data_sources
    assert len(data_sources["dataIndex"]) == 10
    assert len(data_sources["dataMu"]) == 10


def test_parse_tsv():
    data_sources = _parseDataDescriptions(SEDML_READ_TSV)
    assert "dataTime" in data_sources
    assert "dataS1" in data_sources
    assert len(data_sources["dataTime"]) == 200
    assert len(data_sources["dataS1"]) == 200


def test_parse_numl():
    data_sources = _parseDataDescriptions(SEDML_READ_NUML)
    assert "dataTime" in data_sources
    assert "dataS1" in data_sources
    assert len(data_sources["dataTime"]) == 200
    assert len(data_sources["dataS1"]) == 200


def test_parse_numl_1D():
    data_sources = _parseDataDescriptions(SEDML_READ_NUML_1D)
    assert data_sources is not None
    assert len(data_sources) == 6
    assert "data_s_glu" in data_sources
    assert "data_s_pyr" in data_sources
    assert "data_s_acetate" in data_sources
    assert "data_s_acetald" in data_sources
    assert "data_s_EtOH" in data_sources
    assert "data_x" in data_sources
    assert len(data_sources["data_s_glu"]) == 1


def test_parse_numl_2D():
    data_sources = _parseDataDescriptions(SEDML_READ_NUML_2D)
    assert data_sources is not None
    assert len(data_sources) == 4
    assert "dataBL" in data_sources
    assert "dataB" in data_sources
    assert "dataS1" in data_sources
    assert "dataTime" in data_sources
    assert len(data_sources["dataB"]) == 6


def test_parse_numl_2DRC():
    data_sources = _parseDataDescriptions(SEDML_READ_NUML_2DRC)
    assert data_sources is not None
    assert len(data_sources) == 4
    assert "dataBL" in data_sources
    assert "dataB" in data_sources
    assert "dataS1" in data_sources
    assert "dataTime" in data_sources
    assert len(data_sources["dataB"]) == 6
