from tqdm import tqdm
import pandas as pd
import os

"""
This stage loads the raw data of the French HTS (ENTD).
"""

Q_MENAGE_COLUMNS = [
    "DEP", "idENT_MEN", "PONDV1", "RG",
    "V1_JNBVELOADT",
    "V1_JNBVEH", "V1_JNBMOTO", "V1_JNBCYCLO"
]

Q_TCM_MENAGE_COLUMNS = [
    "NPERS", "PONDV1", "TrancheRevenuMensuel",
    "DEP", "idENT_MEN", "RG", "numcom_UU2010"
]

Q_INDIVIDU_COLUMNS = [
    "IDENT_IND", "idENT_MEN",
    "RG", "V1_GPERMIS", "V1_ICARTABON",
    "V1_GPERMIS2R"
]

Q_TCM_INDIVIDU_COLUMNS = [
    "AGE", "ETUDES", "IDENT_IND", "IDENT_MEN",
    "PONDV1", "CS24", "SEXE", "DEP", "SITUA",
]

K_DEPLOC_REQUIRED_COLUMNS = [
    "IDENT_IND", "V2_MMOTIFDES", "V2_MMOTIFORI",
    "V2_TYPJOUR", "V2_MORIHDEP", "V2_MDESHARR", "V2_MDISTTOT",
    "IDENT_JOUR", "V2_MTP",
    "V2_MDESDEP", "V2_MORIDEP", "NDEP", "V2_MOBILREF",
    "PONDKI"
]

K_DEPLOC_OPTIONAL_COLUMNS = [
    "V2_JOUR_DEP"
]

K_MOBILITE_REQUIRED_COLUMNS = [
    "IDENT_IND", "V2_JOURSEMMOB"
]

K_MOBILITE_OPTIONAL_COLUMNS = [
    "V2_MDATEDEP", "V2_MDATESAMDEP", "V2_MDATEDIMDEP"
]


def _available_columns(path, sep = ';', encoding = 'latin1'):
    return list(pd.read_csv(path, sep = sep, encoding = encoding, nrows = 0).columns)


def read_csv_with_optional_columns(path, required_columns, optional_columns = None, sep = ';', encoding = 'latin1', dtype = None):
    optional_columns = optional_columns or []
    available_columns = set(_available_columns(path, sep = sep, encoding = encoding))
    usecols = list(required_columns) + [column for column in optional_columns if column in available_columns]
    return pd.read_csv(path, sep = sep, encoding = encoding, usecols = usecols, dtype = dtype)


def configure(context):
    context.config("data_path")


def execute(context):
    df_individu = pd.read_csv(
        "%s/entd_2008/Q_individu.csv" % context.config("data_path"),
        sep = ";", encoding = "latin1", usecols = Q_INDIVIDU_COLUMNS,
        dtype = { "DEP": str }
    )

    df_tcm_individu = pd.read_csv(
        "%s/entd_2008/Q_tcm_individu.csv" % context.config("data_path"),
        sep = ";", encoding = "latin1", usecols = Q_TCM_INDIVIDU_COLUMNS,
        dtype = { "DEP": str }
    )

    df_menage = pd.read_csv(
        "%s/entd_2008/Q_menage.csv" % context.config("data_path"),
        sep = ";", encoding = "latin1", usecols = Q_MENAGE_COLUMNS,
        dtype = { "DEP": str }
    )

    df_tcm_menage = pd.read_csv(
        "%s/entd_2008/Q_tcm_menage_0.csv" % context.config("data_path"),
        sep = ";", encoding = "latin1", usecols = Q_TCM_MENAGE_COLUMNS,
        dtype = { "DEP": str }
    )

    df_deploc = read_csv_with_optional_columns(
        "%s/entd_2008/K_deploc.csv" % context.config("data_path"),
        K_DEPLOC_REQUIRED_COLUMNS, K_DEPLOC_OPTIONAL_COLUMNS,
        dtype = { "DEP": str, "V2_MTP": str }
    )

    df_mobilite = read_csv_with_optional_columns(
        "%s/entd_2008/K_mobilite.csv" % context.config("data_path"),
        K_MOBILITE_REQUIRED_COLUMNS, K_MOBILITE_OPTIONAL_COLUMNS
    )

    return df_individu, df_tcm_individu, df_menage, df_tcm_menage, df_deploc, df_mobilite


def validate(context):
    for name in ("Q_individu.csv", "Q_tcm_individu.csv", "Q_menage.csv", "Q_tcm_menage_0.csv", "K_deploc.csv", "K_mobilite.csv"):
        if not os.path.exists("%s/entd_2008/%s" % (context.config("data_path"), name)):
            raise RuntimeError("File missing from ENTD: %s" % name)

    return [
        os.path.getsize("%s/entd_2008/Q_individu.csv" % context.config("data_path")),
        os.path.getsize("%s/entd_2008/Q_tcm_individu.csv" % context.config("data_path")),
        os.path.getsize("%s/entd_2008/Q_menage.csv" % context.config("data_path")),
        os.path.getsize("%s/entd_2008/Q_tcm_menage_0.csv" % context.config("data_path")),
        os.path.getsize("%s/entd_2008/K_deploc.csv" % context.config("data_path")),
        os.path.getsize("%s/entd_2008/K_mobilite.csv" % context.config("data_path"))
    ]
