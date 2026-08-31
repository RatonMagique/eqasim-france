import pandas as pd
import pytest

from data.census import completed


class Context:
    def __init__(self, stages, config):
        self.stages = stages
        self.values = config
        self.info = {}

    def stage(self, name):
        return self.stages[name]

    def config(self, name):
        return self.values[name]

    def set_info(self, name, value):
        self.info[name] = value


def _census():
    return pd.DataFrame({
        "person_id": [0, 1, 2, 3],
        "household_id": [10, 10, 11, 11],
        "weight": [1.0, 1.0, 1.0, 1.0],
        "household_size": [2, 2, 2, 2],
        "iris_id": ["100010001"] * 4,
        "commune_id": ["10001"] * 4,
        "departement_id": ["10"] * 4,
        "region_id": ["11"] * 4,
        "age": [30, 28, 12, 10],
        "sex": ["male", "female", "male", "female"],
        "socioprofessional_class": [3, 5, 8, 8],
    })


def _codes():
    return pd.DataFrame({
        "iris_id": ["100020000"], "commune_id": ["10002"],
        "departement_id": ["10"], "region_id": [11],
    })


def _population():
    return pd.DataFrame({
        "iris_id": ["100020000"], "commune_id": ["10002"],
        "departement_id": ["10"], "region_id": [11], "population": [4.0],
        "P22_POPH": [2.0], "P22_POPF": [2.0],
        "P22_POP1117": [2.0], "P22_POP2539": [2.0],
    })


def _context(strategy):
    return Context(
        {"data.census.cleaned": _census(), "data.spatial.codes": _codes(), "data.spatial.population": _population()},
        {"census_missing_iris_strategy": strategy, "census_fallback_max_donor_households": 100,
         "census_fallback_iterations": 20, "random_seed": 1234},
    )


def test_missing_iris_is_synthesized_from_department_donors():
    context = _context("department_region_donors")
    result = completed.execute(context)

    assert set(result["iris_id"].astype(str)) == {"100020000"}
    assert set(result["commune_id"].astype(str)) == {"10002"}
    assert set(result["departement_id"].astype(str)) == {"10"}
    assert result["household_id"].nunique() == 2
    weights = result.groupby("household_id", observed=True)["weight"].first()
    sizes = result.groupby("household_id", observed=True).size()
    assert weights @ sizes == pytest.approx(4.0)
    assert context.info["fallback"]["100020000"]["source_scope"] == "department"


def test_missing_iris_is_not_synthesized_without_opt_in():
    context = _context("error")
    result = completed.execute(context)

    assert len(result) == len(_census())
    assert context.info["fallback"]["missing_iris"] == ["100020000"]


def test_missing_iris_escalates_to_region_when_department_has_no_donors():
    context = _context("department_region_donors")
    context.stages["data.spatial.codes"] = pd.DataFrame({
        "iris_id": ["110020000"], "commune_id": ["11002"],
        "departement_id": ["11"], "region_id": [11],
    })
    context.stages["data.spatial.population"] = pd.DataFrame({
        "iris_id": ["110020000"], "commune_id": ["11002"],
        "departement_id": ["11"], "region_id": [11], "population": [4.0],
    })

    result = completed.execute(context)

    assert set(result["departement_id"].astype(str)) == {"11"}
    assert context.info["fallback"]["110020000"]["source_scope"] == "region"
