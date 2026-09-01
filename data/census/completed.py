import zlib

import numpy as np
import pandas as pd

"""Complete missing IRIS census microdata with calibrated donor households."""

AGE_MARGINALS = {
    "P22_POP0002": (0, 2), "P22_POP0305": (3, 5),
    "P22_POP0610": (6, 10), "P22_POP1117": (11, 17),
    "P22_POP1824": (18, 24), "P22_POP2539": (25, 39),
    "P22_POP4054": (40, 54), "P22_POP5564": (55, 64),
    "P22_POP6579": (65, 79), "P22_POP80P": (80, np.inf),
}
SOCIO_MARGINALS = {
    "C22_POP15P_STAT_GSEC11_21": 1, "C22_POP15P_STAT_GSEC12_22": 2,
    "C22_POP15P_STAT_GSEC13_23": 3, "C22_POP15P_STAT_GSEC14_24": 4,
    "C22_POP15P_STAT_GSEC15_25": 5, "C22_POP15P_STAT_GSEC16_26": 6,
    "C22_POP15P_STAT_GSEC32": 7, "C22_POP15P_STAT_GSEC40": 8,
}


def configure(context):
    context.stage("data.census.cleaned")
    context.stage("data.spatial.codes")
    context.stage("data.spatial.population")
    context.config("census_missing_iris_strategy", "error")
    context.config("census_fallback_max_donor_households", 1000)
    context.config("census_fallback_iterations", 30)
    context.config("random_seed")


def _requested_mask(df, codes):
    return (
        df["departement_id"].astype(str).isin(codes["departement_id"].astype(str))
        & df["commune_id"].astype(str).isin(codes["commune_id"].astype(str))
        & df["iris_id"].astype(str).isin(codes["iris_id"].astype(str))
    )


def _households(df):
    households = df[["household_id", "weight"]].drop_duplicates("household_id").copy()
    if households.empty:
        return households
    weight_count = df.groupby("household_id", observed=True)["weight"].nunique()
    if (weight_count > 1).any():
        raise RuntimeError("Census donor household members must share one weight")
    return households.sort_values("household_id").reset_index(drop=True)


def _select_donors(df, maximum, random_seed, iris_id):
    households = _households(df)
    if len(households) <= maximum:
        return households["household_id"].to_numpy()
    probabilities = households["weight"].to_numpy(dtype=float, copy=True)
    probabilities /= probabilities.sum()
    seed = (random_seed + zlib.crc32(str(iris_id).encode())) % (2**32)
    random = np.random.default_rng(seed)
    selected = random.choice(len(households), size=maximum, replace=False, p=probabilities)
    return households.iloc[np.sort(selected)]["household_id"].to_numpy()


def _constraint_matrix(df, household_ids, target):
    donor = df[df["household_id"].isin(household_ids)]
    index = pd.Index(household_ids)
    constraints = [("population", pd.Series(True, index=donor.index), float(target["population"]))]

    for column, value in (("P22_POPH", "male"), ("P22_POPF", "female")):
        if column in target.index and pd.notna(target[column]):
            constraints.append((column, donor["sex"] == value, float(target[column])))
    for column, (minimum, maximum) in AGE_MARGINALS.items():
        if column in target.index and pd.notna(target[column]):
            constraints.append((column, donor["age"].between(minimum, maximum), float(target[column])))
    for column, category in SOCIO_MARGINALS.items():
        if column in target.index and pd.notna(target[column]):
            constraints.append((column, (donor["age"] >= 15) & (donor["socioprofessional_class"] == category), float(target[column])))

    names, targets, counts = [], [], []
    for name, mask, value in constraints:
        counts.append(donor.loc[mask].groupby("household_id", observed=True).size().reindex(index, fill_value=0).to_numpy(dtype=float))
        names.append(name)
        targets.append(value)
    return np.column_stack(counts), np.asarray(targets), names


def calibrate_household_weights(df, household_ids, target, iterations=30):
    """Rake whole donor households; total population remains an exact constraint."""
    donors = df[df["household_id"].isin(household_ids)]
    base_weights = _households(donors).set_index("household_id").loc[household_ids, "weight"].to_numpy(dtype=float)
    counts, targets, names = _constraint_matrix(df, household_ids, target)
    weighted_total = counts[:, 0] @ base_weights
    if weighted_total <= 0:
        raise RuntimeError("Donor households have no weighted population")
    weights = base_weights * (targets[0] / weighted_total)
    household_sizes = np.maximum(counts[:, 0], 1.0)

    for _ in range(iterations):
        for column in range(1, counts.shape[1]):
            current, desired = counts[:, column] @ weights, targets[column]
            if current <= 0 or desired <= 0:
                continue
            factor = np.clip(desired / current, 0.1, 10.0)
            weights *= factor ** (counts[:, column] / household_sizes)
        weights *= targets[0] / (counts[:, 0] @ weights)

    residuals = {name: float(counts[:, i] @ weights - targets[i]) for i, name in enumerate(names)}
    return pd.Series(weights, index=household_ids), residuals


def _synthesize(target, scope, maximum, random_seed, iterations, next_household_id):
    donor_ids = _select_donors(scope, maximum, random_seed, target["iris_id"])
    if len(donor_ids) == 0:
        raise RuntimeError("No usable census donor households are available")
    weights, residuals = calibrate_household_weights(scope, donor_ids, target, iterations)
    fallback = scope[scope["household_id"].isin(donor_ids)].copy()
    new_ids = np.arange(next_household_id, next_household_id + len(donor_ids))
    replacement = pd.Series(new_ids, index=donor_ids)
    fallback["household_id"] = fallback["household_id"].map(replacement).astype(int)
    fallback["weight"] = fallback["household_id"].map(pd.Series(weights.to_numpy(), index=new_ids)).astype(float)
    fallback["region_id"] = str(target["region_id"])
    for column in ["iris_id", "commune_id", "departement_id"]:
        fallback[column] = str(target[column])
    return fallback, residuals, next_household_id + len(donor_ids)


def execute(context):
    census = context.stage("data.census.cleaned")
    codes = context.stage("data.spatial.codes")
    population = context.stage("data.spatial.population")
    strategy = context.config("census_missing_iris_strategy")

    direct = census[_requested_mask(census, codes)].copy()
    requested_iris = set(codes["iris_id"].astype(str))
    missing = sorted(requested_iris - set(direct["iris_id"].astype(str)))
    if not missing:
        return direct
    if strategy == "error":
        # Preserve the historical behavior exactly: data.census.filtered keeps
        # its original categorical filtering semantics in strict mode.
        context.set_info("fallback", {"missing_iris": missing, "strategy": "error"})
        return census
    if strategy != "department_region_donors":
        raise RuntimeError("Unknown census_missing_iris_strategy: %s" % strategy)

    targets = population[population["iris_id"].astype(str).isin(missing)].copy()
    if len(targets) != len(missing):
        known = set(targets["iris_id"].astype(str))
        raise RuntimeError("Aggregate population is missing requested IRIS: %s" % sorted(set(missing) - known))

    # An IRIS can be present in the spatial selection while having no residents.
    # It needs no synthetic donor households; trying to calibrate one would
    # initialise all weights to zero and subsequently normalise 0 / 0.
    empty_targets = targets[targets["population"] <= 0]
    targets = targets[targets["population"] > 0]

    fallback, diagnostics = [], {}
    for _, target in empty_targets.sort_values("iris_id").iterrows():
        diagnostics[str(target["iris_id"])] = {
            "source_scope": None,
            "target_population": float(target["population"]),
            "weighted_population": 0.0,
            "residuals": {},
        }
    next_household_id = int(census["household_id"].max()) + 1 if len(census) else 0
    for _, target in targets.sort_values("iris_id").iterrows():
        department = census[census["departement_id"].astype(str) == str(target["departement_id"])]
        region = census[census["region_id"].astype(str) == str(target["region_id"])]
        scope_name, scope = "department", department
        if _households(scope).empty:
            scope_name, scope = "region", region
        generated, residuals, next_household_id = _synthesize(
            target, scope, context.config("census_fallback_max_donor_households"),
            context.config("random_seed"), context.config("census_fallback_iterations"), next_household_id)
        fallback.append(generated)
        household_weights = generated.groupby("household_id", observed=True)["weight"].first()
        household_sizes = generated.groupby("household_id", observed=True).size()
        diagnostics[str(target["iris_id"])] = {
            "source_scope": scope_name,
            "target_population": float(target["population"]),
            "weighted_population": float(household_weights @ household_sizes),
            "residuals": residuals,
        }

    completed = pd.concat([direct] + fallback, ignore_index=True)
    completed["person_id"] = np.arange(len(completed))
    for column in ["iris_id", "commune_id", "departement_id", "region_id"]:
        completed[column] = completed[column].astype("category")
    context.set_info("fallback", diagnostics)
    return completed
