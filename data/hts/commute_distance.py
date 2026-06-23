import pandas as pd
import numpy as np

def configure(context):
    context.config("random_seed")
    context.stage("data.hts.selected")
    context.stage("data.hts.selected", dict(weekday = "any"), alias = "hts_reference")

def _build_distribution(df_persons, df_trips, activity_type):
    if "euclidean_distance" in df_trips:
        distance_slot = "euclidean_distance"
        distance_factor = 1.0
    else:
        distance_slot = "routed_distance"
        distance_factor = 1.0 # / 1.3

    df_commute_distance = df_trips[
        ((df_trips["preceding_purpose"] == "home") & (df_trips["following_purpose"] == activity_type)) |
        ((df_trips["preceding_purpose"] == activity_type) & (df_trips["following_purpose"] == "home"))
    ][["person_id", distance_slot]].copy()

    df_commute_distance = pd.merge(
        df_commute_distance,
        df_persons[["person_id", "person_weight"]],
        on = "person_id",
        how = "left"
    ).dropna(subset = [distance_slot, "person_weight"])

    return df_commute_distance.rename(columns = { distance_slot: "commute_distance" }), distance_factor

def get_commuting_distance(df_persons, df_trips, df_reference_persons, df_reference_trips, activity_type, random):
    df_commute_distance, distance_factor = _build_distribution(df_persons, df_trips, activity_type)

    if len(df_commute_distance) == 0:
        # Weekend-only samples may have no commute trips. Use the full HTS as reference.
        df_commute_distance, distance_factor = _build_distribution(df_reference_persons, df_reference_trips, activity_type)

    if len(df_commute_distance) == 0:
        raise ValueError("No commute-distance observations available for '%s' in either the selected sample or the full HTS reference." % activity_type)

    # Add commuting distances from the selected sample when available.
    #
    # The distribution above can contain multiple trips per person. That is
    # useful for sampling, but the returned table must stay person-level so
    # downstream merges remain one-to-one. Keep the first observed commute
    # distance per person, matching the analysis code.
    df_commute_distance_lookup = df_commute_distance[["person_id", "commute_distance"]].drop_duplicates("person_id", keep = "first")
    df_persons = pd.merge(df_persons, df_commute_distance_lookup, on = "person_id", how = "left")

    # For the ones without commuting distance, sample from the distribution.
    f_missing = df_persons["commute_distance"].isna()

    values = df_commute_distance["commute_distance"].values
    weights = df_commute_distance["person_weight"].values

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    indices = [
        np.searchsorted(cdf, r)
        for r in random.random(size = np.count_nonzero(f_missing))
    ]

    df_persons.loc[f_missing, "commute_distance"] = values[indices]

    # Set flag for reference
    df_persons["imputed"] = f_missing

    # Attach euclidean factor
    df_persons["commute_distance"] *= distance_factor

    print("Missing %s commute distances: %.2f%%" % (
        activity_type, 100 * np.count_nonzero(f_missing) / len(f_missing)
    ))

    return df_persons

def execute(context):
    df_households, df_persons, df_trips = context.stage("data.hts.selected")
    df_reference_households, df_reference_persons, df_reference_trips = context.stage("hts_reference")
    random = np.random.default_rng(context.config("random_seed"))

    return dict(
        work = get_commuting_distance(df_persons, df_trips, df_reference_persons, df_reference_trips, "work", random),
        education = get_commuting_distance(df_persons, df_trips, df_reference_persons, df_reference_trips, "education", random)
    )
