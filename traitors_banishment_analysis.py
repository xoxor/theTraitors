import pandas as pd
import numpy as np
from pathlib import Path
import glob


# This will look for all CSVs starting with 'UK_traitors'
DATA_FILES_PATTERN = "data/*.csv"
OUTPUT_DIR = Path("outputs")
EARLY_EPISODE_CUTOFFS = 4   
GROUP_COLS = ["Inferred_Gender", "ethnicity_group"]

VOTES_FILES_PATTERN = "data/votes/*.csv"


def load_and_prepare_all_seasons(pattern):
    all_files = glob.glob(pattern)
    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    combined_list = []
    
    for file in all_files:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        
        # Standardize text fields
        for col in ["Finish", "Inferred_Gender", "Inferred_Ethnicity"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()

        # Group ethnicity
        if "Inferred_Ethnicity" in df.columns:
            df["ethnicity_group"] = np.where(
                df["Inferred_Ethnicity"] == "white",
                "white",
                "person_of_color"
            )

        df["is_banished"] = df["Finish"] == "banished"
        df["is_murdered"] = df["Finish"] == "murdered"

        combined_list.append(df)

    return pd.concat(combined_list, ignore_index=True)

def load_votes(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No vote files found: {pattern}")

    dfs = []
    for file in files:
        print(f"Loading votes from {file}...")
        df = pd.read_csv(file)

        # Standardize columns
        df.columns = df.columns.str.strip()
        df["player"] = df["player"].astype(str).str.strip()
        df["target"] = df["target"].astype(str).str.strip()
        
        # Both player and target are now IDs; no need to generate target_id
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def enrich_votes_with_demographics(votes_df, contestants_df):

    votes = votes_df.copy()

    votes = votes[votes["target"].astype(str).str.contains(r'^\d|^C', regex=True, na=False)]

    # Merge target demographics
    merged = votes.merge(
        contestants_df[['player_id', 'Inferred_Gender', 'ethnicity_group']],
        left_on='target',
        right_on='player_id',
        how='left'
    ).rename(columns={
        'Inferred_Gender': 'target_gender',
        'ethnicity_group': 'target_ethnicity'
    }).drop(columns=['player_id'])

    # Merge voter demographics
    merged = merged.merge(
        contestants_df[['player_id', 'Inferred_Gender', 'ethnicity_group']],
        left_on='player',
        right_on='player_id',
        how='left'
    ).rename(columns={
        'Inferred_Gender': 'voter_gender',
        'ethnicity_group': 'voter_ethnicity'
    }).drop(columns=['player_id'])

    print(f"Votes after enrichment: {len(merged)} rows")
    print(f"Columns in enriched votes: {merged.columns.tolist()}")
    return merged


def baseline_composition(df, season=None):
    data = df if season is None else df[df["Season"] == season]
    rows = []
    for col in GROUP_COLS:
        vc = data[col].value_counts(normalize=True).reset_index()
        vc.columns = ["group_value", "proportion"]
        vc["group_type"] = col
        vc["season"] = "all" if season is None else season
        rows.append(vc)
    return pd.concat(rows, ignore_index=True)

def early_banishment_stats(df, episode_cutoff, season=None):
    data = df if season is None else df[df["Season"] == season]

    early = data[
        (data["is_banished"]) &
        (data["Episode"] <= episode_cutoff)
    ].copy()

    if early.empty:
        return pd.DataFrame()

    total_early = len(early)
    rows = []

    for col in GROUP_COLS:
        grouped = early.groupby(col)

        for group_value, g in grouped:
            rows.append({
                "group_type": col,
                "group_value": group_value,
                "season": "all" if season is None else season,
                "episode_cutoff": episode_cutoff,
                "early_banished": len(g),
                "percentage_of_early_banishments": len(g) / total_early,
                "early_banished_names": sorted(
                    g["Contestant"].dropna().unique().tolist()
                )
            })

    return pd.DataFrame(rows)


def survival_stats(df, season=None):
    data = df if season is None else df[df["Season"] == season]
    summary = data.groupby(["Inferred_Gender", "ethnicity_group"]).agg(
        median_episode=("Episode", "median"),
        mean_episode=("Episode", "mean"),
        count=("Episode", "count")
    ).reset_index()
    summary["season"] = "all" if season is None else season
    return summary

def age_survival_stats(df, season=None):
    data = df if season is None else df[df["Season"] == season]

    if "Age" not in data.columns:
        return pd.DataFrame()
   
    bins = [0, 30, 45, 60, 100]
    labels = ["<30", "30-44", "45-59", "60+"]
    data["age_group"] = pd.cut(data["Age"], bins=bins, labels=labels)

    summary = data.groupby("age_group").agg(
        median_episode=("Episode", "median"),
        mean_episode=("Episode", "mean"),
        count=("Episode", "count")
    ).reset_index()
    
    summary["season"] = "all" if season is None else season
    return summary


def early_banishment_composition(df, episode_cutoff, season=None):
    data = df if season is None else df[df["Season"] == season]

    # Only early banishments
    early = data[
        (data["is_banished"]) &
        (data["Episode"] <= episode_cutoff)
    ].copy()

    if early.empty:
        return pd.DataFrame()

    rows = []

    for col in GROUP_COLS:
        counts = (
            early[col]
            .value_counts()
            .reset_index()
        )
        counts.columns = ["group_value", "early_banished"]

        total_early = counts["early_banished"].sum()

        counts["percentage_of_early_banishments"] = (
            counts["early_banished"] / total_early
        )

        counts["group_type"] = col
        counts["episode_cutoff"] = episode_cutoff
        counts["season"] = "all" if season is None else season

        rows.append(counts)

    return pd.concat(rows, ignore_index=True)

def finalist_composition(df, season=None):
    data = df if season is None else df[df["Season"] == season]

    # Identify finalists
    max_ep = data.groupby('Season')['Episode'].max().reset_index()
    max_ep = max_ep.rename(columns={'Episode': 'final_episode'})
    
    finalists = data.merge(max_ep, on='Season')
    finalists = finalists[finalists['Episode'] == finalists['final_episode']]
    if finalists.empty:
        return pd.DataFrame()

    rows = []

    for col in GROUP_COLS:
        vc = finalists[col].value_counts(normalize=True).reset_index()
        vc.columns = ["group_value", "proportion"]
        vc["group_type"] = col
        vc["season"] = "all" if season is None else season
        rows.append(vc)

    return pd.concat(rows, ignore_index=True)

def contestant_lookup(df):
    return (
        df[[
            "Season",
            "Contestant",
            "player_id",
            "Inferred_Gender",
            "ethnicity_group"
        ]]
        .drop_duplicates()
        .rename(columns={"Contestant": "target"})
    )

def get_round_baseline(votes_df, round_number, season=None):

    data = votes_df[votes_df['round_table'] == round_number]
    if season is not None:
        data = data[data['Season'] == season]

    if data.empty:
        return pd.DataFrame()

    # Unique players in this round
    active_players = data[['player', 'voter_gender', 'voter_ethnicity']].drop_duplicates()

    baseline = (
        active_players.groupby(['voter_gender', 'voter_ethnicity'])
        .size()
        .reset_index(name='player_count')
    )

    total_active = baseline['player_count'].sum()
    
    baseline['baseline_proportion'] = baseline['player_count'] / total_active
    baseline['Round'] = round_number
    baseline['Season'] = season if season else 'all'
    return baseline

def get_round_votes(votes_enriched_df, round, season=None):
    data = votes_enriched_df[votes_enriched_df['round_table'] == round]
    if season is not None:
        data = data[data["Season"] == season]

    if data.empty:
        return pd.DataFrame()

    data = data.copy()
    data["Round"] = round
    data["Season"] = season if season else "all"
    return data


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        df = load_and_prepare_all_seasons(DATA_FILES_PATTERN)
    except FileNotFoundError as e:
        print(e)
        return

    seasons = sorted(df["Season"].unique(), key=lambda x: (isinstance(x, str), x))
    print(f"Analyzing Seasons: {seasons}")

    # Baseline
    baseline = pd.concat([baseline_composition(df)] + 
                         [baseline_composition(df, s) for s in seasons], ignore_index=True)

    # Early Banishment
    early_rows = []
    cutoff = EARLY_EPISODE_CUTOFFS
    early_rows.append(early_banishment_stats(df, cutoff))
    for season in seasons:
        early_rows.append(early_banishment_stats(df, cutoff, season))
    early_stats = pd.concat(early_rows, ignore_index=True)

    # Survival
    survival = pd.concat([survival_stats(df)] + 
                         [survival_stats(df, s) for s in seasons], ignore_index=True)

    # Survival by Age
    age_survival = pd.concat([age_survival_stats(df)] +
                         [age_survival_stats(df, s) for s in seasons],
                         ignore_index=True)

    finalist_comp = pd.concat([finalist_composition(df)] +
                         [finalist_composition(df, s) for s in seasons],
                         ignore_index=True)
    
    votes_df = load_votes(VOTES_FILES_PATTERN)
    votes_enriched = enrich_votes_with_demographics(votes_df, df)
    print(f"Total votes after enrichment: {len(votes_enriched)}")
    # Vote Composition
    early_votes = []
    max_round = int(votes_enriched['round_table'].max())
    for i in range(1, max_round + 1):
        early_votes.append(get_round_votes(votes_enriched, i))
        for s in seasons:
            early_votes.append(get_round_votes(votes_enriched, i, s))

    early_votes = pd.concat(early_votes, ignore_index=True)
  
    # Baseline Episodes
    baseline_rounds = None
    for i in range(1, max_round + 1):
        baseline_round = pd.concat(
            [get_round_baseline(votes_enriched, i)] +
            [get_round_baseline(votes_enriched, i, s) for s in seasons],
            ignore_index=True
        )
        if baseline_rounds is None:
            baseline_rounds = baseline_round
        else:
            baseline_rounds = pd.concat([baseline_rounds, baseline_round], ignore_index=True)   
    vote_counts = votes_enriched.groupby(["voter_gender", "voter_ethnicity"]).size().reset_index(name="vote_count")
    print("\nOverall Vote Counts by Demographics:")
    print(vote_counts)
    # Save outputs
    baseline.to_csv(OUTPUT_DIR / "baseline_composition.csv", index=False)
    early_stats.to_csv(OUTPUT_DIR / "early_banishment_stats.csv", index=False)
    survival.to_csv(OUTPUT_DIR / "survival_stats.csv", index=False)
    age_survival.to_csv(OUTPUT_DIR / "age_survival_stats.csv", index=False)
    finalist_comp.to_csv(OUTPUT_DIR / "finalist_composition.csv", index=False)
    early_votes.to_csv(OUTPUT_DIR / "early_vote_composition.csv", index=False)
    baseline_rounds.to_csv(OUTPUT_DIR / "baseline_rounds.csv", index=False)
    
                                
    # JSON Outputs (unique columns fix included)
    baseline.to_json(OUTPUT_DIR / "baseline_composition.json", orient="records")
    early_stats.to_json(OUTPUT_DIR / "early_banishment_stats.json", orient="records")
    survival.to_json(OUTPUT_DIR / "survival_stats.json", orient="records")
    age_survival.to_json(OUTPUT_DIR / "age_survival_stats.json", orient="records")
    finalist_comp.to_json(OUTPUT_DIR / "finalist_composition.json", orient="records")
    early_votes.to_json(OUTPUT_DIR / "early_vote_composition.json", orient="records")
    baseline_rounds.to_json(OUTPUT_DIR / "baseline_rounds.json", orient="records")
    
    print(f"\nSuccess! Combined analysis for {len(seasons)} seasons completed.")
    print(f"Files saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()