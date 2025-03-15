from datafc.sofascore import match_data

match_df = match_data(
    tournament_id=7,
    season_id=61644,
    week_number=5,
    tournament_type="uefa",
    tournament_stage="round_of_16"
)