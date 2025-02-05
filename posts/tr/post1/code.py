from datafc.sofascore import (
    match_data,
    match_odds_data,
    match_stats_data,
    momentum_data,
    lineups_data,
    coordinates_data,
    substitutions_data,
    goal_networks_data,
    shots_data,
    standings_data
)

match_df = match_data(
    tournament_id=52,
    season_id=63814,
    week_number=22
)

match_odds_df = match_odds_data(
    match_df=match_df
)

match_stats_df = match_stats_data(
    match_df=match_df
)

momentum_df = momentum_data(
    match_df=match_df
)

lineups_df = lineups_data(
    match_df=match_df
)

substitutions_df = substitutions_data(
    match_df=match_df
)

goal_networks_df = goal_networks_data(
    match_df=match_df
)

shots_df = shots_data(
    match_df=match_df
)

coordinates_df = coordinates_data(
    lineups_df=lineups_df
)

standings_df = standings_data(
    tournament_id=52,
    season_id=63814
)