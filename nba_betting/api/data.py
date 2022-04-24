import logging
import os
from typing import Any, Tuple

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

year_to_reg_season_start ={
    2018: '2018-10-16',
    2019: '2019-10-22',
}

year_to_reg_season_end ={
    2018: '2019-04-10',
    2019: '2020-04-14',
}

# descriptive_cols = [
#     "SEASON_ID",
#     "TEAM_ID",
#     "TEAM_ABBREVIATION",
#     "TEAM_NAME",
#     "WL",
#     "MIN",
# ]
descriptive_cols = [
    "SEASON_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
]
overlap_cols = [
    "GAME_DATE",
    "OPPONENT",
    "TEAM_ABBREVIATION",
    "WL"
]


def get_team_id_from_abbr(team_abbr: str) -> int:

    nba_teams = teams.get_teams()
    # Select the dictionary for the Celtics, which contains their team ID
    team = [team for team in nba_teams if team["abbreviation"] == team_abbr][0]
    return int(team["id"])


def get_games_by_team_and_year(
    team_id: int, year: int, drop_extra: bool = False
) -> pd.DataFrame:
    """
    year: 2018 for the 2017-18 season
    """

    # Query for games where the Celtics were playing
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
    # The first DataFrame of those returned is what we want.
    games: pd.DataFrame = gamefinder.get_data_frames()[0]
    # adjust for year
    games = games.where((games.SEASON_ID.astype(int) % 10000) == year)

    if drop_extra:
        games = games.drop(descriptive_cols, axis=1)

    return games


def get_games_by_year(year: int, with_opponents=True) -> pd.DataFrame:
    """
    Get all games for a season
    """

    nba_teams = [t['id'] for t in teams.get_teams()]    
    
    # games by team
    games_dict = {}
    games = [get_games_by_team_szn(year, team_id) for team_id in nba_teams]

    logging.info('Getting game data...')
    for g in games:
        team, games = make_cumulative(g)
        games_dict[team] = games
    logging.info('Done getting game data')

    logging.info('Creating matchups...')
    full_data = None
    if with_opponents:
        full_data = add_opponents(games_dict, nba_teams)
    logging.info('Done creating matchups.')

    full_year_df = pd.concat([v for k,v in full_data.items()])
    return full_year_df

def add_opponents(games_dict, teams):

    with_opps = {}

    for key in games_dict:
        sub_df = games_dict[key]
        sub_dict = sub_df.to_dict('records')
        team_opp_dict = {}
        col_names = None
        
        for i, row in enumerate(sub_dict):

            # find opponent (abbreviation), find game date
            opponent = row['OPPONENT']
            game_date = row['GAME_DATE']

            row2 = pd.DataFrame(row, index=[0]) 
            opponent_df = games_dict[opponent]

            # get opponent cumulative stats
            # using game date, validate current team abbreviation
            opponent_row = opponent_df.loc[opponent_df['GAME_DATE'] == game_date].drop(overlap_cols, axis=1).add_prefix('OPP_').set_index(pd.Index([0]))
            # opp_cols = opponent_row

            # combine stats into 1 row
            combined = pd.concat([row2, opponent_row], axis=1).loc[0]
            # add to new dict
            team_opp_dict[i] = combined.tolist()

            if i == 0:
                col_names = row2.columns.append(opponent_row.columns)

        team_opp_df = pd.DataFrame.from_dict(team_opp_dict, orient='index')
        team_opp_df.columns = col_names
        with_opps[key] = team_opp_df

    return with_opps

def make_cumulative(df):

    # setup
    dates, matchup, team_name = df["GAME_DATE"], df["MATCHUP"], df["TEAM_ABBREVIATION"]
    matchup = [m[-3:] for m in matchup]
    wl = df['WL']
    df = df.drop(descriptive_cols, axis=1)
    dictionary_data = {}

    # get previous rows, get averages and append to dict
    for i in range(df.shape[0]):
        preceding_rows = df.iloc[:i]
        avg_til_now = list(preceding_rows.mean())
        if np.isnan(avg_til_now).any():
            avg_til_now = [0] * len(avg_til_now)
        dictionary_data[i] = avg_til_now

    # create and fix up final df
    df_final = pd.DataFrame.from_dict(dictionary_data, orient='index')
    df_final.columns = df.columns
    df_final['GAMES_PLAYED'] = df['GAMES_PLAYED']
    df_final['GAME_DATE'] = dates
    df_final['OPPONENT'] = matchup
    df_final['TEAM_ABBREVIATION'] = team_name
    df_final['WL'] = wl

    return team_name.iloc[0], df_final


def make_xy(df: pd.DataFrame) -> "Tuple[pd.DataFrame, pd.Series[str]]":
    """
    Drop columns we won't need
    """
    y = df["WL"]
    X = df.drop(overlap_cols, axis=1)
    return X, y


def games_team_szn_filename(
    year: int, team_id: int, remove_playoff: bool = True
) -> str:
    if remove_playoff:
        rp = "noplayoff"
    else:
        rp = "yesplayoff"
    return str(year) + "_" + str(team_id) + "_" + rp + ".csv"


def get_games_by_team_szn(
    year: int, team_id: int, remove_playoff: bool = True
) -> pd.DataFrame:
    """
    number of games played this season before current game
    team_id required because each game has 2 rows, one for each team
    """
    # get filename, make data folders
    filename = games_team_szn_filename(year, team_id, remove_playoff)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    filename = os.path.join("data/raw", filename)

    # check cached
    if os.path.exists(filename):
        logging.info(
            "Getting cached data for \n\tTeam:{}, \n\tYear:{}".format(team_id, year)
        )
        sorted_szn_games = pd.read_csv(filename)

    # get the data if it doesn't already exist
    else:
        logging.info(
            "No cached data for \n\tTeam:{}, \n\tYear:{}".format(team_id, year)
        )
        logging.info("Fetching data...")

        # get all games
        result = leaguegamefinder.LeagueGameFinder()
        all_games: pd.DataFrame = result.get_data_frames()[0]

        # get this seasons' games for this team
        szn_games = all_games.where(all_games.SEASON_ID.astype(int) % 10000 == year)
        team_szn_games = szn_games.loc[szn_games["TEAM_ID"] == team_id]

        # sort by date
        sorted_szn_games = team_szn_games.sort_values("GAME_DATE", ascending=True)
        
        # remove playoff games
        if remove_playoff:
            game_dates: pd.Series[Any] = sorted_szn_games["GAME_DATE"]
            sorted_szn_games = sorted_szn_games.loc[
                game_dates < year_to_reg_season_end[year]  # type: ignore
            ]
            sorted_szn_games = sorted_szn_games.loc[sorted_szn_games['GAME_DATE'] >= year_to_reg_season_start[year]]

        # add 'games played'
        sorted_szn_games["GAMES_PLAYED"] = range(0, sorted_szn_games.shape[0])

        sorted_szn_games.to_csv(filename, index=False)
        logging.info("Fetched and cached.\n")

    return sorted_szn_games


def get_per_game(game_id: int) -> None:
    """
    get averages for all basic stats
    """
    pass


def get_trailing(game_id: int, trail_by: int = 10) -> None:
    """
    get averages for previous X games
    """
    pass


def get_oppt_ppg(team_id: int) -> None:
    """
    opponents ppg before this game
    """


def get_winrate_stats(team_id: int) -> None:
    """ """
    pass


def get_points_by_quarter(team_id: int) -> None:
    """
    both scored and scored on
    """
    pass


def main() -> None:

    year = 2018
    logging.info(f"Getting data for {year}")
    g = get_games_by_year(year, with_opponents=True)
    X, y = make_xy(g)

    # logging.info(X)
    # logging.info("===========")
    # logging.info(y)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
