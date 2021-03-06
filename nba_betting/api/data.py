import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

from nba_betting.logging.tools import logger

year_to_reg_season_start = {
    2017: "2017-10-17",
    2018: "2018-10-16",
    2019: "2019-10-22",
}

year_to_reg_season_end = {
    2017: "2018-04-11",
    2018: "2019-04-10",
    2019: "2020-04-14",
}

DESCRIPTIVE_COLS = [
    "SEASON_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
    "ALLOWED_SEASON_ID",
    "ALLOWED_TEAM_ID",
    "ALLOWED_TEAM_NAME",
    "ALLOWED_GAME_ID",
    "ALLOWED_MATCHUP",
    "ALLOWED_MIN",
]
OVERLAP_COLS = ["GAME_DATE", "OPPONENT", "TEAM_ABBREVIATION", "WL"]
OVERLAP_COLS2 = ["GAME_DATE", "TEAM_ABBREVIATION", "WL"]
TEAM_FEATURE_COLS = [
    "PTS",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PLUS_MINUS",
    "GAMES_PLAYED",
]
FEATURE_COLS = TEAM_FEATURE_COLS + [f"OPP_{col}" for col in TEAM_FEATURE_COLS]
TEST_COL = "WL"


def get_team_id_from_abbr(team_abbr: str) -> int:

    nba_teams = teams.get_teams()
    # Select the dictionary for the Celtics, which contains their team ID
    team = [team for team in nba_teams if team["abbreviation"] == team_abbr][0]
    return int(team["id"])


def get_games_by_year(year: int) -> pd.DataFrame:
    """
    Get all games for a season
    """

    nba_teams = [t["id"] for t in teams.get_teams()]
    logger.debug("nba_teams: %r", nba_teams)
    team_abbr = [t["abbreviation"] for t in teams.get_teams()]
    logger.debug("team_abbr: %r", team_abbr)

    logging.info("Getting game data...")
    # games by team
    games_dict_no_opp = {}
    games = [get_games_by_team_szn(year, team_id) for team_id in nba_teams]
    logger.debug("got games: %r", games)
    for team_id, g in zip(team_abbr, games):
        logger.debug("team_id, g: %r, %r", team_id, g.shape)
        games_dict_no_opp[team_id] = g

    logger.debug("games_dict_no_opp.keys(): %r", list(games_dict_no_opp.keys()))

    games_with_opponents = add_opponents(games_dict_no_opp, is_cumulative=False)

    games_dict = {}
    for game_with_opponents in games_with_opponents.values():
        team_name, cumulative_df = make_cumulative(game_with_opponents)
        games_dict[team_name] = cumulative_df
    logger.debug("Done.")

    logging.info("Creating matchups...")
    games_with_ops = add_opponents(games_dict, is_cumulative=True)
    logging.info("Done creating matchups.")

    full_year_df = pd.concat([v for v in games_with_ops.values()])
    full_year_df.dropna(inplace=True, subset=[TEST_COL], how="all")
    full_year_df.reset_index(inplace=True)

    logger.debug(
        "Created data for %s:\n%r",
        year,
        full_year_df,
    )

    return full_year_df


def add_opponents(
    games_dict: Dict[str, pd.DataFrame], is_cumulative: bool
) -> Dict[str, pd.DataFrame]:

    logger.debug(games_dict)

    if is_cumulative:
        prefix = "OPP_"
        drop_cols = OVERLAP_COLS
    else:
        prefix = "ALLOWED_"
        drop_cols = OVERLAP_COLS2
    with_opps: Dict[str, pd.DataFrame] = {}
    for key in games_dict:
        sub_df = games_dict[key]
        sub_dict = sub_df.to_dict("records")
        team_opp_dict = {}
        col_names: List[str] = []

        for i, row in enumerate(sub_dict):

            # find opponent (abbreviation), find game date
            if is_cumulative:
                opponent = row["OPPONENT"]
            else:
                opponent = row["MATCHUP"][-3:]
            game_date = row["GAME_DATE"]

            row2 = pd.DataFrame(row, index=[0])
            opponent_df = games_dict[opponent]

            # get opponent cumulative stats
            # using game date, validate current team abbreviation
            opponent_row = (
                opponent_df.loc[opponent_df["GAME_DATE"] == game_date]
                .drop(drop_cols, axis=1)
                .add_prefix(prefix)
                # opponent_df.loc[opponent_df["GAME_DATE"] == game_date][
                #     TEAM_FEATURE_COLS
                # ]
                # .add_prefix("OPP_")
                .set_index(pd.Index([0]))
            )
            # opp_cols = opponent_row

            # combine stats into 1 row
            combined = pd.concat([row2, opponent_row], axis=1).iloc[0]
            # add to new dict
            team_opp_dict[i] = combined.tolist()

            if i == 0:
                col_names = list(row2.columns) + list(opponent_row.columns)

        opp_df: pd.DataFrame = pd.DataFrame.from_dict(  # type: ignore[attr-defined]
            team_opp_dict, orient="index"
        )

        assert len(col_names) > 0
        opp_df.columns = col_names  # type: ignore[assignment]
        with_opps[key] = opp_df

    return with_opps


def make_cumulative(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:

    # setup
    dates, matchup, team_name = df["GAME_DATE"], df["MATCHUP"], df["TEAM_ABBREVIATION"]
    opponent_abv = [str(m).strip()[-3:] for m in matchup]
    wl = df[TEST_COL]
    df = df.drop(DESCRIPTIVE_COLS, axis=1)
    dictionary_data = {}

    # get previous rows, get averages and append to dict
    for i in range(df.shape[0]):
        preceding_rows = df.iloc[:i]
        avg_til_now = list(preceding_rows.mean())
        if np.isnan(avg_til_now).any():
            avg_til_now = [0] * len(avg_til_now)
        dictionary_data[i] = avg_til_now

    # create and fix up final df
    df_final = pd.DataFrame.from_dict(  # type: ignore[attr-defined]
        dictionary_data, orient="index"
    )
    df_final.columns = df.columns
    df_final["GAMES_PLAYED"] = df["GAMES_PLAYED"]
    df_final["GAME_DATE"] = dates
    df_final["OPPONENT"] = opponent_abv
    df_final["TEAM_ABBREVIATION"] = team_name
    df_final[TEST_COL] = wl

    return str(team_name.iloc[0]), df_final


def make_xy(df: pd.DataFrame) -> "Tuple[pd.DataFrame, pd.Series[str]]":
    """
    Drop columns we won't need
    """

    y = df[TEST_COL]
    X = df.drop(OVERLAP_COLS, axis=1)
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
    logger.debug("filename: %r", filename)
    if os.path.exists(filename):
        # logger.debug(
        #     "Getting cached data for \n\tTeam:{}, \n\tYear:{}".format(team_id, year)
        # )
        sorted_szn_games = pd.read_csv(filename)
        logger.debug(sorted_szn_games)

    # get the data if it doesn't already exist
    else:
        logger.debug(
            "No cached data for \n\tTeam:{}, \n\tYear:{}".format(team_id, year)
        )
        logger.debug("Fetching data...")

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
            sorted_szn_games = sorted_szn_games.loc[
                sorted_szn_games["GAME_DATE"]
                < year_to_reg_season_end[year]  # type: ignore[operator]
            ]
            sorted_szn_games = sorted_szn_games.loc[
                sorted_szn_games["GAME_DATE"]
                >= year_to_reg_season_start[year]  # type: ignore[operator]
            ]

        # add 'games played'
        sorted_szn_games["GAMES_PLAYED"] = range(0, sorted_szn_games.shape[0])

        sorted_szn_games.to_csv(filename, index=False)
        logger.debug("Fetched and cached.\n")

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
    logger.info(f"Getting data for {year}")
    g = get_games_by_year(year)
    X, y = make_xy(g)

    logger.info(X)
    logger.info("===========")
    logger.info(y)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    main()
