import logging
import os
from typing import Any, Tuple

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

"""
STATS:
basic stats: drb, orb, ... fta, .etc.
winrate stats:
"""


year_to_playoff_start = {
    # https://en.wikipedia.org/wiki/2018_NBA_playoffs
    # 2018: "2018-04-14",  # do not use
    2018: "2019-04-09",
    # https://en.wikipedia.org/wiki/2019_NBA_playoffs
    # 2019: "2019-04-13",  # do not use
    2019: "2018-04-14",  # TODO: @dankrasno needs to be adjusted
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


def get_games_by_year(year: int) -> pd.DataFrame:
    """
    Get all games for a season
    """

    nba_teams = [t["id"] for t in teams.get_teams()]

    games = [get_games_by_team_szn(year, team_id) for team_id in nba_teams]
    all_game_stats = pd.concat(games[:])

    # result = leaguegamefinder.LeagueGameFinder()
    # all_games = result.get_data_frames()[0]
    # # adjust for year
    # all_games = all_games[all_games.SEASON_ID.str[-4:] == year]
    # all_games = all_games[all_games.TEAM_ABBREVIATION.isin(nba_teams)]

    # # only teams in nba
    # logging.info(all_games.shape)
    return all_game_stats


def make_xy(df: pd.DataFrame) -> "Tuple[pd.DataFrame, pd.Series[str]]":
    """
    Drop columns we won't need
    """
    X = df.drop(descriptive_cols, axis=1)
    y = df["WL"]
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
        # add 'games played'
        sorted_szn_games["GAMES_PLAYED"] = range(0, sorted_szn_games.shape[0])

        # remove playoff games
        if remove_playoff:
            game_dates: pd.Series[Any] = sorted_szn_games["GAME_DATE"]
            sorted_szn_games = sorted_szn_games.loc[
                game_dates < year_to_playoff_start[year]  # type: ignore
            ]

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
    g = get_games_by_year(year)
    X, y = make_xy(g)

    logging.info(X)
    logging.info("===========")
    logging.info(y)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
