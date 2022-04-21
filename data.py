# from cmath import nan
# from copyreg import remove_extension
from re import I
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

'''
STATS:
basic stats: drb, orb, ... fta, .etc.
winrate stats: 
'''
year_to_reg_season_start ={
    '2018': '2018-10-16',
    '2019': '2019-10-22',
}

year_to_reg_season_end ={
    '2018': '2019-04-10',
    '2019': '2018-04-14',
}

# descriptive_cols = ['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'WL', 'MIN']
descriptive_cols = ['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN']

def get_team_id_from_abbr(team_abbr):

    nba_teams = teams.get_teams()
    # Select the dictionary for the Celtics, which contains their team ID
    team = [team for team in nba_teams if team['abbreviation'] == team_abbr][0]
    team_id = team['id']

    return team_id

def get_games_by_team_and_year(team_id, year, drop_extra=False):
    '''
    year: 2018 for the 2017-18 season
    '''

    # Query for games where the Celtics were playing
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
    # The first DataFrame of those returned is what we want.
    games = gamefinder.get_data_frames()[0]
    # adjust for year
    games = games[games.SEASON_ID.str[-4:] == year]

    if drop_extra:
        games = games.drop(descriptive_cols, axis=1)
    
    return games

def get_games_by_year(year, with_opponents=False):
    '''
    Get all games for a season
    '''

    nba_teams = [t['id'] for t in teams.get_teams()]    
    
    # games by team
    games_dict = {}
    games = [get_games_by_team_szn(year, team_id) for team_id in nba_teams]
    for g in games:
        team, games = make_cumulative(g)
        games_dict[team] = games

    full_data = None
    if with_opponents:
        full_data = add_opponents(games_dict, nba_teams)

    return full_data

def add_opponents(games_dict, teams):

    with_opps = {}

    for key in games_dict:
        print(key)
        sub_df = games_dict[key]
        sub_dict = sub_df.to_dict('records')
        # for each row in subdf
        for row in sub_dict:
            # print(row)
            # print(type(row))

            # find opponent (abbreviation), find game date
            opponent = row['OPPONENT']
            game_date = row['GAME_DATE']

            row = pd.DataFrame.from_dict(row)
            print(row)

            # get opponent df
            opponent_df = games_dict[opponent]

            # get opponent cumulative stats
            # using game date, validate current team abbreviation
            opponent_row = opponent_df.loc[opponent_df['GAME_DATE'] == game_date]
            print(opponent_row)
            print(type(opponent_row))

            # combine stats into 1 row

            # add to new ditct 


            print('=========')



    # make df from dict
    with_opps = pd.DataFrame.from_dict(with_opps, orient='index')

    return with_opps


def make_cumulative(df):

    # setup
    dates, matchup, team_name = df["GAME_DATE"], df["MATCHUP"], df["TEAM_ABBREVIATION"]
    matchup = [m[-3:] for m in matchup]
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

    return team_name.iloc[0], df_final

def make_xy(df):
    '''
    Drop columns we won't need
    '''
    X = df.drop(descriptive_cols, axis=1)
    y = df['WL']
    return X, y

def games_team_szn_filename(year, team_id, remove_playoff=True):
    if remove_playoff: rp = 'noplayoff' 
    else: rp = 'yesplayoff'
    return str(year) + "_" + str(team_id) + "_" + rp + '.csv' 

def get_games_by_team_szn(year, team_id, remove_playoff=True):
    '''
    number of games played this season before current game
    team_id required because each game has 2 rows, one for each team
    '''
    # get filename, make data folders
    filename = games_team_szn_filename(year, team_id, remove_playoff)
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    filename = os.path.join('data/raw', filename)

    # check cached
    if os.path.exists(filename):
        print('Getting cached data for \n\tTeam:{}, \n\tYear:{}'.format(team_id, year))
        sorted_szn_games = pd.read_csv(filename)

    # get the data if it doesn't already exist
    else:
        print('No cached data for \n\tTeam:{}, \n\tYear:{}'.format(team_id, year))
        print('Fetching data...')

        # get all games
        result = leaguegamefinder.LeagueGameFinder()
        all_games = result.get_data_frames()[0]

        # get this seasons' games for this team
        szn_games = all_games[all_games.SEASON_ID.str[-4:] == year]
        team_szn_games = szn_games.loc[szn_games['TEAM_ID'] == team_id]

        # sort by date
        sorted_szn_games = team_szn_games.sort_values('GAME_DATE', ascending=True)

        # remove playoff games and preseason
        if remove_playoff:
            print(year_to_reg_season_start[year], year_to_reg_season_end[year])
            
            sorted_szn_games = sorted_szn_games.loc[sorted_szn_games['GAME_DATE'] <= year_to_reg_season_end[year]]
            sorted_szn_games = sorted_szn_games.loc[sorted_szn_games['GAME_DATE'] >= year_to_reg_season_start[year]]
        
        # add 'games played'
        sorted_szn_games['GAMES_PLAYED'] = range(0, sorted_szn_games.shape[0])

        sorted_szn_games.to_csv(filename, index=False)
        print('Fetched and cached.\n')

    # print('xasgasgasf')
    # print(sorted_szn_games)
    return sorted_szn_games



def get_per_game(game_id):
    '''
    get averages for all basic stats 
    '''
    pass

def get_trailing(game_id, trail_by=10):
    '''
    get averages for previous X games
    '''
    pass

def get_oppt_ppg(team_id):
    '''
    opponents ppg before this game
    '''

def get_winrate_stats(team_id):
    '''

    '''
    pass

def get_points_by_quarter(team_id):
    '''
    both scored and scored on
    '''
    pass


def main():

    year = '2018'
    g = get_games_by_year(year)

main()