from data import get_games_by_year, make_xy


if __name__ == "__main__":
    year = '2018'
    g = get_games_by_year(year, with_opponents=True)
    # X, y = make_xy(g)

