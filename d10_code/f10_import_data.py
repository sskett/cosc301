import pandas as pd


def import_player_data(filename):
    # print('Reading in player data')
    return pd.read_csv(filename)


def import_game_data(filename):
    # print('Reading in game data')
    return pd.read_csv(filename)


def import_play_data(filename):
    # print('Reading in play data')
    return pd.read_csv(filename)


def import_tracking_data(folder, week, game_id, play_id):
    filename = folder + 'week' + str(week) + '.csv'
    # print(f'Reading in tracking data for {game_id} - {play_id} (Week: {week})')
    return pd.read_csv(filename)

