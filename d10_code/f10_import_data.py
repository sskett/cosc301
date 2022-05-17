import pandas as pd
import numpy as np


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
    df = pd.read_csv(filename)
    if df.shape[0] > 0:
        return df
    else:

        return pd.DataFrame(columns=cols)



def string_to_vector(s):
    s = s.split('[')[1].split(']')[0]
    x = float(s.split()[0])
    y = float(s.split()[1])
    return np.array([x, y])



