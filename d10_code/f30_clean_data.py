import numpy as np
import pandas as pd


def clean_player_data(df):
    def convert_height(height):
        ht = height.split('-')
        if len(ht) == 2:
            return int(ht[0]) * 12 + int(ht[1])
        else:
            return int(height)

    df['height'] = df['height'].apply(convert_height)
    df['birthDate'] = df['birthDate'].apply(pd.to_datetime)
    df.set_index('nflId')
    return df


def clean_game_data(df):
    # TODO: basic data integrity checks and cleanup
    df['gameDate'] = df['gameDate'].apply(pd.to_datetime)
    df.set_index('gameId')
    return df


def clean_play_data(df):
    def determine_play_type(play):
        if play == 'S':
            return 'play_type_sack'
        else:
            return 'play_type_pass'

    df['playType'] = df['passResult'].apply(determine_play_type)

    # Convert null values to possession team where ball is at mid-field
    df.yardlineSide.where((~df.yardlineSide.isnull()), df.possessionTeam, inplace=True)

    # Convert forced floats to ints
    df['defendersInTheBox'] = df['defendersInTheBox'].astype('Int64')
    df['numberOfPassRushers'] = df['numberOfPassRushers'].astype('Int64')
    df['preSnapVisitorScore'] = df['preSnapVisitorScore'].astype('Int64')
    df['preSnapHomeScore'] = df['preSnapHomeScore'].astype('Int64')
    df['absoluteYardlineNumber'] = df['absoluteYardlineNumber'].astype('Int64')

    # Drop rows with missing data or penalties
    df = df[df['penaltyCodes'].isnull()]
    df = df[~df['offenseFormation'].isnull()]

    # Add unique index
    df['gpid'] = df['gameId'].astype(str) + '-' + df['playId'].astype(str)
    df.set_index('gpid')
    return df


def clean_tracking_data(df):
    # Convert time strings to datetime
    df['time'] = df['time'].apply(pd.to_datetime)

    # Convert x,y cols to vectors
    df['pos'] = df.apply(lambda x: np.array([x['x'], x['y']]), axis=1)
    df.drop(['x', 'y'], axis=1, inplace=True)

    # Convert nan forced floats to ints and NA
    df['nflId'] = df['nflId'].astype('Int64')
    df['jerseyNumber'] = df['jerseyNumber'].astype('Int64')

    return df
