import pandas as pd
import ray
from .f10_import_data import import_tracking_data
from .f20_filter_data import filter_tracking_data
from .f30_clean_data import clean_tracking_data
from .f41_transform_data import transform_tracking_data
from .f45_state_transitions import analyse_play_data_state_transition


def define_pandas_options():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


@ray.remote
def analyse_play(play_id, plays_df, games_df, players_df, source_dir):
    print(f'Analysing play {play_id}')
    define_pandas_options()

    game_id = int(play_id.split('-')[0])
    play_id = int(play_id.split('-')[1])
    week = games_df.loc[games_df['gameId'] == game_id]['week'].tolist()[0]

    o_team = plays_df.loc[(plays_df['playId'] == play_id)]['possessionTeam'].values[0]
    o_team_type = 'home' if games_df['homeTeamAbbr'][0] == o_team else 'away'

    tracking_df = import_tracking_data(source_dir, week, game_id, play_id)
    tracking_df = filter_tracking_data(tracking_df, game_id, play_id)
    tracking_df = clean_tracking_data(tracking_df)
    tracking_df = transform_tracking_data(tracking_df, o_team_type)
    tracking_df, group_df, summary_df = analyse_play_data_state_transition(tracking_df)

    return {game_id: {play_id: [tracking_df, group_df, summary_df]}}
