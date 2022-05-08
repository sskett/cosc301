import pandas as pd
import ray
import time

from d10_code import f10_import_data as dfi
from d10_code import f20_filter_data as dff
from d10_code import f30_clean_data as dfc
from d10_code import f40_analyse_data as dfa


def define_pandas_options():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


if __name__ == '__main__':
    start_time = time.time()
    # Set options
    define_pandas_options()
    # TODO: Fix 2018092301-453(3)
    # TODO: Add step to assess tracking data for missing/incomplete data and list games/plays to exclude
    options = {
        # no errors in 'weeks_to_import': [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17],
        'weeks_to_import': [1],
        'players_to_import_by_id': [],
        'players_to_import_by_name': [],
        'games_to_import': [],
        'teams_to_import': [],
        'team_type_to_import': ['home', 'away', 'football'],
        'plays_to_import': [],
        'directions_to_select': ['left', 'right'],
        'routes_to_select': []
    }

    # Import generic data
    source_folder = './d00_source_data/'
    players_df = dfi.import_player_data(source_folder + 'players.csv')
    games_df = dfi.import_game_data(source_folder + 'games.csv')
    plays_df = dfi.import_play_data(source_folder + 'plays.csv')

    # Filter data to match options
    players_df = dff.filter_player_data(players_df, options)
    player_ids = players_df['nflId'].unique().tolist()

    games_df = dff.filter_game_data(games_df, options)
    game_ids = games_df['gameId'].unique().tolist()

    plays_df = dff.filter_play_data(plays_df, options, game_ids)

    # Clean data
    players_df = dfc.clean_player_data(players_df)
    games_df = dfc.clean_game_data(games_df)
    plays_df = dfc.clean_play_data(plays_df)
    play_ids = plays_df['gpid'].unique().tolist()

    # Start multiprocessing pool
    ray.init()

    # Process selected games
    futures = [dfa.analyse_play.remote(play, plays_df, games_df, players_df, source_folder) for play in play_ids]
    results = ray.get(futures)

    # Collate results
    tracking_results = pd.DataFrame()
    frame_results = pd.DataFrame()
    play_results = pd.DataFrame()

    for result_idx in results:
        for game_idx in result_idx.keys():
            for play_idx in result_idx[game_idx].keys():
                tracking_results = pd.concat([tracking_results, result_idx[game_idx][play_idx][0]])
                frame_results = pd.concat([frame_results, result_idx[game_idx][play_idx][1]])
                play_results = pd.concat([play_results, result_idx[game_idx][play_idx][2]])
    del results

    finish_time = time.time()
    print(f'Completed in {finish_time - start_time} seconds.')
    # Combine analysis data
    # Select training/test sets
    # Do stuff...
    pass
