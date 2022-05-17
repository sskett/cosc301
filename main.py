import pandas as pd
import ray
import time
import os.path

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
        'weeks_to_import': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        'players_to_import_by_id': [],
        'players_to_import_by_name': [],
        'games_to_import': [],
        'teams_to_import': ['ATL'],
        'team_type_to_import': ['home', 'away', 'football'],
        'plays_to_import': [116],
        'directions_to_select': ['left', 'right'],
        'routes_to_select': []
    }
    total_weeks = options['weeks_to_import']

    # Start multiprocessing pool
    ray.init()

    for week in total_weeks:
        options['weeks_to_import'] = [week]
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

        result_files = [(tracking_results, 'tracking_results.csv'), (frame_results, 'frame_results.csv'), (play_results, 'play_results.csv')]

        # TODO: Change method to write header only if file doesn't exist (avoid issue of empty file on first write) - needs to be in DF returned from ray.futures
        for file in result_files:
            path = 'd20_intermediate_files/' + file[1]
            if os.path.exists(path):
                print(f'File {file[1]} exists')
                file[0].to_csv(path, mode='a', sep=';', index=False, header=False)
            else:
                print(f'File {file[1]} does not exist')
                file[0].to_csv(path, mode='w', sep=';', index=False, header=True)

    finish_time = time.time()
    print(f'Completed in {finish_time - start_time} seconds.')
    # Combine analysis data
    # Select training/test sets // Not required as unsupervised learning used to cluster plays?
    # Do stuff...
    pass
