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
        players_df = dff.filter_player_data(players_df, options).copy()
        player_ids = players_df['nflId'].unique().tolist()

        games_df = dff.filter_game_data(games_df, options).copy()
        game_ids = games_df['gameId'].unique().tolist()

        plays_df = dff.filter_play_data(plays_df, options, game_ids).copy()

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

        result_files = [
            (tracking_results, 'tracking_results.csv',
             ['time', 's', 'a', 'dis', 'event', 'nflId', 'displayName', 'jerseyNumber', 'position', 'frameId', 'team', 'gameId', 'playId', 'playDirection', 'route', 'pos', 'teamType', 'o_vec', 'dir_vec', 'r_vec']),
            (frame_results, 'frame_results.csv', ['gameId', 'playId', 'frameId', 'offense_p_group', 'defense_p_group', 'offense_m_group', 'defense_m_group', 'o_state', 'd_state', 'offense_v_group', 'defense_v_group', 'offense_a_group', 'defense_a_group', 'a_group_ratio']),
            (play_results, 'play_results.csv', ['gameId', 'playId', 'offense_h_play', 'offense_h_presnap', 'offense_h_to_throw', 'offense_h_to_arrived', 'offense_h_to_end', 'defense_h_play', 'defense_h_presnap', 'defense_h_to_throw', 'defense_h_to_arrived', 'defense_h_to_end', 'offense_p_play', 'offense_p_presnap', 'offense_p_to_throw', 'offense_p_to_arrived', 'offense_p_to_end', 'offense_m_play', 'offense_m_presnap', 'offense_m_to_throw', 'offense_m_to_arrived', 'offense_m_to_end', 'offense_v_play', 'offense_v_presnap', 'offense_v_to_throw', 'offense_v_to_arrived', 'offense_v_to_end', 'offense_a_play', 'offense_a_presnap', 'offense_a_to_throw', 'offense_a_to_arrived','offense_a_to_end', 'defense_p_play', 'defense_p_presnap','defense_p_to_throw', 'defense_p_to_arrived', 'defense_p_to_end','defense_m_play', 'defense_m_presnap', 'defense_m_to_throw','defense_m_to_arrived', 'defense_m_to_end', 'defense_v_play', 'defense_v_presnap', 'defense_v_to_throw', 'defense_v_to_arrived','defense_v_to_end', 'defense_a_play', 'defense_a_presnap','defense_a_to_throw', 'defense_a_to_arrived', 'defense_a_to_end','HITCH', 'OUT', 'FLAT', 'CROSS', 'GO', 'SLANT', 'SCREEN', 'CORNER', 'IN', 'ANGLE', 'POST', 'WHEEL'])]

        # TODO: Change method to write header only if file doesn't exist (avoid issue of empty file on first write) - needs to be in DF returned from ray.futures
        for file in result_files:
            path = 'd20_intermediate_files/' + file[1]
            if os.path.exists(path):
                print(f'File {file[1]} exists')
                file[0].to_csv(path, mode='a', index=False, header=False)
            elif file[0].shape[0] > 0:
                print(f'File {file[1]} does not exist, writing data to new file.')
                file[0].to_csv(path, mode='w', index=False, header=True)
            else:
                print(f'File {file[1]} does not exist, creating header file.')
                df = pd.DataFrame(columns=file[2])
                df.to_csv(path, mode='w', index=False, header=True)

    finish_time = time.time()
    print(f'Completed in {finish_time - start_time} seconds.')
    # Combine analysis data
    # Select training/test sets // Not required as unsupervised learning used to cluster plays?
    # Do stuff...
    pass
