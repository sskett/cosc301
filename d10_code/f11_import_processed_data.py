import pandas as pd


def import_processed_play_data(filename):
    play_df_cols = ['gameId', 'playId', 'offense_h_play', 'offense_h_presnap', 'offense_h_to_throw',
                    'offense_h_to_arrived', 'offense_h_to_end', 'defense_h_play', 'defense_h_presnap',
                    'defense_h_to_throw', 'defense_h_to_arrived', 'defense_h_to_end', 'offense_p_play',
                    'offense_p_presnap', 'offense_p_to_throw', 'offense_p_to_arrived', 'offense_p_to_end',
                    'offense_m_play', 'offense_m_presnap', 'offense_m_to_throw', 'offense_m_to_arrived',
                    'offense_m_to_end', 'offense_v_play', 'offense_v_presnap', 'offense_v_to_throw',
                    'offense_v_to_arrived', 'offense_v_to_end', 'offense_a_play', 'offense_a_presnap',
                    'offense_a_to_throw', 'offense_a_to_arrived', 'offense_a_to_end', 'defense_p_play',
                    'defense_p_presnap', 'defense_p_to_throw', 'defense_p_to_arrived', 'defense_p_to_end',
                    'defense_m_play', 'defense_m_presnap', 'defense_m_to_throw', 'defense_m_to_arrived',
                    'defense_m_to_end', 'defense_v_play', 'defense_v_presnap', 'defense_v_to_throw',
                    'defense_v_to_arrived', 'defense_v_to_end', 'defense_a_play', 'defense_a_presnap',
                    'defense_a_to_throw', 'defense_a_to_arrived', 'defense_a_to_end', 'HITCH', 'OUT', 'FLAT', 'CROSS',
                    'GO', 'SLANT', 'SCREEN', 'CORNER', 'IN', 'ANGLE', 'POST', 'WHEEL']

    play_df = pd.read_csv(filename, usecols=play_df_cols)
    play_df['num_routes'] = play_df[['HITCH', 'OUT', 'FLAT', 'CROSS', 'GO', 'SLANT', 'SCREEN', 'CORNER', 'IN', 'ANGLE', 'POST', 'WHEEL']].T.sum()
    play_df.drop(play_df[play_df['num_routes'] == 0].index, inplace=True)
    play_df.dropna(inplace=True)
    return play_df


def import_processed_frame_data(filename):
    frame_df_cols = ['gameId', 'playId', 'frameId', 'offense_p_group', 'defense_p_group', 'offense_m_group',
                     'defense_m_group', 'o_state', 'd_state', 'offense_v_group', 'defense_v_group', 'offense_a_group',
                     'defense_a_group', 'a_group_ratio']

    frame_df = pd.read_csv(filename, usecols=frame_df_cols)
    return frame_df


def import_processed_tracking_data(filename):
    tracking_df_cols = ['time', 's', 'a', 'dis', 'event', 'nflId', 'displayName', 'jerseyNumber', 'position', 'frameId',
                        'team', 'gameId', 'playId', 'playDirection', 'route', 'pos', 'teamType', 'o_vec', 'dir_vec',
                        'r_vec']

    data = pd.read_csv(filename, usecols=tracking_df_cols, chunksize=100000)
    return pd.concat(data)



