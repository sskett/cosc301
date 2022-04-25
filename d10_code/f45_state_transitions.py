import pandas as pd
import numpy as np
import math


def analyse_play_data(data):
    # print('Calculating p_group data')
    data, group_data = calc_play_p_group(data)
    # print('Calculating m_group data')
    data, group_data = calc_play_m_group(data, group_data)
    # print('Calculating v_group data')
    #group_data = calc_play_v_group(data, group_data, 0.1)
    # print('Calculating group data complete')
    return data, group_data


def calc_play_p_group(play_data, group_data=pd.DataFrame()):

    def calc_team_p_group(team):
        team_data = play_data.loc[(play_data['teamType'] == team)]
        p_group = []

        # print('Calculating polarisation of the home team')
        for frame in range(1, team_data['frameId'].max() + 1):
            n_players = team_data['nflId'].nunique()
            p_group.append(np.abs(team_data.loc[(team_data['frameId'] == frame)]['dir_vec'].sum()) / n_players)
        return team_data, p_group

    # 1. Initialise results dataframe
    num_frames = play_data['frameId'].max()
    group_data['frameId'] = pd.Series(np.arange(1, num_frames + 1))
    group_data['gameId'] = pd.Series(play_data['gameId'])
    group_data['playId'] = pd.Series(play_data['playId'])
    group_data = group_data.reindex(columns=['gameId', 'playId', 'frameId'])

    # 2. Determine polarisation for home and away teams
    o_data, o_p_group = calc_team_p_group('offense')
    d_data, d_p_group = calc_team_p_group('defense')

    # 3. Return values
    group_data['offense_p_group'] = pd.Series(o_p_group)
    group_data['defense_p_group'] = pd.Series(d_p_group)

    return play_data, group_data


def calc_play_m_group(play_data, group_data):

    def calc_team_m_group(team):
        play_data_t = play_data.loc[(play_data['teamType'] == team)].copy()
        play_data_t['rx'] = 0
        play_data_t['ry'] = 0
        num_frames = play_data_t['frameId'].max()
        c_group_t = np.zeros((1, num_frames), dtype=tuple)

        # Calculate c_group
        for frame in range(1, num_frames + 1):
            c_group = (0, 0)
            num_players = play_data_t['nflId'].nunique()
            for idx, player in play_data_t.loc[(play_data_t['frameId'] == frame)].iterrows():
                c_group = c_group[0] + player.pos[0], c_group[1] + player.pos[1]
            c_group_t[0][frame - 1] = c_group[0] / num_players, c_group[1] / num_players

        # Calculate vectors to centres (r)
        for frame in range(1, num_frames + 1):
            c_group_r = c_group_t[0][frame - 1]
            for idx, player in play_data_t.loc[(play_data_t['frameId'] == frame)].iterrows():
                r = player.pos[0] - c_group_r[0], player.pos[1] - c_group_r[1]
                r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2)
                r = r[0] / r_mag, r[1] / r_mag
                play_data_t.at[idx, 'rx'] = r[0]
                play_data_t.at[idx, 'ry'] = r[1]

        # Calculate m group
        mx = np.zeros(num_frames)
        my = np.zeros(num_frames)
        for frame in range(1, num_frames + 1):
            m = (0, 0)
            num_players = play_data_t['nflId'].nunique()
            for idx, player in play_data_t.loc[(play_data_t['frameId'] == frame)].iterrows():
                m = m[0] + player.rx * player.dir_vec[0], m[1] + player.ry * player.dir_vec[1]
            m = m[0] / num_players, m[1] / num_players
            mx[frame - 1] = m[0]
            my[frame - 1] = m[1]
        return play_data_t, mx, my

    # Analyse home and away teams
    o_data, m_home_x, m_home_y = calc_team_m_group('offense')
    d_data, m_away_x, m_away_y = calc_team_m_group('defense')

    # Merge dataframes
    play_data = pd.concat([o_data, d_data], ignore_index=True, sort=False)
    group_data['mx_home'] = pd.Series(m_home_x)
    group_data['my_home'] = pd.Series(m_home_y)
    group_data['mx_away'] = pd.Series(m_away_x)
    group_data['my_away'] = pd.Series(m_away_y)

    # Return values
    return play_data, group_data
