import pandas as pd
import numpy as np
import math


def analyse_play_data(data):
    # print('Calculating p_group data')
    data, group_data = calc_play_p_group(data)
    # print('Calculating m_group data')
    #data, group_data = calc_play_m_group(data, group_data)
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

    # 1. Determine polarisation for home and away teams
    o_data, o_p_group = calc_team_p_group('offense')
    d_data, d_p_group = calc_team_p_group('defense')

    # 2. Return values
    num_frames = max(len(o_p_group), len(d_p_group))
    group_data['frameId'] = pd.Series(np.arange(1, num_frames + 1))
    group_data['gameId'] = pd.Series(play_data['gameId'])
    group_data['playId'] = pd.Series(play_data['playId'])
    group_data = group_data.reindex(columns=['gameId', 'playId', 'frameId'])
    group_data['offense_p_group'] = pd.Series(o_p_group)
    group_data['defense_p_group'] = pd.Series(d_p_group)
    return play_data, group_data
