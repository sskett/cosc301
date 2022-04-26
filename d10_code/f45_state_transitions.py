import pandas as pd
import numpy as np
import math


def analyse_play_data(data):
    # Calculating p_group data (Polarisation)
    data, group_data = calc_play_p_group(data)
    # Calculating m_group data (Angular Momentum)
    data, group_data = calc_play_m_group(data, group_data)
    # Calculating v_group data (Group Speed)
    group_data = calc_play_v_group(data, group_data, 0.1)
    # Calculate h_group data (Shannon Entropy)
    calc_play_h_group(data, group_data)
    print('Calculating group data complete')
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
    group_data.loc[:, 'gameId'] = play_data['gameId'].values[0]
    group_data.loc[:, 'playId'] = play_data['playId'].values[0]
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
    play_data_o, off_mx, off_my = calc_team_m_group('offense')
    play_data_d, def_mx, def_my = calc_team_m_group('defense')

    # Merge dataframes
    play_data = pd.concat([play_data_o, play_data_d], ignore_index=True, sort=False)
    group_data['mx_off'] = pd.Series(off_mx)
    group_data['my_off'] = pd.Series(off_my)
    group_data['mx_def'] = pd.Series(def_mx)
    group_data['my_def'] = pd.Series(def_my)

    # Convert to 2D vectors
    play_data['r_vec'] = play_data.apply(lambda x: np.array([x['rx'], x['ry']]), axis=1)
    play_data.drop(['rx', 'ry'], axis=1, inplace=True)

    group_data['offense_m_group'] = group_data.apply(lambda x: np.array([x['mx_off'], x['my_off']]), axis=1)
    group_data['defense_m_group'] = group_data.apply(lambda x: np.array([x['mx_def'], x['my_def']]), axis=1)
    group_data.drop(['mx_off', 'my_off', 'mx_def', 'my_def'], axis=1, inplace=True)

    # Return values
    return play_data, group_data


def calc_play_v_group(play_data, group_data, time_step):

    def calc_team_v_group(team):
        play_data_t = play_data.loc[(play_data['teamType'] == team)].copy()
        play_data_t['rx'] = 0
        play_data_t['ry'] = 0
        num_frames_t = play_data_t['frameId'].max()

        # Calculate c_group
        c_group_t = np.zeros((1, num_frames_t), dtype=tuple)
        for frame in range(1, num_frames_t + 1):
            c_group = (0, 0)
            num_players = play_data_t['nflId'].nunique()
            for idx, player in play_data_t.loc[(play_data_t['frameId'] == frame)].iterrows():
                c_group = c_group[0] + player.pos[0], c_group[1] + player.pos[1]
            c_group_t[0][frame - 1] = c_group[0] / num_players, c_group[1] / num_players

        # Calculate v_group
        vx = []
        vy = []
        vx.append(0)
        vy.append(0)

        for frame in range(1, num_frames_t):
            vx.append(math.fabs((c_group_t[0][frame][0] - c_group_t[0][frame - 1][0])) / time_step)
            vy.append(math.fabs((c_group_t[0][frame][1] - c_group_t[0][frame - 1][1])) / time_step)

        del play_data_t
        return vx, vy

    # For HOME team
    off_vx, off_vy = calc_team_v_group('offense')

    # For AWAY team
    def_vx, def_vy = calc_team_v_group('defense')

    # Merge dataframes
    group_data['vx_off'] = pd.Series(off_vx)
    group_data['vy_off'] = pd.Series(off_vy)
    group_data['vx_def'] = pd.Series(def_vx)
    group_data['vy_def'] = pd.Series(def_vy)

    # Convert to 2D vectors

    group_data['offense_v_group'] = group_data.apply(lambda x: np.array([x['vx_off'], x['vy_off']]), axis=1)
    group_data['defense_v_group'] = group_data.apply(lambda x: np.array([x['vx_def'], x['vy_def']]), axis=1)
    group_data.drop(['vx_off', 'vy_off', 'vx_def', 'vy_def'], axis=1, inplace=True)

    return group_data


def calc_play_h_group(play_data, group_data):
    # Max speed ~10yd/sec == ~1yd/frame
    # TODO: Modify to use a limited number of frames instead of whole play
    # a) Initialise grid
    grid_size = 64
    step = 0.5
    grid = np.zeros(grid_size * grid_size).reshape((grid_size, grid_size))

    # b) Calculate position vectors
    team = play_data.loc[(play_data['teamType'] == 'offense')]
    for idx, row in team.iterrows():
        i = round((row.dir_vec[0] * row.s) / step) + int(grid_size / 2 - 1)
        j = round((row.dir_vec[1] * row.s) / step) + int(grid_size / 2 - 1)
        grid[i][j] = grid[i][j] + 1

    # c) Estimate probabilities
    p_grid = np.zeros(grid_size * grid_size).reshape((grid_size, grid_size))
    sum_f = grid.sum()
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            p_grid[i][j] = grid[i][j] / sum_f

    # d) Calculate Shannon-Entropy (h)
    h_home = []
    h_grid = np.zeros(grid_size * grid_size).reshape((grid_size, grid_size))
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            if p_grid[i][j] > 0:
                h_grid[i][j] = p_grid[i][j] * math.log(p_grid[i][j], 2)
    h_home.append(-1 * h_grid.sum())
    print(h_home)

    # e) Plot position probabilities
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    from matplotlib import cm
    import scipy.ndimage.filters as filters

    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0), (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]

    cm = LinearSegmentedColormap.from_list('sample', colors)
    plt.imshow(p_grid, cmap=cm)
    plt.colorbar()
    plt.show()

    # 7. Calculate area of convex hulls
    # hull_frame = game_data.loc[(game_data['event'] == 'ball_snapped')]['frameId'].tolist()[0]
    hull_frame = 30
    #los = plays.loc[(plays['playId'] == play)]['absoluteYardlineNumber'].tolist()[0]

    # Home team convex hull
    a_data = play_data.loc[(play_data['teamType'] == 'offense') & (play_data['frameId'] == hull_frame)]
    a_points = np.zeros((len(a_data), 2))
    a_points[:, 0] = np.array(a_data['pos'].tolist())[:, 0]
    a_points[:, 1] = np.array(a_data['pos'].tolist())[:, 1]
    a_hull = ConvexHull(a_points)


    # Away team convex hull
    a2_data = play_data.loc[(play_data['teamType'] == 'defense') & (play_data['frameId'] == hull_frame)]
    a2_points = np.zeros((len(a2_data), 2))
    a2_points[:, 0] = np.array(a2_data['pos'].tolist())[:, 0]
    a2_points[:, 1] = np.array(a2_data['pos'].tolist())[:, 1]

    a2_hull = ConvexHull(a2_points)

    # Plot image of convex hulls for home and away
    a = plt.figure()
    axes = a.add_axes([0, 0, 1, 1])

    axes.set_xlim([0, 120])
    axes.set_ylim([0, 60])

    axes.plot(a_points[:, 0], a_points[:, 1], 'o')
    axes.plot(a2_points[:, 0], a2_points[:, 1], 'x')
    # axes.axvline(x=los)
    for simplex in a_hull.simplices:
        axes.plot(a_points[simplex, 0], a_points[simplex, 1], 'r-')
    for simplex in a2_hull.simplices:
        axes.plot(a2_points[simplex, 0], a2_points[simplex, 1], 'g-')
    a.show()
