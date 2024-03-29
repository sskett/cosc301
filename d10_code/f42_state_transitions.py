import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import ConvexHull


def analyse_play_data_state_transition(data):
    # Calculating p_group data (Polarisation)
    data, group_data = calc_play_p_group(data)
    # Calculating m_group data (Angular Momentum)
    data, group_data = calc_play_m_group(data, group_data)
    # Determine transition state from p and m values
    group_data = calc_transition_state(group_data)
    # Calculating v_group data (Group Speed)
    group_data = calc_play_v_group(data, group_data, 0.1)
    # Calculate h_group data (Shannon Entropy)
    group_data_summary = calc_play_h_group(data, group_data)
    # Calculate A_group data (Convex hulls)
    group_data = calc_play_a_group(data, group_data)
    # Summarise play data
    group_data_summary = summarise_play_data(data, group_data, group_data_summary)

    return data, group_data, group_data_summary


def calc_play_p_group(play_data, group_data=pd.DataFrame()):

    def calc_team_p_group(team):
        team_data = play_data.loc[(play_data['teamType'] == team)]
        p_group = []
        # print('Calculating polarisation of the home team')
        for frame in range(1, team_data['frameId'].max() + 1):
            n_players = team_data['nflId'].nunique()
            v_sum = team_data.loc[(team_data['frameId'] == frame)]['dir_vec'].sum()
            p_group.append((np.sqrt(v_sum.dot(v_sum))) / n_players)
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
        # play_data_t['rx'] = 0
        # play_data_t['ry'] = 0
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
        mt = np.zeros(num_frames)
        for frame in range(1, num_frames + 1):
            m = (0, 0)
            num_players = play_data_t['nflId'].nunique()
            for idx, player in play_data_t.loc[(play_data_t['frameId'] == frame)].iterrows():
                m = m[0] + player.rx * player.dir_vec[0], m[1] + player.ry * player.dir_vec[1]
            m = math.sqrt(m[0] ** 2 + m[1] ** 2) / num_players
            mt[frame - 1] = m
        return play_data_t, mt

    # Analyse home and away teams
    play_data['rx'] = 0
    play_data['ry'] = 0

    play_data_f = play_data.loc[(play_data['teamType'] == 'ball')].copy()
    play_data_o, o_m_group = calc_team_m_group('offense')
    play_data_d, d_m_group = calc_team_m_group('defense')

    # Merge dataframes
    play_data = pd.concat([play_data_o, play_data_d, play_data_f], ignore_index=True, sort=False)

    # Convert to 2D vectors
    play_data['r_vec'] = play_data.apply(lambda x: np.array([x['rx'], x['ry']]), axis=1)
    play_data.drop(['rx', 'ry'], axis=1, inplace=True)

    # Return values
    group_data['offense_m_group'] = pd.Series(o_m_group)
    group_data['defense_m_group'] = pd.Series(d_m_group)

    return play_data, group_data


def calc_transition_state(group_data):
    group_data['o_state'] = group_data.apply(lambda x: 'polar' if (x['offense_p_group'] > 0.65 and x['offense_m_group'] < 0.35) else 'swarm' if (x['offense_p_group'] < 0.35 and x['offense_m_group'] < 0.35) else 'milling' if (x['offense_p_group'] < 0.35 and x['offense_m_group'] > 0.65) else 'transitional', axis=1)
    group_data['d_state'] = group_data.apply(lambda x: 'polar' if (x['defense_p_group'] > 0.65 and x['defense_m_group'] < 0.35) else 'swarm' if (x['defense_p_group'] < 0.35 and x['defense_m_group'] < 0.35) else 'milling' if (x['defense_p_group'] < 0.35 and x['defense_m_group'] > 0.65) else 'transitional', axis=1)

    return group_data


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
        v = []
        v.append(0)

        for frame in range(1, num_frames_t):
            v.append(math.sqrt((((c_group_t[0][frame][0] - c_group_t[0][frame - 1][0]) ** 2) + (c_group_t[0][frame][1] - c_group_t[0][frame - 1][1]) ** 2) / time_step))

        del play_data_t
        return v

    # For HOME team
    off_v = calc_team_v_group('offense')

    # For AWAY team
    def_v = calc_team_v_group('defense')

    # Merge dataframes
    group_data['offense_v_group'] = pd.Series(off_v)
    group_data['defense_v_group'] = pd.Series(def_v)

    return group_data


def calc_play_h_group(play_data, group_data):
    # Max speed ~10yd/sec == ~1yd/frame
    # TODO: Modify to use a limited number of frames instead of whole play
    def calc_team_h_group(team, start_frame=play_data['frameId'].min(), end_frame=play_data['frameId'].max()):

        def plot_heatmap():
            colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0), (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]
            cm = LinearSegmentedColormap.from_list('sample', colors)
            plt.imshow(p_grid, cmap=cm)
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.colorbar()
            plt.show()

        # a) Initialise grid
        max_distance_per_frame = 1.2
        step = 0.1
        freq = 10
        grid_size = int(max_distance_per_frame / step * 2)

        grid = np.zeros(grid_size * grid_size).reshape((grid_size, grid_size))

        # b) Calculate position vectors
        team_df = play_data.loc[(play_data['teamType'] == team) & (play_data['frameId'] >= start_frame) & (play_data['frameId'] <= end_frame)]
        for idx, row in team_df.iterrows():
            # TODO: Unhandled error - Analysing play 2018092300-1846
            i = round((row.dir_vec[0] * (row.s / freq)) / step) + int(grid_size / 2)
            j = round((row.dir_vec[1] * (row.s / freq)) / step) + int(grid_size / 2)
            try:
                grid[i][j] = grid[i][j] + 1
            except IndexError:
                print(f'Index error in {team_df["gameId"].values[0]} - {team_df["playId"].values[0]}')
        # c) Estimate probabilities
        p_grid = np.zeros(grid_size * grid_size).reshape((grid_size, grid_size))
        sum_f = grid.sum()
        for i in range(0, grid_size):
            for j in range(0, grid_size):
                p_grid[i][j] = grid[i][j] / sum_f if grid[i][j] > 0 else 0

        # d) Calculate Shannon-Entropy (h)
        h_home = []
        h_grid = np.zeros(grid_size * grid_size).reshape((grid_size, grid_size))
        for i in range(0, grid_size):
            for j in range(0, grid_size):
                if p_grid[i][j] > 0:
                    h_grid[i][j] = p_grid[i][j] * math.log(p_grid[i][j], 2)
        h_home.append(-1 * h_grid.sum())

        # e) Plot position probabilities
        # plot_heatmap()

        return h_home

    # Initialise new dataframe
    group_data_summary = pd.DataFrame()
    try:
        first, snap, throw, catch, last = find_reference_frames(play_data, 'offense')
        group_data_summary['offense_h_play'] = calc_team_h_group('offense')
        group_data_summary['offense_h_presnap'] = calc_team_h_group('offense', first, snap)
        group_data_summary['offense_h_to_throw'] = calc_team_h_group('offense', snap, throw)
        group_data_summary['offense_h_to_arrived'] = calc_team_h_group('offense', throw, catch)
        group_data_summary['offense_h_to_end'] = calc_team_h_group('offense', catch, last)

        first, snap, throw, catch, last = find_reference_frames(play_data, 'defense')
        group_data_summary['defense_h_play'] = calc_team_h_group('defense')
        group_data_summary['defense_h_presnap'] = calc_team_h_group('defense', first, snap)
        group_data_summary['defense_h_to_throw'] = calc_team_h_group('defense', snap, throw)
        group_data_summary['defense_h_to_arrived'] = calc_team_h_group('defense', throw, catch)
        group_data_summary['defense_h_to_end'] = calc_team_h_group('defense', catch, last)
    except ValueError:
        print(f'Error in {group_data["gameId"].values[0]} - {group_data["playId"].values[0]}')
    group_data_summary['gameId'] = group_data['gameId'].values[0]
    group_data_summary['playId'] = group_data['playId'].values[0]
    group_data_summary = group_data_summary.reindex(
        columns=['gameId', 'playId', 'offense_h_play', 'offense_h_presnap', 'offense_h_to_throw',
                 'offense_h_to_arrived', 'offense_h_to_end', 'defense_h_play', 'defense_h_presnap',
                 'defense_h_to_throw', 'defense_h_to_arrived', 'defense_h_to_end'])
    return group_data_summary


def calc_play_a_group(play_data, group_data):

    def plot_convex_hulls(play_data, ref_frame):
        # Calculate area of convex hulls
        # hull_frame = play_data.loc[(play_data['event'] == 'ball_snapped')]['frameId'].tolist()[0]
        # los = plays.loc[(plays['playId'] == play)]['absoluteYardlineNumber'].tolist()[0]

        # Home team convex hull
        a_data = play_data.loc[(play_data['teamType'] == 'offense') & (play_data['frameId'] == ref_frame)]
        a_points = np.zeros((len(a_data), 2))
        a_points[:, 0] = np.array(a_data['pos'].tolist())[:, 0]
        a_points[:, 1] = np.array(a_data['pos'].tolist())[:, 1]
        a_hull = ConvexHull(a_points)

        # Away team convex hull
        a2_data = play_data.loc[(play_data['teamType'] == 'defense') & (play_data['frameId'] == ref_frame)]
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

    # plot_convex_hulls(play_data, 30)

    def calc_team_a_group(team):
        team_data = play_data.loc[(play_data['teamType'] == team)]
        a_group = []

        for frame in range(1, team_data['frameId'].max() + 1):
            a_data = team_data.loc[(team_data['frameId'] == frame)]
            a_hull = ConvexHull(np.array(a_data['pos'].tolist()))
            a_group.append(a_hull.volume)
        return a_group

    # Determine area occupied by home and away teams
    o_a_group = calc_team_a_group('offense')
    d_a_group = calc_team_a_group('defense')

    # 3. Return values
    group_data['offense_a_group'] = pd.Series(o_a_group)
    group_data['defense_a_group'] = pd.Series(d_a_group)
    group_data['a_group_ratio'] = group_data.apply(lambda x: x.offense_a_group / x.defense_a_group, axis=1)

    return group_data


def find_reference_frames(df, team):
    data = df.loc[(df['teamType'] == team)]
    first_frame = data['frameId'].min()
    ball_snap = data.loc[(data['event'] == 'ball_snap')]['frameId'].min()
    pass_forward = data.loc[(data['event'] == 'pass_forward')]['frameId'].min()
    pass_arrived = data.loc[(data['event'] == 'pass_arrived')]['frameId'].min()
    last_frame = data['frameId'].max()

    return first_frame, ball_snap, pass_forward, pass_arrived, last_frame


def summarise_play_data(tracking, state, summary):
    # Create a supplemental dataframe calculating the parameters for each portion of the play
    possible_events = ['None', 'ball_snap', 'pass_forward', 'pass_arrived', 'pass_outcome_caught', 'out_of_bounds',
                       'pass_outcome_incomplete', 'first_contact', 'tackle', 'man_in_motion', 'play_action', 'handoff',
                       'pass_tipped', 'pass_outcome_interception', 'pass_shovel', 'line_set', 'pass_outcome_touchdown',
                       'fumble', 'fumble_offense_recovered', 'fumble_defense_recovered', 'touchdown', 'shift',
                       'touchback', 'penalty_flag', 'penalty_accepted', 'field_goal_blocked']
    possible_routes = ['HITCH', 'OUT', 'FLAT', 'CROSS', 'GO', 'SLANT', 'SCREEN', 'CORNER', 'IN', 'ANGLE', 'POST',
                       'WHEEL']

    teams = ['offense', 'defense']
    values = ['p', 'm', 'v', 'a']
    segments = ['play', 'presnap', 'to_throw', 'to_arrived', 'to_end']

    # Find the reference frames for each play segment for offense and defense (should match but this allows for data errors)
    for team in teams:
        first_frame, ball_snap, pass_forward, pass_arrived, last_frame = find_reference_frames(tracking, team)
        bounds = {
            'play': [first_frame, last_frame],
            'presnap': [first_frame, ball_snap],
            'to_throw': [ball_snap, pass_forward],
            'to_arrived': [pass_forward, pass_arrived],
            'to_end': [pass_arrived, last_frame]
        }
        # Find the average of each order parameter for the given period of the play
        for value in values:
            for segment in segments:
                segment_data = state.loc[(tracking['frameId'] >= first_frame) & (tracking['frameId'] <= last_frame)]
                col_to_find = team + '_' + value + '_group'
                col_to_add = team + '_' + value + '_' + segment
                start_frame = bounds[segment][0]
                end_frame = bounds[segment][1]
                segment_data = segment_data.loc[(segment_data['frameId'] >= start_frame) & (segment_data['frameId'] <= end_frame)]
                summary[col_to_add] = segment_data[col_to_find].mean()

    # Calculate the number of each route that are run in the play and tabulate them in the dataframe
    players = tracking.loc[(tracking['teamType'] == 'offense')].copy()['nflId'].unique().tolist()
    routes_run = {}
    for player in players:
        route = tracking.loc[(tracking['nflId'] == player)]['route'].values[0]
        if route in possible_routes:
            if route in routes_run.keys():
                routes_run[route] = routes_run[route] + 1
            else:
                routes_run[route] = 1

    for route in possible_routes:
        summary[route] = routes_run[route] if route in routes_run.keys() else 0

    # Record the total number of routes run in the play
    summary['num_routes'] = sum(routes_run.values())

    return summary
