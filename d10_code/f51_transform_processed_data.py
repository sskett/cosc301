import pandas as pd
import numpy as np


def string_to_vector(s):
    try:
        s = s.split('[')[1].split(']')[0]
        x = float(s.split()[0])
        y = float(s.split()[1])
        return np.array([x, y])
    except AttributeError:
        return None


def get_position_delta(row):
    return row.s / 10 * row.dir_vec


def get_relative_position(row):
    if row.frameId == 1:
        return np.array([0, 0])
    else:
        last_pos = row.shift(1).rel_pos
        return last_pos + row.pos_delta


def column_concat(df, col1, col2):
    return df.apply(lambda x: str(x[col1]) + '-' + str(x[col2]), axis=1)


def prepare_routes_data(tracking_data, qb_positions, gpids):
    routes_df = pd.DataFrame(columns=['gpid', 'nflId', 'route', 'pos'])
    index = 0
    end_events = ['pass_arrived', 'pass_outcome_caught', 'out_of_bounds',
                  'pass_outcome_incomplete', 'first_contact', 'tackle', 'man_in_motion', 'play_action', 'handoff',
                  'pass_tipped', 'pass_outcome_interception', 'pass_shovel', 'line_set', 'pass_outcome_touchdown',
                  'fumble', 'fumble_offense_recovered', 'fumble_defense_recovered', 'touchdown', 'shift',
                  'touchback', 'penalty_flag', 'penalty_accepted', 'field_goal_blocked']

    # Generate vectors of positions
    for gpid in gpids[:100]:
        df = tracking_data.loc[tracking_data['gpid'] == gpid]
        players = df['nflId'].unique().tolist()
        centre_pos = qb_positions.loc[(qb_positions['gpid'] == gpid)]['pos'].values[0][1]
        for player in players:
            player_df = df.loc[(df['nflId'] == player)]
            if player > 0:
                start_frame = player_df.loc[player_df['event'] == 'ball_snap']['frameId'].min()
                end_frame = player_df.loc[player_df['event'].isin(end_events)]['frameId'].min()
                if end_frame > start_frame:
                    positions = []
                    start_pos = player_df.loc[(player_df['frameId'] == start_frame)]['pos'].values[0]
                    flip_arr = True if start_pos[1] < centre_pos else False
                    for frame in range(start_frame + 1, end_frame + 1):
                        new_pos = player_df.loc[(player_df['frameId'] == frame)]['pos'].values[0].tolist()
                        if flip_arr:
                            new_pos[1] = new_pos[1] * -1
                        new_pos.append(player_df.loc[(player_df['frameId'] == frame)]['s'].values[0])
                        positions.append(new_pos)
                    positions = np.array(positions)
                    try:
                        routes_df.loc[index] = {
                            'gpid': str(gpid),
                            'nflId': int(player),
                            'route': str(player_df.loc[(player_df['frameId'] == 1)]['route'].values[0]),
                            'pos': positions - positions[0]
                        }
                        index += 1
                    except IndexError:
                        print(f'Data error for gpid {gpid}, player {player} (Start {start_frame} | End {end_frame})')

    return routes_df