import pandas as pd
import numpy as np
import ray


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


@ray.remote
def process_data(dataset, qb_positions):
    routes_df = pd.DataFrame(columns=['gpid', 'nflId', 'route', 'pos'])
    index = 0
    end_events = ['pass_arrived', 'pass_outcome_caught', 'out_of_bounds',
                  'pass_outcome_incomplete', 'first_contact', 'tackle', 'man_in_motion', 'play_action', 'handoff',
                  'pass_tipped', 'pass_outcome_interception', 'pass_shovel', 'line_set', 'pass_outcome_touchdown',
                  'fumble', 'fumble_offense_recovered', 'fumble_defense_recovered', 'touchdown', 'shift',
                  'touchback', 'penalty_flag', 'penalty_accepted', 'field_goal_blocked']

    gpids = dataset['gpid'].unique().tolist()
    # Process route data on a per-player, per-play basis
    for gpid in gpids:
        try:
            df = dataset.loc[dataset['gpid'] == gpid]
            players = df['nflId'].unique().tolist()
            # Determine middle of offense to put route directions in a 'inside/outside' vs 'left/right' context
            centre_pos = qb_positions.loc[(qb_positions['gpid'] == gpid)]['pos'].values[0][1]
            for player in players:
                player_df = df.loc[(df['nflId'] == player)]
                if player > 0:
                    # Filter data to start of actual play until the pass_arrives or play ends for another reason
                    start_frame = player_df.loc[player_df['event'] == 'ball_snap']['frameId'].min()
                    end_frame = player_df.loc[player_df['event'].isin(end_events)]['frameId'].min()
                    # Check for plays that end prior to the starting event being triggered
                    if end_frame > start_frame:
                        positions = []
                        start_pos = player_df.loc[(player_df['frameId'] == start_frame)]['pos'].values[0]
                        # If the player is to the left of the QB flip the left/right dimension
                        flip_arr = True if start_pos[1] < centre_pos else False
                        for frame in range(start_frame + 1, end_frame + 1):
                            new_pos = player_df.loc[(player_df['frameId'] == frame)]['pos'].values[0].tolist()
                            if flip_arr:
                                new_pos[1] = new_pos[1] * -1
                            new_pos.append(player_df.loc[(player_df['frameId'] == frame)]['s'].values[0])
                            positions.append(new_pos)
                        # Collate all positions within the play and convert from a simple list to an array
                        positions = np.array(positions)
                        # Record the route details in a dataframe
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


def prepare_routes_data(tracking_data, qb_positions, gpids, n_procs):
    n_gpids = len(gpids)
    max_gpids = int(len(gpids[:n_gpids]))
    gpid_sets = []
    set_pos = 0
    step = int(max_gpids / (n_procs - 1))

    while set_pos < max_gpids:
        start = 0 if set_pos == 0 else set_pos + 1
        set_pos = set_pos + step
        end = set_pos if set_pos < max_gpids else max_gpids
        gpid_sets.append((start, end))

    routes_df = pd.DataFrame(columns=['gpid', 'nflId', 'route', 'pos'])
    futures = [process_data.remote(tracking_data.loc[(tracking_data['gpid'].isin(gpids[idx[0]:idx[1]]))], qb_positions)
               for idx in gpid_sets]
    results = ray.get(futures)

    for i in range(0, len(results)):
        routes_df = pd.concat([routes_df, results[i]], ignore_index=True)

    return routes_df
