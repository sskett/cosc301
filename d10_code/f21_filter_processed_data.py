def get_players_by_position(df, positions):
    return df.loc[(df['position'].isin(positions))].copy()


def get_field_positions(df, position, frame):
    return df.loc[(df['frameId'] == frame) & (df['position'] == position)][['gpid', 'pos']].copy()

