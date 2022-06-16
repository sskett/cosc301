def filter_player_data(df, options):
    # print('Filtering player data')
    num_rows_initial = df.size
    if options['players_to_import_by_id']:
        df = df.loc[(df['nflId'].isin(options['players_to_import_by_id']))]
    if options['players_to_import_by_name']:
        df = df.loc[(df['displayName'].isin(options['players_to_import_by_name']))]
    num_rows_final = df.size
    # print(f'Removed {num_rows_initial - num_rows_final} player entries.')
    return df


def filter_game_data(df, options):
    # print('Filtering game data')
    num_rows_initial = df.size
    if options['teams_to_import']:
        df = df.loc[(df['homeTeamAbbr'].isin(options['teams_to_import'])) | (df['visitorTeamAbbr'].isin(options['teams_to_import']))]
    if options['weeks_to_import']:
        if type(options['weeks_to_import']) == int:
            df = df.loc[df['week'] <= options['weeks_to_import']]
        if type(options['weeks_to_import']) == list:
            df = df.loc[df['week'].isin(options['weeks_to_import'])]
    if options['games_to_import']:
        df = df.loc[df['gameId'].isin(options['games_to_import'])]
    num_rows_final = df.size
    # print(f'Removed {num_rows_initial - num_rows_final} game entries.')
    return df


def filter_play_data(df, options, game_ids):
    # print('Filtering play data')
    num_rows_initial = df.size
    df = df.loc[df['gameId'].isin(game_ids)]
    if options['plays_to_import']:
        df = df.loc[df['playId'].isin(options['plays_to_import'])]
    num_rows_final = df.size
    # print(f'Removed {num_rows_initial - num_rows_final} play-by-play entries.')
    return df


def filter_tracking_data(df, game_id, play_id):
    # print(f'Filtering tracking data for {game_id} - {play_id}')
    return df.loc[(df['gameId'] == game_id) & (df['playId'] == play_id)]
