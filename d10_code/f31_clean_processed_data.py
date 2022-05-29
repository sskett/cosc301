from d10_code import f51_transform_processed_data as dft


def fix_imported_tracking_data(df):
    # Correction for stringified vectors
    df['pos'] = df['pos'].apply(dft.string_to_vector)
    df['o_vec'] = df['o_vec'].apply(dft.string_to_vector)
    df['dir_vec'] = df['dir_vec'].apply(dft.string_to_vector)
    df['r_vec'] = df['r_vec'].apply(dft.string_to_vector)

    # Correction nan values forcing ints to floats
    df['nflId'] = df['nflId'].astype('Int64')
    df['jerseyNumber'] = df['jerseyNumber'].astype('Int64')
    return df
