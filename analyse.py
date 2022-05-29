import pandas
import numpy
import time

from d10_code import f11_import_processed_data as dfi
from d10_code import f21_filter_processed_data as dff
from d10_code import f31_clean_processed_data as dfc
from d10_code import f42_transform_processed_data as dft

# Import data
data_directory = './d20_intermediate_files'

start_time = time.time()
print('import plays data')
play_df = dfi.import_processed_play_data(data_directory + '/play_results.csv')
print('import frames data')
frame_df = dfi.import_processed_frame_data(data_directory + '/frame_results.csv')
print('import tracking data')
tracking_df = dfi.import_processed_tracking_data(data_directory + '/tracking_results.csv')

# Filter data
print('filtering data')
o_players_df = dff.get_players_by_position(tracking_df, ['QB', 'WR', 'TE', 'RB'])
del tracking_df

# Prepare data
print('prepping route_runners')
o_players_df = dfc.fix_imported_tracking_data(o_players_df)
o_players_df['gpid'] = dft.column_concat(o_players_df, 'gameId', 'playId')

print('get QB positions')
qb_positions = dff.get_field_positions(o_players_df, 'QB', 1)

print('removing non-route runners')
players_with_routes_df = o_players_df.loc[(~o_players_df['route'].isna())]

print('getting gpids')
gpids = players_with_routes_df['gpid'].unique().tolist()

# Extract data for analysis
routes_df = dft.prepare_routes_data(players_with_routes_df, qb_positions, gpids)

finish_time = time.time()
print(f'Completed in {finish_time - start_time} seconds.')
