import pandas
import numpy
import time

from d10_code import f11_import_processed_data as dfi

data_directory = './d20_intermediate_files'

start_time = time.time()
print('import plays')
play_df = dfi.import_processed_play_data(data_directory + '/play_results.csv')
print('import frames')
frame_df = dfi.import_processed_frame_data(data_directory + '/frame_results.csv')
print('import tracking')
tracking_df = dfi.import_processed_tracking_data(data_directory + '/tracking_results.csv')

finish_time = time.time()
print(f'Completed in {finish_time - start_time} seconds.')
