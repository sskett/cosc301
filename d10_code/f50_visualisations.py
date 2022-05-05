import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .f51_draw import draw_play_state
from .f52_animate import animate_play


def visualise_play(game_id, play_id, week, plays_df, tracking_df, start_event=None, end_event=None):

    def draw_play():
        play_data = plays_df.loc[plays_df['playId'] == play_id]
        los = 120 - play_data['absoluteYardlineNumber'].values[0] if tracking_df['playDirection'].values[0] == 'left' else play_data['absoluteYardlineNumber'].values[0]

        start_frame = tracking_df.loc[(tracking_df['event'] == start_event)]['frameId'].min() if start_event else 1
        end_frame = tracking_df.loc[(tracking_df['event'] == end_event)]['frameId'].max() if end_event else tracking_df['frameId'].max()
        draw_play_state(tracking_df, start=start_frame, end=end_frame, los=los)

        # label the plot
        plt.suptitle(f'(Week {week}) Game {game_id} - Play {play_id} [From {start_event} to {end_event}]')
        plt.title(plays_df['playDescription'].values[0])
        plt.legend(loc=1)
        plt.show()

    def anim_play():
        folder = './d20_intermediate_files/' + str(game_id) + '/' + str(play_id) + '.mp4'
        animate_play(folder, play_id, game_id, plays_df, tracking_df)

    draw_play()
    anim_play()
