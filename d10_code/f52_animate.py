import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

import os.path
import dateutil

from .f51_draw import draw_field


def animate_player_movement(play_id, game_id, plays_df, tracking_df):
    def update_animation(frame):
        def update_frame(team_data, team_colour):
            # Offensive players
            frame_df = team_data.query('frameId == ' + str(frame))
            frame_x = frame_df.apply(lambda x: x.pos[0], axis=1)
            frame_y = frame_df.apply(lambda x: x.pos[1], axis=1)

            # Location
            patch.extend(plt.plot(frame_x, frame_y, 'o', c=team_colour, ms=20, mec='white'))

            # Jersey Numbers
            for x, y, num in zip(frame_x, frame_y, frame_df['jerseyNumber']):
                patch.append(plt.text(x, y, int(num), va='center', ha='center', color='black', size='medium'))

            # Orientation
            for x, y, orient in zip(frame_x, frame_y, frame_df['o_vec']):
                dx = orient[0]
                dy = orient[1]
                patch.append(plt.arrow(x, y, dx, dy, color=team_colour, width=0.5, shape='full'))

            # Direction
            for x, y, direction, speed in zip(frame_x, frame_y, frame_df['dir_vec'], frame_df['s']):
                dx = direction[0] * speed / 10
                dy = direction[1] * speed / 10
                patch.append(plt.arrow(x, y, dx, dy, color='black', width=0.25, shape='full'))

        patch = []
        update_frame(play_offense, 'gold')
        update_frame(play_defense, 'orangered')

        return patch

    # Filter team data
    play_offense = tracking_df.loc[(tracking_df['teamType'] == 'offense')].copy()
    play_defense = tracking_df.loc[(tracking_df['teamType'] == 'defense')].copy()
    play_football = tracking_df.loc[(tracking_df['team'] == 'football')].copy()

    # Find boundaries of play data
    max_frame = tracking_df['frameId'].unique().max()
    min_frame = tracking_df['frameId'].unique().min()

    # Determine key characteristics
    play_dir = tracking_df.sample(1)['playDirection'].values[0]
    yards_to_go = plays_df['yardsToGo'].values[0] if play_dir == 'left' else plays_df['yardsToGo'].values[0] * -1
    yardline_number = plays_df['yardlineNumber'].values[0]
    abs_yardline_number = 120 - plays_df['absoluteYardlineNumber'].values[0] if tracking_df['playDirection'].values[0] == 'left' else plays_df['absoluteYardlineNumber'].values[0]

    # Draw field
    fig, ax = draw_field(highlight_line=True, highlight_line_number=abs_yardline_number)
    play_desc = plays_df['playDescription'].values[0]
    plt.title(f'Game {game_id} Play {play_id}\n {play_desc}')

    # Draw animation frames
    ims = [[]]
    for frame in np.arange(min_frame, max_frame + 1):
        patch = update_animation(frame)
        ims.append(patch)

    return animation.ArtistAnimation(fig, ims, repeat=False)


def animate_play(filename, play_id, game_id, plays_df, tracking_df, speed=10):
    anim = animate_player_movement(play_id, game_id, plays_df, tracking_df)
    writer = FFMpegWriter(fps=speed)
    if not os.path.exists(filename.rsplit('/', 1)[0]):
        os.makedirs(filename.rsplit('/', 1)[0])
    anim.save(filename, writer=writer)

