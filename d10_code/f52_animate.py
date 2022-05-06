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
        patch = []

        # Offensive players
        offense_df = play_offense.query('frameId == ' + str(frame))
        off_x = offense_df.apply(lambda x: x.pos[0], axis=1)
        off_y = offense_df.apply(lambda x: x.pos[1], axis=1)
        off_number = offense_df['jerseyNumber']
        off_orientation = offense_df['o_vec']
        off_dir = offense_df['dir_vec']
        off_speed = offense_df['s']

        # Location
        patch.extend(plt.plot(off_x, off_y, 'o', c='gold', ms=20, mec='white'))

        # Jersey Numbers
        for x, y, num in zip(off_x, off_y, off_number):
            patch.append(plt.text(x, y, int(num), va='center', ha='center', color='black', size='medium'))

        # Orientation
        for x, y, orient in zip(off_x, off_y, off_orientation):
            dx = orient[0]
            dy = orient[1]
            patch.append(plt.arrow(x, y, dx, dy, color='gold', width=0.5, shape='full'))

        # Direction
        for x, y, direction, speed in zip(off_x, off_y, off_dir, off_speed):
            dx = direction[0] * speed / 10
            dy = direction[1] * speed / 10
            patch.append(plt.arrow(x, y, dx, dy, color='black', width=0.25, shape='full'))

        # Defensive players
        defense_df = play_defense.query('frameId == ' + str(frame))
        def_x = defense_df.apply(lambda x: x.pos[0], axis=1)
        def_y = defense_df.apply(lambda x: x.pos[1], axis=1)
        def_number = defense_df['jerseyNumber']
        def_orientation = defense_df['o_vec']
        def_dir = defense_df['dir_vec']
        def_speed = defense_df['s']

        # Location
        patch.extend(plt.plot(def_x, def_y, 'o', c='orangered', ms=20, mec='white'))

        # Jersey Numbers
        for x, y, num in zip(def_x, def_y, def_number):
            patch.append(plt.text(x, y, int(num), va='center', ha='center', color='white', size='medium'))

        # Orientation
        for x, y, orient in zip(def_x, def_y, def_orientation):
            dx = orient[0]
            dy = orient[1]
            patch.append(plt.arrow(x, y, dx, dy, color='orangered', width=0.5, shape='full'))

        # Direction
        for x, y, direction, speed in zip(def_x, def_y, def_dir, def_speed):
            dx = direction[0] * speed / 10
            dy = direction[1] * speed / 10
            patch.append(plt.arrow(x, y, dx, dy, color='black', width=0.25, shape='full'))

        return patch

    play_offense = tracking_df.loc[(tracking_df['teamType'] == 'offense')].copy()
    play_defense = tracking_df.loc[(tracking_df['teamType'] == 'defense')].copy()

    max_frame = tracking_df['frameId'].unique().max()
    min_frame = tracking_df['frameId'].unique().min()

    play_dir = tracking_df.sample(1)['playDirection'].values[0]
    yards_to_go = plays_df['yardsToGo'].values[0] if play_dir == 'left' else plays_df['yardsToGo'].values[0] * -1
    yardline_number = plays_df['yardlineNumber'].values[0]
    abs_yardline_number = 120 - plays_df['absoluteYardlineNumber'].values[0] if tracking_df['playDirection'].values[0] == 'left' else plays_df['absoluteYardlineNumber'].values[0]

    fig, ax = draw_field(highlight_line=True, highlight_line_number=abs_yardline_number)
    play_desc = plays_df['playDescription'].values[0]
    plt.title(f'Game {game_id} Play {play_id}\n {play_desc}')

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

