import pandas as pd
import numpy as np

import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


def draw_field(linenumbers=True, endzones=True, highlight_line=False, highlight_line_number=50, highlight_line_name='Line of Scrimmage', fifty_is_los=False, size=(12, 6.33)):
    # Draw the field
    x_path = [10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120]
    y_path = [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 53.3, 0, 0, 53.3]
    line_colour = 'white'

    field = patches.Rectangle(
        (0, 0),
        120,
        53.3,
        linewidth=0.1,
        edgecolor='r',
        facecolor='darkgreen',
        zorder=0
    )
    fig, ax = plt.subplots(1, figsize=size)
    ax.add_patch(field)
    plt.plot(x_path, y_path, line_colour)

    # Draw the LOS
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], colour='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')

    # Draw the endzones
    if endzones:
        left_ez = patches.Rectangle((0, 0), 10, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
        right_ez = patches.Rectangle((110, 0), 120, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
        ax.add_patch(left_ez)
        ax.add_patch(right_ez)

    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')

    # Draw the line markings
    if linenumbers:
        for x in range(20, 110, 10):
            yardage = x
            if x > 50:
                yardage = 120 - x
            plt.text(x, 5, str(yardage - 10),
                     horizontalalignment='center',
                     fontsize=20,
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(yardage - 10),
                     horizontalalignment='center',
                     fontsize=20,
                     color='white',
                     rotation=180)

    # Draw the hash marks
    hash_range = range(11, 110) if endzones else range(1, 120)
    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    # Draw the highlighted line
    if highlight_line:
        hl = highlight_line_number
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<-{}'.format(highlight_line_name), color='yellow')

    return fig, ax


# draw snap shot of play
def draw_play_state(play_data, start=1, end=1, los=50):
    def draw_team_state(team, colour):
        team_data = play_data.loc[(play_data['teamType'] == team) & (play_data['frameId'] >= start) & (play_data['frameId'] <= end)].copy()
        # draw positions
        team_data['x'] = team_data.apply(lambda x: x.pos[0], axis=1)
        team_data['y'] = team_data.apply(lambda x: x.pos[1], axis=1)
        team_data.plot(x='x', y='y', kind='scatter', ax=ax, color=colour, s=1, marker='.')
        team_data.loc[team_data['frameId'] == start].plot(x='x', y='y', kind='scatter', ax=ax, color=colour, s=20, legend=team, label=team)
        # draw orientations
        # draw offense direction vectors
        # draw offense convex hull

    start = start if play_data['frameId'].min() <= start <= play_data['frameId'].max() else play_data['frameId'].min()
    end = end if play_data['frameId'].min() <= end <= play_data['frameId'].max() else play_data['frameId'].max()
    # draw field as background
    fig, ax = draw_field(highlight_line=True, highlight_line_number=los)

    # draw the teams
    draw_team_state('offense', 'blue')
    draw_team_state('defense', 'red')
    draw_team_state('ball', 'black')


def draw_play(filename, game_id, play_id, week, plays_df, tracking_df, start_event=None, end_event=None):
    if not os.path.exists(filename.rsplit('/', 1)[0]):
        os.makedirs(filename.rsplit('/', 1)[0])

    play_data = plays_df.loc[plays_df['playId'] == play_id]
    if pd.isna(play_data['absoluteYardlineNumber'].values[0]):
        los = 0
    else:
        los = 120 - play_data['absoluteYardlineNumber'].values[0] if tracking_df['playDirection'].values[0] == 'left' else play_data['absoluteYardlineNumber'].values[0]

    start_frame = tracking_df.loc[(tracking_df['event'] == start_event)]['frameId'].min() if start_event else 1
    end_frame = tracking_df.loc[(tracking_df['event'] == end_event)]['frameId'].max() if end_event else tracking_df['frameId'].max()
    draw_play_state(tracking_df, start=start_frame, end=end_frame, los=los)

    # label the plot
    plt.suptitle(f'(Week {week}) Game {game_id} - Play {play_id} [From {start_event} to {end_event}]')
    plt.title(play_data['playDescription'].values[0])
    plt.legend(loc=1)
    plt.savefig(filename)
    plt.close('all')


def draw_heatmap(df, x_col, x_dim, y_col, y_dim, game_id, play_id):

    def smooth(series, window=5, std=1, centering=True):
        s_df = series.rolling(window=window, win_type='gaussian', center=centering).mean(std=std)
        return s_df.dropna()

    def normalise_df(df_n):
        return df_n / df_n.max() * 0.999

    x = smooth(normalise_df(df.loc[(df['gameId'] == game_id) & (df['playId'] == play_id)][x_col])).tolist()
    y = smooth(normalise_df(df.loc[(df['gameId'] == game_id) & (df['playId'] == play_id)][y_col])).tolist()

    bins = np.zeros(x_dim * y_dim).reshape((x_dim, y_dim))

    for frame in zip(x, y):
        x_bin = int(np.floor(frame[0] * x_dim))
        y_bin = int(np.floor(frame[1] * y_dim))
        bins[x_bin][y_bin] = bins[x_bin][y_bin] + 1

    binned_p = np.zeros(x_dim * y_dim).reshape((x_dim, y_dim))

    for i in range(0, x_dim):
        for j in range(0, y_dim):
            count = 0
            samples = 0
            for row in range(-1, 2):
                for col in range(-1, 2):
                    if 0 <= i + row <= x_dim - 1 and 0 <= j + col <= y_dim - 1:
                        samples = samples + 1
                        count = count + bins[i + row][j + col]
            binned_p[i][j] = count / samples

    colours = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0), (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]
    cm = LinearSegmentedColormap.from_list('sample', colours)
    plt.imshow(binned_p, cmap=cm)
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.show()

