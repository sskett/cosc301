from .f61_draw import draw_play
from .f62_animate import animate_play


def visualise_play(game_id, play_id, week, plays_df, tracking_df, group_df, start_event=None, end_event=None):

    def draw(folder):
        filepath = folder + str(game_id) + '/' + str(play_id) + '.png'
        draw_play(filepath, game_id, play_id, week, plays_df, tracking_df, start_event=None, end_event=None)

    def animate(folder):
        filepath = folder + str(game_id) + '/' + str(play_id) + '.mp4'
        animate_play(filepath, play_id, game_id, plays_df, tracking_df)

    root_dir = './d20_intermediate_files/'
    draw(root_dir)
    animate(root_dir)
