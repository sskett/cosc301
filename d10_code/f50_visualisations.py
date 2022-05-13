from .f51_draw import draw_play
from .f51_draw import draw_heatmap
from .f52_animate import animate_play


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

    # draw_heatmap(group_df, 'offense_m_group', 50, 'offense_p_group', 50, game_id, play_id)
    # draw_heatmap(group_df, 'offense_v_group', 50, 'offense_p_group', 50, game_id, play_id)
    # draw_heatmap(group_df, 'offense_a_group', 50, 'offense_p_group', 50, game_id, play_id)
