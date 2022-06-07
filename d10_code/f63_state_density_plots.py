import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from d10_code import f11_import_processed_data as dfi


def draw_prob_density(x_series, y_series, x_bins, y_bins, ax, xlab=None, ylab=None, cbar=True):
    dims = {'x': x_series.max() * 1.00001, 'y': y_series.max() * 1.00001}

    bins = np.zeros(x_bins * y_bins).reshape(y_bins, x_bins)
    rows = min(len(y_series), len(x_series))

    for row in range(rows):
        i = int(x_series.iloc[row] / dims['x'] * x_bins)
        j = int(y_series.iloc[row] / dims['y'] * y_bins)
        bins[j][i] += 1

    smoothing_grid = np.zeros(x_bins * y_bins).reshape(y_bins, x_bins)
    for i in range(x_bins):
        for j in range(y_bins):
            sum = 0
            count = 0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if 0 <= i + x < x_bins and 0 <= j + y < y_bins:
                        sum += bins[j + y][i + x]
                        count += 1
            smoothing_grid[j][i] = sum / count

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if cbar:
        c = ax.pcolor(smoothing_grid, cmap='jet')
        plt.colorbar(c, ax=ax)
    else:
        ax.pcolor(smoothing_grid, cmap='jet')


def get_o_transition_state(row):
    if row.offense_p_play > 0.65 and row.offense_m_play < 0.35:
        return 'polar'
    elif row.offense_p_play < 0.35 and row.offense_m_play < 0.35:
        return 'swarm'
    elif row.offense_p_play < 0.35 and row.offense_m_play > 0.65:
        return 'milling'
    else:
        return 'transitional'


def get_d_transition_state(row):
    if row.defense_p_play > 0.65 and row.defense_m_play < 0.35:
        return 'polar'
    elif row.defense_p_play < 0.35 and row.defense_m_play < 0.35:
        return 'swarm'
    elif row.defense_p_play < 0.35 and row.defense_m_play > 0.65:
        return 'milling'
    else:
        return 'transitional'


def generate_plots():
    # Import data
    data_directory = './d20_intermediate_files'
    print('import plays data')
    play_df = dfi.import_processed_play_data(data_directory + '/play_results.csv')

    fig_height = 3
    output_folder = './d30_results/'

    p_axis = r'$P_{group}$'
    m_axis = r'$M_{group}$'
    v_axis = r'$V_{group}$ (yd/s)'
    h_axis = r'$H_{group}$ (bits)'
    a_axis = r'$A_{group}$ ($yd^2$)'

    # TODO: Convert the following plots to one or two functions
    # Density plots for various pair-wise combinations of order parameters
    # Plot Figure 1
    fig, axs = plt.subplots(2, 2, sharey=True, figsize=(3*fig_height, fig_height * 2), gridspec_kw={'width_ratios': [1, 2]})

    draw_prob_density(play_df['offense_m_play'], play_df['offense_p_play'], 50, 50, axs[0][0], ylab=p_axis, cbar=False)
    draw_prob_density(play_df['offense_v_play'], play_df['offense_p_play'], 60, 50, axs[0][1])
    draw_prob_density(play_df['defense_m_play'], play_df['defense_p_play'], 50, 50, axs[1][0], m_axis, p_axis, cbar=False)
    draw_prob_density(play_df['defense_v_play'], play_df['defense_p_play'], 60, 50, axs[1][1], v_axis)

    fig.suptitle(r'FIGURE 1: Density plots for polarisation, angular momentum and mean group velocity')
    plt.savefig(f'{output_folder}figure1.png')

    # Plot Figure 2
    fig, axs = plt.subplots(2, 2, sharey=True, figsize=(3 * fig_height, fig_height * 2), gridspec_kw={'width_ratios': [1, 2]})

    draw_prob_density(play_df['offense_h_play'], play_df['offense_v_play'], 140, 60, axs[0][0], ylab=r'$V_{group}$ (yd/s)', cbar=False)
    draw_prob_density(play_df['offense_a_play'], play_df['offense_v_play'], 60, 60, axs[0][1],)
    draw_prob_density(play_df['defense_h_play'], play_df['defense_v_play'], 140, 60, axs[1][0], h_axis, v_axis, cbar=False)
    draw_prob_density(play_df['defense_a_play'], play_df['defense_v_play'], 60, 60, axs[1][1], a_axis)

    fig.suptitle('FIGURE 2: V_Group vs H_group and A_group')
    plt.savefig(f'{output_folder}figure2.png')

    # Plot Figure 3
    fig, axs = plt.subplots(2, 2, sharey=True, figsize=(3 * fig_height, fig_height * 2), gridspec_kw={'width_ratios': [1, 2]})

    draw_prob_density(play_df['offense_h_play'], play_df['offense_p_play'], 140, 50, axs[0][0], ylab=p_axis, cbar=False)
    draw_prob_density(play_df['offense_a_play'], play_df['offense_p_play'], 60, 50, axs[0][1])
    draw_prob_density(play_df['defense_h_play'], play_df['defense_p_play'], 140, 50, axs[1][0], h_axis, p_axis, cbar=False)
    draw_prob_density(play_df['defense_a_play'], play_df['defense_p_play'], 60, 50, axs[1][1], a_axis)

    fig.suptitle('FIGURE 3: P_group vs H_group and A_group')
    plt.savefig(f'{output_folder}figure3.png')

    # Plot Figure 4
    fig, axs = plt.subplots(2, 3, sharey=True, figsize=(3 * fig_height, fig_height * 2), gridspec_kw={'width_ratios': [1, 1, 1.25]})

    draw_prob_density(play_df['offense_m_to_throw'], play_df['offense_p_to_throw'], 50, 50, axs[0][0], ylab=p_axis, cbar=False)
    draw_prob_density(play_df['offense_m_to_arrived'], play_df['offense_p_to_arrived'], 50, 50, axs[0][1], cbar=False)
    draw_prob_density(play_df['offense_m_to_end'], play_df['offense_p_to_end'], 50, 50, axs[0][2])
    draw_prob_density(play_df['defense_m_to_throw'], play_df['defense_p_to_throw'], 50, 50, axs[1][0], m_axis, p_axis, cbar=False)
    draw_prob_density(play_df['defense_m_to_arrived'], play_df['defense_p_to_arrived'], 50, 50, axs[1][1], m_axis, cbar=False)
    draw_prob_density(play_df['defense_m_to_end'], play_df['defense_p_to_end'], 50, 50, axs[1][2], m_axis)

    fig.suptitle('FIGURE 4: P_Group vs M_Group (Stages of Play)')
    plt.savefig(f'{output_folder}figure4.png')

    # Plot Figure 5
    fig, axs = plt.subplots(2, 3, sharey=True, figsize=(4 * fig_height, fig_height * 2), gridspec_kw={'width_ratios': [1, 1, 1.25]})

    draw_prob_density(play_df['offense_a_to_throw'], play_df['offense_p_to_throw'], 60, 50, axs[0][0], ylab=p_axis, cbar=False)
    draw_prob_density(play_df['offense_a_to_arrived'], play_df['offense_p_to_arrived'], 60, 50, axs[0][1], cbar=False)
    draw_prob_density(play_df['offense_a_to_end'], play_df['offense_p_to_end'], 60, 50, axs[0][2])
    draw_prob_density(play_df['defense_a_to_throw'], play_df['defense_p_to_throw'], 60, 50, axs[1][0], a_axis, p_axis, cbar=False)
    draw_prob_density(play_df['defense_a_to_arrived'], play_df['defense_p_to_arrived'], 60, 50, axs[1][1], a_axis, cbar=False)
    draw_prob_density(play_df['defense_a_to_end'], play_df['defense_p_to_end'], 60, 50, axs[1][2], a_axis)

    fig.suptitle('FIGURE 5: P_group vs A_group (Stages of Play)')
    plt.savefig(f'{output_folder}figure5.png')

    # Plot Figure 6
    fig, axs = plt.subplots(2, 3, sharey=True, figsize=(4 * fig_height, fig_height * 2), gridspec_kw={'width_ratios': [1, 1, 1.25]})

    draw_prob_density(play_df['offense_v_to_throw'], play_df['offense_p_to_throw'], 60, 50, axs[0][0], ylab=p_axis, cbar=False)
    draw_prob_density(play_df['offense_v_to_arrived'], play_df['offense_p_to_arrived'], 60, 50, axs[0][1], cbar=False)
    draw_prob_density(play_df['offense_v_to_end'], play_df['offense_p_to_end'], 60, 50, axs[0][2], v_axis)

    draw_prob_density(play_df['defense_v_to_throw'], play_df['defense_p_to_throw'], 60, 50, axs[1][0], v_axis, p_axis, cbar=False)
    draw_prob_density(play_df['defense_v_to_arrived'], play_df['defense_p_to_arrived'], 60, 50, axs[1][1], v_axis, cbar=False)
    draw_prob_density(play_df['defense_v_to_end'], play_df['defense_p_to_end'], 60, 50, axs[1][2], v_axis)

    fig.suptitle('FIGURE 6: P_group vs V_group (Stages of Play)')
    plt.savefig(f'{output_folder}figure6.png')

    # Plot Figure 7
    fig, axs = plt.subplots(2, 3, sharey=True, figsize=(4 * fig_height, fig_height * 2), gridspec_kw={'width_ratios': [1, 1, 1.25]})

    draw_prob_density(play_df['offense_h_to_throw'], play_df['offense_p_to_throw'], 140, 50, axs[0][0], ylab=p_axis, cbar=False)
    draw_prob_density(play_df['offense_h_to_arrived'], play_df['offense_p_to_arrived'], 140, 50, axs[0][1], cbar=False)
    draw_prob_density(play_df['offense_h_to_end'], play_df['offense_p_to_end'], 140, 50, axs[0][2])

    draw_prob_density(play_df['defense_h_to_throw'], play_df['defense_p_to_throw'], 140, 50, axs[1][0], h_axis, p_axis, cbar=False)
    draw_prob_density(play_df['defense_h_to_arrived'], play_df['defense_p_to_arrived'], 140, 50, axs[1][1], h_axis, cbar=False)
    draw_prob_density(play_df['defense_h_to_end'], play_df['defense_p_to_end'], 140, 50, axs[1][2], h_axis)

    fig.suptitle('FIGURE 7: P_group vs H_group (Stages of Play)')
    plt.savefig(f'{output_folder}figure7.png')

    # Plots for partial periods within plays
    # Add state labels for the plays
    play_df['o_state'] = play_df.apply(get_o_transition_state, axis=1)
    play_df['d_state'] = play_df.apply(get_d_transition_state, axis=1)

    # Aggregate counts
    o_states = {'polar': play_df[(play_df['o_state'] == 'polar')].shape[0],
                'swarm': play_df[(play_df['o_state'] == 'swarm')].shape[0],
                'milling': play_df[(play_df['o_state'] == 'milling')].shape[0],
                'transitional': play_df[(play_df['o_state'] == 'transitional')].shape[0]}

    d_states = {'polar': play_df[(play_df['d_state'] == 'polar')].shape[0],
                'swarm': play_df[(play_df['d_state'] == 'swarm')].shape[0],
                'milling': play_df[(play_df['d_state'] == 'milling')].shape[0],
                'transitional': play_df[(play_df['d_state'] == 'transitional')].shape[0]}

    # Plot distribution
    ind = np.arange(len(o_states))
    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ind - width / 2, o_states.values(), width, label='Offense')
    ax.bar(ind + width / 2, d_states.values(), width, label='Defense')
    ax.set_ylabel('Number of instances')
    ax.set_title('Figure 8: Distribution of offensive and defensive collective states')
    ax.set_xticks(ind)
    ax.set_xticklabels(o_states.keys())
    ax.legend(loc='upper left')
    plt.savefig(f'{output_folder}figure8.png')

    # Clear dataframes from memory
    del play_df

