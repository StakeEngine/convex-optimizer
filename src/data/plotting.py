import matplotlib.pyplot as plt
import streamlit as st
from src.class_setup.state import AppState


def render_plots(state: AppState, containter):
    colours = ["blue", "purple"]
    if state.set_params:
        for i, c in enumerate(state.criteria_list):
            plot_params = state.plot_params[i]
            xmin_disp, xmax_disp = containter.columns(2)
            plot_params.xmin = xmin_disp.number_input(
                label="min x-axis", value=0, key=f"min_x_plot_{i}", width=150
            )
            plot_params.xmax = xmax_disp.number_input(
                label="max x-axis", value=10 * state.cost, key=f"max_x_plot_{i}", width=150
            )
            plot_params.show_the_curve = containter.checkbox(
                "Plot theoretical curve",
                value=plot_params.show_the_curve,
                key=f"plot_the_{i}",
            )
            with st.container():
                fig, ax = plt.subplots()
                for d in range(c.num_dists):
                    if c.plot_log_scale:
                        if plot_params.show_the_curve:
                            ax.semilogx(c.dist_values[d].xthe, c.dist_values[d].ythe, color="black")
                        ax.semilogx(
                            c.dist_values[d].xact,
                            c.dist_values[d].yact,
                            marker="o",
                            color=colours[d],
                            linestyle="None",
                            label=f"payout fit: {c.dist_type[d]}",
                        )
                    else:
                        if plot_params.show_the_curve:
                            ax.plot(c.dist_values[d].xthe, c.dist_values[d].ythe, color="black")
                        ax.plot(
                            c.dist_values[d].xact,
                            c.dist_values[d].yact,
                            marker="o",
                            color=colours[d],
                            linestyle="None",
                            label=f"payout fit: {c.dist_type[d]}",
                        )
                if len(c.merged_dist) > 0:
                    if c.plot_log_scale:
                        if plot_params.show_the_curve:
                            ax.semilogx(c.xthe, c.merged_dist_the, color="black")
                        ax.semilogx(
                            c.xact,
                            c.merged_dist,
                            marker="x",
                            color="g",
                            linewidth=0.8,
                            label="combined payout fit",
                        )
                    else:
                        if plot_params.show_the_curve:
                            ax.plot(c.xthe, c.merged_dist_the, color="black")
                        ax.plot(
                            c.xact,
                            c.merged_dist,
                            marker="x",
                            color="g",
                            linewidth=0.8,
                            label="combined payout fit",
                        )

                if plot_params.show_solution and len(c.solved_weights) > 0:
                    if c.plot_log_scale:
                        ax.semilogx(
                            c.xact,
                            c.solved_weights,
                            marker="x",
                            color="r",
                            linewidth=0.8,
                            label="optimizer solution",
                        )
                    else:
                        ax.plot(
                            c.xact,
                            c.solved_weights,
                            marker="x",
                            color="r",
                            linewidth=0.8,
                            label="optimizer solution",
                        )

                ax.set_xlim([state.plot_params[i].xmin, state.plot_params[i].xmax])
                ax.legend()
                ax.grid(True)
                ax.set_xlabel("payout value")
                ax.set_ylabel("payout probability")
                containter.pyplot(fig)
                st.space()
