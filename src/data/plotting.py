import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import streamlit as st
from src.class_setup.state import AppState
from src.util.utils import hit_rates_ranges


def render_plots(state: AppState, containter):
    # colours = ["blue", "purple", "orange", "black"]
    if state.set_params:
        for i, c in enumerate(state.criteria_list):
            colours = cm.get_cmap("viridis", c.num_dists)(np.linspace(0, 1, c.num_dists))
            if len(c.xact) > 10:  # look at why there is sometimes a 0 win for set win amount
                with st.container():
                    plot_params = state.plot_params[i]
                    xmin_disp, xmax_disp = containter.columns(2)
                    plot_params.xmin = xmin_disp.number_input(
                        label="min x-axis", value=0, key=f"min_x_plot_{i}", width=150
                    )
                    plot_params.xmax = xmax_disp.number_input(
                        label="max x-axis", value=10 * state.cost, step=10.0, key=f"max_x_plot_{i}", width=150
                    )

                    c1, c2, c3, c4 = containter.columns(4)
                    plot_params.show_the_curve = c1.checkbox(
                        "Plot Theoretical Curve",
                        value=plot_params.show_the_curve,
                        key=f"plot_the_{i}",
                    )
                    plot_params.normalize_all = c2.checkbox(
                        "Normalize Curves",
                        value=plot_params.normalize_all,
                        key=f"normalize_{i}",
                    )
                    plot_params.log_scale = c3.checkbox(
                        "Log Scale",
                        value=plot_params.log_scale,
                        key=f"log_scale_{i}",
                    )
                    if len(c.merged_dist) > 0:
                        plot_params.base_curves = c4.checkbox(
                            "Plot Distribution Parts", value=True, key=f"plot_parts_{i}"
                        )

                    if c.num_dists == 1 and c.dist_type[i] in ["Quadratic", "Linear", "Rect"]:
                        st.warning(
                            "Parabolic, Linear and Exponential fits should only be used for mixed distribtuions.\n\nOptimization will likely fail with 1 distribution selected."
                        )

                    fig, ax = plt.subplots()
                    if plot_params.base_curves:
                        for d in range(c.num_dists):
                            ytot_the, ytot_act = 1.0, 1.0
                            if plot_params.normalize_all:
                                ytot_the = sum(c.dist_values[d].ythe)
                                ytot_act = sum(c.dist_values[d].yact)

                            ythe = [x / ytot_the for x in list(c.dist_values[d].ythe)]
                            yact = [x / ytot_act for x in list(c.dist_values[d].yact)]

                            if plot_params.log_scale:
                                if plot_params.show_the_curve:
                                    ax.semilogx(c.dist_values[d].xthe, ythe, color="black")
                                ax.semilogx(
                                    c.dist_values[d].xact,
                                    yact,
                                    marker="o",
                                    color=colours[d],
                                    linestyle="None",
                                    label=f"payout fit: {c.dist_type[d]}",
                                )
                            else:
                                if plot_params.show_the_curve:
                                    ax.plot(c.dist_values[d].xthe, ythe, color="black")
                                ax.plot(
                                    c.dist_values[d].xact,
                                    yact,
                                    marker="o",
                                    color=colours[d],
                                    linestyle="None",
                                    label=f"payout fit: {c.dist_type[d]}",
                                )
                    if len(c.merged_dist) > 0:
                        ymerge_the, ymerge_act = 1.0, 1.0
                        if plot_params.normalize_all:
                            ymerge_the = sum(c.merged_dist_the)
                            ymerge_act = sum(c.merged_dist)
                        ythe_merge = [x / ymerge_the for x in c.merged_dist_the]
                        yact_merge = [x / ymerge_act for x in c.merged_dist]

                        if plot_params.log_scale:
                            if plot_params.show_the_curve:
                                ax.semilogx(c.xthe, ythe_merge, color="black")
                            ax.semilogx(
                                c.xact,
                                yact_merge,
                                marker="x",
                                color="g",
                                linewidth=0.8,
                                label="combined payout fit",
                            )
                        else:
                            if plot_params.show_the_curve:
                                ax.plot(c.xthe, ythe_merge, color="black")
                            ax.plot(
                                c.xact,
                                yact_merge,
                                marker="x",
                                color="g",
                                linewidth=0.8,
                                label="combined payout fit",
                            )

                    if plot_params.show_solution and len(c.solved_weights) > 0:
                        ysol_tot = 1.0
                        if plot_params.normalize_all:
                            ysol_tot = sum(c.solved_weights)
                        ysol = [x / ysol_tot for x in c.solved_weights]
                        if plot_params.log_scale:
                            ax.semilogx(
                                c.xact,
                                ysol,
                                marker="x",
                                color="r",
                                linewidth=0.8,
                                label="optimizer solution",
                            )
                        else:
                            ax.plot(
                                c.xact,
                                ysol,
                                marker="x",
                                color="r",
                                linewidth=0.8,
                                label="optimizer solution",
                            )

                    ax.set_xlim([state.plot_params[i].xmin, state.plot_params[i].xmax])
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True)
                    ax.legend()
                    ax.grid(True)
                    ax.set_xlabel("payout value")
                    ax.set_ylabel("payout probability")

                    containter.pyplot(fig)
                    st.space()

                    if len(c.xact) > 1:
                        with containter.expander("Target Distribtuion Hit-Rates"):
                            if len(c.merged_dist) > 0:
                                x, y = c.xact, yact_merge
                            else:
                                x, y = c.dist_values[d].xact, c.dist_values[0].yact

                            hr_ranges = hit_rates_ranges(x, y, False)
                            df = pd.DataFrame(
                                {"Win Range": hr_ranges.keys(), "Hit-Rate": hr_ranges.values()},
                            )
                            df["Hit-Rate"] = df["Hit-Rate"].map(lambda x: f"{x:.2e}" if x > 1e6 else round(x, 2))
                            st.table(df)
