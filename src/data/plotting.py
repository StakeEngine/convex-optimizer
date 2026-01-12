import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import streamlit as st
from src.class_setup.state import AppState
from src.util.utils import hit_rates_ranges


def render_plots(state: AppState):
    if not state.set_params:
        return

    for i, c in enumerate(state.criteria_list):
        if len(c.xact) <= 10:
            continue

        with st.expander(f"Criterion: {c.name}", expanded=True):

            plot_params = state.plot_params[i]

            ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1, 1, 1, 1, 1])
            plot_params.xmin = ctrl1.number_input("xmin", value=plot_params.xmin or 0.0, key=f"xmin_{i}")
            plot_params.xmax = ctrl2.number_input(
                "xmax", value=plot_params.xmax or 10 * state.cost, key=f"xmax_{i}"
            )
            plot_params.show_the_curve = ctrl3.checkbox("Theoretical", plot_params.show_the_curve, key=f"the_{i}")
            plot_params.normalize_all = ctrl4.checkbox("Normalize", plot_params.normalize_all, key=f"norm_{i}")
            plot_params.log_scale = ctrl5.checkbox("Log X", plot_params.log_scale, key=f"log_{i}")

            if len(c.merged_dist) > 0:
                plot_params.base_curves = st.checkbox("Show distribution components", True, key=f"parts_{i}")

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
            colours = cm.get_cmap("tab10")(np.linspace(0, 1, max(c.num_dists, 3)))

            def normalize(y):
                return np.asarray(y) / sum(y) if plot_params.normalize_all else y

            plot_fn = ax.semilogx if plot_params.log_scale else ax.plot

            if plot_params.base_curves:
                for d in range(c.num_dists):
                    ythe = normalize(c.dist_values[d].ythe)
                    yact = normalize(c.dist_values[d].yact)

                    if plot_params.show_the_curve:
                        plot_fn(
                            c.dist_values[d].xthe,
                            ythe,
                            color=colours[d],
                            alpha=0.35,
                            linewidth=1.5,
                        )

                    plot_fn(
                        c.dist_values[d].xact,
                        yact,
                        marker="o",
                        linestyle="None",
                        markersize=5,
                        color=colours[d],
                        label=f"{c.dist_type[d]}",
                    )

            if len(c.merged_dist) > 0:
                ythe = normalize(c.merged_dist_the)
                yact = normalize(c.merged_dist)

                plot_fn(
                    c.xact,
                    yact,
                    marker="x",
                    markersize=6,
                    linestyle="--",
                    linewidth=1.8,
                    color="black",
                    label="Combined",
                )

                if plot_params.show_the_curve:
                    plot_fn(
                        c.xthe,
                        ythe,
                        linestyle="--",
                        linewidth=1.2,
                        color="black",
                        alpha=0.6,
                    )

            if plot_params.show_solution and len(c.solved_weights) > 0:
                ysol = normalize(c.solved_weights)
                plot_fn(
                    c.xact,
                    ysol,
                    marker="s",
                    linestyle="None",
                    color="red",
                    markersize=5,
                    label="Solution",
                )

            ax.set_xlim(plot_params.xmin, plot_params.xmax)
            ax.set_xlabel("Payout Multiplier", size=11)
            ax.set_ylabel("Probability of Payout", size=11)
            xmin, xmax = ax.get_xlim()
            ymins, ymaxs = [], []

            for line in ax.lines:
                x = np.asarray(line.get_xdata())
                y = np.asarray(line.get_ydata())
                mask = (x >= xmin) & (x <= xmax)
                if mask.any():
                    ymins.append(y[mask].min())
                    ymaxs.append(y[mask].max())

            ymin, ymax = min(ymins), max(ymaxs)
            pad = 0.05 * (ymax - ymin) if ymax > ymin else ymax * 0.1
            ax.set_ylim(max(0, ymin - pad), ymax + pad)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, ncol=2)
            fig.tight_layout()

            st.pyplot(fig, width="stretch")

            if len(c.xact) > 1:
                with st.expander("Target distribution hit-rates"):
                    if len(c.merged_dist) > 0:
                        x, y = c.xact, normalize(c.merged_dist)
                    else:
                        x, y = c.dist_values[0].xact, normalize(c.dist_values[0].yact)

                    hr_ranges = hit_rates_ranges(x, y, False)
                    df = pd.DataFrame(
                        {
                            "Win Range": hr_ranges.keys(),
                            "Hit Rate": [f"{v:.2e}" if v > 1e6 else round(v, 4) for v in hr_ranges.values()],
                        }
                    )
                    st.dataframe(df, width="stretch")
