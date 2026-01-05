import streamlit as st
import numpy as np
import cvxpy as cp
from src.class_setup.state import AppState
from src.computation.compute_params import reset_optimizer_and_merge
from src.util.utils import hit_rates_ranges


def render_optimizer_settings(state: AppState):
    for j, s in enumerate(state.opt_settings):
        with st.container(border=True):
            st.write(f"Criteria: {state.criteria_list[j].name}")
            s.smoothness = st.slider(
                f"Distribution smoothness: {j}",
                0.0,
                100.0,
                s.smoothness,
                0.5,
                key=f"set_smoothness_{j}",
                on_change=reset_optimizer_and_merge,
                args=(state,),
            )

            s.kl_divergence = st.slider(
                f"KL Divergence weight: {j}",
                0.1,
                100.0,
                s.kl_divergence,
                0.1,
                key=f"set_kl_divergence_{j}",
                on_change=reset_optimizer_and_merge,
                args=(state,),
            )


def solve_optimizer(state: AppState):
    for m, c in enumerate(state.criteria_list):
        if c.num_dists == 2:
            w0 = np.array(c.merged_dist)
        else:
            w0 = np.array(c.dist_values[0].yact)

        eps = 1e-9
        w0 = (w0 + eps) / (w0 + eps).sum()

        n = len(c.xact)
        w = cp.Variable(n)

        kl = cp.sum(cp.rel_entr(w, w0))
        smooth = cp.sum_squares(w[:-1] - w[1:])

        # todo: create a better objective function
        obj = cp.Minimize(state.opt_settings[m].kl_divergence * kl + state.opt_settings[m].smoothness * smooth)

        cons = [
            w >= 0,
            cp.sum(w) == (1.0 / c.hr),
            c.xact @ w == c.rtp,
        ]

        cp.Problem(obj, cons).solve(verbose=True)

        c.solved_weights = w.value
        c.solution_metrics = {
            "target_sum": 1.0 / c.hr,
            "actual_sum": float(np.sum(w.value)),
            "rtp": float(c.xact @ w.value),
        }
        state.plot_params[m].show_solution = True

    state.optimization_success = True


def render_optimizer_results(state: AppState):
    with st.expander("Optimized solution result"):
        for m, c in enumerate(state.criteria_list):
            st.subheader(f"Criterion {m + 1}")

            st.write(state.opt_settings[m])
            st.write("Solved weights:")
            st.write(c.solved_weights)

            metrics = c.solution_metrics
            st.write(f"Target weight sum: {metrics['target_sum']}")
            st.write(f"Actual weight sum: {metrics['actual_sum']}")
            st.write(f"Computed RTP: {metrics['rtp']}")

            c.criteria_hr_dict = hit_rates_ranges(c.xact, c.solved_weights)

    with st.expander("Criteria-Seperated Hit-Rates"):
        for c in state.criteria_list:
            hr_json = {}
            for r, val in c.criteria_hr_dict.items():
                hr_json[str(r)] = val
            st.json(hr_json)


def merge_solutions(state: AppState):
    criteria_prob = {}
    final_lookup = []
    zero_ids = set([i + state.book_offset for i in range(state.lookup_length)])  # assume zero index, fix this

    contain = st.container(border=True)
    for i, d in enumerate(state.criteria_list):

        criteria_prob[d.name] = 1.0 / d.hr
        unique_weights = dict(zip(d.dist_values[0].xact, d.solved_weights))

        d.optimized_unique_distribution = unique_weights

        unique_counter = {}
        for p in state.dist_objects[i].payouts:
            unique_counter[p] = unique_counter.get(p, 0) + 1

        d.unique_payout_counter = unique_counter
        d.lookup_segment_length = len(d.xact)

        d.optimized_final_distribution = {}
        for p, w in unique_weights.items():
            d.optimized_final_distribution[p] = w * (1.0 / unique_counter[p])

        sanity_rtp = 0.0
        cumulative_prob = 0.0

        for sim, pay in zip(state.dist_objects[i].book_ids, state.dist_objects[i].payouts):
            w = d.optimized_final_distribution[pay]
            final_lookup.append((sim, w, int(pay * 100)))
            sanity_rtp += pay * w
            cumulative_prob += w
        zero_ids -= set(state.dist_objects[i].book_ids)

        contain.write(f"RTP after sim split: {round(sanity_rtp,5)}")
        contain.write(f"Hit-Rate after sim split: {round(1.0/cumulative_prob, 5)}")
        contain.write(f"Expected Hit-Rate: {d.hr}")

    # get 0 wins
    idv_zero_prob = state.zero_prob / len(state.zero_ids)
    for i in list(state.zero_ids):
        final_lookup.append((i, idv_zero_prob, 0))

    sorted_lookup = sorted(final_lookup, key=lambda x: x[0])
    contain.write(f"Zero-Weight: {state.zero_prob}")

    weight_array = [x[1] for x in sorted_lookup]
    pay_array = [round(x[2] / 100, 2) for x in sorted_lookup]
    final_weights = [int((2**state.weight_scale)) * w for w in weight_array]

    state.hr_ranges = hit_rates_ranges(pay_array, final_weights)

    state.final_optimized_lookup = sorted_lookup
