import streamlit as st
import numpy as np
from src.class_setup.state import (
    AppState,
    PlotSettings,
    ConvexOptSetup,
)
from src.util.utils import extract_ids, read_csv, calculate_params
from src.computation.math_functions import (
    calculate_theoretical_expectation,
    calculate_mu_from_mode,
    get_log_normal_pdf,
    get_gaussian_pdf,
    get_exp_pdf,
    calculate_act_expectation,
)
from src.class_setup.state import DistributionInput
from src.class_setup.models import LogNormalParams


class SummaryGame:
    def __init__(self, lookup_length: int, zero_id_len: int):
        self.lookup_length = lookup_length
        self.zero_id_len = zero_id_len
        self.mode_summary = []


class SummaryModes:
    def __init__(self, mode_ids: int = None, unique_payouts: int = None):
        self.mode_ids = mode_ids
        self.unique_payouts = unique_payouts


def reset_optimizer_and_merge(state: AppState):
    state.optimization_success = False
    state.run_optimizer = False
    state.merge_solutions = False
    state.final_optimized_lookup = []
    state.hr_ranges = {}
    for c in state.criteria_list:
        c.solved_weights = []
        c.merged_dist = []
        c.merged_dist_the = []


def render_compute_params(state: AppState):
    summaryObj = SummaryGame(None, None)

    if len(state.lut_file) > 0 and not (state.lut_read_complete):
        state.all_payout_ints = read_csv(state.lut_file)
        max_payout_float = round(max(state.all_payout_ints) / 100, 2)
        state.max_payout = max_payout_float
        state.lut_read_complete = True

    exclude_payouts = []
    if state.auto_assign_zero_hr:
        exclude_payouts = [0]

    for i, o in enumerate(state.dist_objects):
        # extract book ids
        if not o.book_ids:
            criteria_match = False
            for c in state.criteria_list:
                if c.name == o.criteria:
                    autosolve_c = c.auto_solve_zero_criteria
                    criteria_match = True
                    break
            if not criteria_match:
                raise RuntimeError("no criteria match")
            o.book_ids, o.payouts, state.lookup_length, state.zero_ids = extract_ids(
                state, o.criteria, exclude_payouts, autosolve_c
            )
            if summaryObj.zero_id_len is None or summaryObj.lookup_length is None:
                summaryObj.zero_id_len = len(state.zero_ids)
                summaryObj.lookup_length = state.lookup_length

            o.unique_payouts = sorted(list(set(o.payouts)))
            if len(o.unique_payouts) == 1:
                with st.container():
                    st.warning(
                        "Criteria has 1 unique payout - distribution fits not avalaliable. \n\nProbabilty will be automatically assigned based on hit-rate and RTP contribution."
                    )

        if len(o.unique_payouts) == 0 or len(o.book_ids) == 0:
            st.error(f"ERROR: could not find book ids / payouts for criteria: {o.criteria}")

        summaryObj.mode_summary.append(SummaryModes(len(o.book_ids), len(o.unique_payouts)))

        if len(state.log_normal_params) <= i:
            state.log_normal_params.append(LogNormalParams())
            state.plot_params.append(PlotSettings())

    remaining_sim_ids = set([i + state.book_offset for i in range(state.lookup_length)])
    for o in state.dist_objects:
        remaining_sim_ids -= set(o.book_ids)
    remaining_sim_ids -= set(state.zero_ids)

    if len(state.lut_file) > 0:
        state.disused_sims = list(remaining_sim_ids)
        state.disused_int_payouts = [
            int(state.all_payout_ints[i - state.book_offset]) for i in list(remaining_sim_ids)
        ]

    with st.expander("Game/Mode Split Summary"):
        st.write(
            f"Game Info: \n\nLookup length: {summaryObj.lookup_length} \n\nNum Zero-IDs: {summaryObj.zero_id_len}"
        )
        if state.set_params:
            st.warning(f"{state.lookup_length - len(remaining_sim_ids)} non-zero ids without matching criteria!")
        for i, o in enumerate(state.dist_objects):
            st.space()
            with st.container():
                st.write(
                    f"Mode Info ({o.criteria}): \n\n{summaryObj.mode_summary[i].mode_ids} mode sumulations found"
                )
                st.write(f"{summaryObj.mode_summary[i].unique_payouts} unique payouts in mode")

    # normalize hit-rates so HR of all modes = 1
    if state.set_params:
        if len(state.criteria_list) == 1 and not state.criteria_list[0].auto_solve_zero_criteria:
            state.criteria_list[0].hr = 1.0
        else:
            total_hr = sum([(1.0 / c.hr) for c in state.criteria_list]) + state.zero_prob
            for c in state.criteria_list:
                c.hr /= total_hr


def merge_dist_pdf(pdf1, pdf2, mix_factor, criteria_scale=1.0):
    final_pdf = []
    for x, y in zip(pdf1, pdf2):
        final_pdf.append((mix_factor * x) + ((1 - mix_factor) * y))

    final_pdf = np.asarray(final_pdf)
    final_pdf /= final_pdf.sum()

    return (final_pdf * criteria_scale).tolist()


def render_target_dist_params(state: AppState):
    for i, c in enumerate(state.criteria_list):
        if any([c.rtp is None, c.hr is None, c.av is None]):
            c.rtp, c.hr, c.av = calculate_params(c.rtp, c.hr, c.av, state.cost)

        dist_object = state.dist_objects[i]
        x = np.linspace(
            min(dist_object.unique_payouts),
            max(dist_object.unique_payouts),
            int(max(dist_object.unique_payouts) / state.win_step_size),
        )

        c.xthe = [round(y, 2) for y in x]
        c.xact = [round(y, 2) for y in dist_object.unique_payouts]
        with st.sidebar:
            if len(state.dist_objects[i].unique_payouts) > 1:
                if f"checkbox_{i}" not in st.session_state:
                    st.session_state[f"checkbox_{i}"] = c.is_2_dist

                c.is_2_dist = st.checkbox(
                    "Use 2 distribtuions",
                    key=f"checkbox_{i}",
                )

                if c.is_2_dist:
                    c.num_dists = 2
                    if f"dist1_mix_{i}" not in st.session_state:
                        st.session_state[f"dist1_mix_{i}"] = c.dist1_mix

                    c.dist1_mix = st.number_input(
                        "Dist 1 Weight Factor",
                        0.0,
                        1.0,
                        key=f"dist1_mix_{i}",
                    )
                    c.dist2_mix = 1.0 - c.dist1_mix
                else:
                    c.num_dists = 1

                for d in range(c.num_dists):
                    dist_type = st.radio(
                        "Distribution Type",
                        ["Log-Normal", "Gaussian", "Exponential"],
                        key=f"dist_type_{i}_{d}",
                        on_change=reset_optimizer_and_merge,
                        args=(state,),
                    )
                    c.dist_type[d] = st.session_state[f"dist_type_{i}_{d}"]
                    # change_dist_params(state, dist_type)
                    dist_params = c.dist1_params
                    if d == 1:
                        dist_params = c.dist2_params

                    ythe, yact = [], []
                    with st.container(border=True):
                        st.write(f"Criteria: {c.name}")
                        if c.dist_type[d] == "Log-Normal":
                            def_mode = 1.0
                            def_var = 0.1
                            def_scale = 1.0  # to do: put in a global scaling factor
                            if f"log_mode_{i}_{d}" in st.session_state:
                                def_mode = st.session_state[f"log_mode_{i}_{d}"]
                            if f"log_var_{i}_{d}" in st.session_state:
                                def_var = st.session_state[f"log_var_{i}_{d}"]
                            if f"log_mu_{i}_{d}" in st.session_state:
                                def_mean = st.session_state[f"log_mu_{i}_{d}"]
                            dist_params.mode = st.number_input(
                                "Distribution Mode",
                                0.01 * state.cost,
                                1000.0 * state.cost,
                                def_mode,
                                0.01 * state.cost,
                                key=f"log_mode_{i}_{d}",
                                on_change=reset_optimizer_and_merge,
                                args=(state,),
                            )  # label,min,max,start,imcrement
                            dist_params.std = st.number_input(
                                "Distribution Variance",
                                0.01,
                                100.0,
                                def_var,
                                0.01,
                                key=f"log_var_{i}_{d}",
                                on_change=reset_optimizer_and_merge,
                                args=(state,),
                            )
                            # dist_params.scale = st.slider(
                            #     "Distribution Scale",
                            #     0.1,
                            #     10.0,
                            #     def_scale,
                            #     0.1,
                            #     key=f"log_mu_{i}_{d}",
                            #     on_change=reset_optimizer_and_merge,
                            #     args=(state,),
                            # )
                            dist_params.mean = calculate_mu_from_mode(dist_params.mode, dist_params.std)
                            dist_params.the_exp = calculate_theoretical_expectation(
                                dist_params.mode, dist_params.std
                            )

                            st.text(f"Target Mean: {round(dist_params.the_exp *(1.0 / c.hr) ,3)}")

                            # compute the probability distribution
                            ythe = get_log_normal_pdf(c.xthe, dist_params.mode, dist_params.std, 1.0 / c.hr)
                            yact = get_log_normal_pdf(c.xact, dist_params.mode, dist_params.std, 1.0 / c.hr)

                        elif c.dist_type[d] == "Gaussian":
                            def_mean = state.cost
                            def_std = 0.1
                            def_scale = 1.0
                            if f"gauss_mode_{i}_{d}" in st.session_state:
                                def_mean = st.session_state[f"gauss_mode_{i}_{d}"]
                            if f"gauss_var_{i}_{d}" in st.session_state:
                                def_std = st.session_state[f"gauss_var_{i}_{d}"]
                            if f"gauss_mu_{i}_{d}" in st.session_state:
                                def_mean = st.session_state[f"gauss_mu_{i}_{d}"]
                            dist_params.mean = st.number_input(
                                "Distribution Mode",
                                0.01 * state.cost,
                                1000.0 * state.cost,
                                def_mean,
                                0.01 * state.cost,
                                key=f"gauss_mode_{i}_{d}",
                                on_change=reset_optimizer_and_merge,
                                args=(state,),
                            )
                            dist_params.std = st.number_input(
                                "Distribution Standard Deviation",
                                0.01,
                                1000.0,
                                def_std,
                                0.01,
                                key=f"gauss_var_{i}_{d}",
                                on_change=reset_optimizer_and_merge,
                                args=(state,),
                            )
                            # dist_params.scale = st.slider(
                            #     "Distribution Scale",
                            #     0.1,
                            #     10.0,
                            #     def_scale,
                            #     0.1,
                            #     key=f"gauss_mu_{i}_{d}",
                            #     on_change=reset_optimizer_and_merge,
                            #     args=(state,),
                            # )
                            ythe = get_gaussian_pdf(c.xthe, dist_params.mean, dist_params.std, 1.0 / c.hr)
                            yact = get_gaussian_pdf(c.xact, dist_params.mean, dist_params.std, 1.0 / c.hr)

                        elif c.dist_type[d] == "Exponential":
                            def_power = 1.0
                            def_scale = 1.0
                            if f"exp_mode_{i}_{d}" in st.session_state:
                                def_power = st.session_state[f"exp_mode_{i}_{d}"]
                            if f"exp_mu_{i}_{d}" in st.session_state:
                                def_mean = st.session_state[f"exp_mu_{i}_{d}"]
                            dist_params.power = st.number_input(
                                "Exponential Power",
                                0.01,
                                10.0,
                                def_power,
                                0.01,
                                key=f"exp_mode_{i}_{d}",
                                on_change=reset_optimizer_and_merge,
                                args=(state,),
                            )
                            # dist_params.scale = st.slider(
                            #     "Distribution Scale",
                            #     0.1,
                            #     10.0,
                            #     def_scale,
                            #     0.1,
                            #     key=f"exp_mu_{i}_{d}",
                            #     on_change=reset_optimizer_and_merge,
                            #     args=(state,),
                            # )

                            ythe = get_exp_pdf(c.xthe, dist_params.power, 1.0 / c.hr)
                            yact = get_exp_pdf(c.xact, dist_params.power, 1.0 / c.hr)

                    if d == 0:
                        c.dist1_params = dist_params
                    elif d == 1:
                        c.dist2_params = dist_params

                    c.dist_values[d] = DistributionInput(dist_type, c.xthe, c.xact, ythe, yact)
                # mix stuff here
                if c.num_dists == 1:
                    c.effective_rtp, c.effective_pdf = calculate_act_expectation(
                        c.xact, c.dist_values[0].yact, state.cost
                    )
                elif c.num_dists == 2:
                    ymergedact = merge_dist_pdf(
                        c.dist_values[0].yact, c.dist_values[1].yact, c.dist1_mix, 1.0 / c.hr
                    )
                    ymergedthe = merge_dist_pdf(
                        c.dist_values[0].ythe, c.dist_values[1].ythe, c.dist1_mix, 1.0 / c.hr
                    )

                    c.effective_rtp, c.effective_pdf = calculate_act_expectation(c.xact, ymergedact, state.cost)
                    c.merged_dist = ymergedact
                    c.merged_dist_the = ymergedthe

                st.write(f"Actual Effective RTP Target: {round((1.0/c.hr) * c.effective_rtp,5)}")
            else:
                st.write(
                    f"Criteria [{c.name}] has only 1 average win amount [{state.dist_objects[i].unique_payouts[0]}]"
                )
                prob = (1.0 / c.hr) / state.cost
                c.dist_values[0] = DistributionInput("fixed_amt", [c.av], [c.av], prob, prob)
                c.yact, c.ythe = [prob], [prob]

        if len(state.opt_settings) <= i:
            state.opt_settings.append(ConvexOptSetup(1.0, 1.0, c.xact, list(np.ones(len(x)))))
