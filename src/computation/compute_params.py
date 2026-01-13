import streamlit as st
import numpy as np
from src.class_setup.state import (
    AppState,
    PlotSettings,
    ConvexOptSetup,
)
from src.util.utils import extract_ids, read_csv, calculate_params, DIST_PARAM_CLASSES
from src.computation.math_functions import calculate_act_expectation
from src.computation.distribution_assign import reset_optimizer_and_merge
from src.class_setup.state import DistributionInput
from src.class_setup.models import LogNormalParams
from src.computation.distribution_assign import (
    assign_log_normal,
    assign_exponential,
    assign_gaussian,
    assign_quadratic,
    assign_linear,
    assign_rect,
)


class SummaryGame:
    def __init__(self, lookup_length: int, zero_id_len: int):
        self.lookup_length = lookup_length
        self.zero_id_len = zero_id_len
        self.mode_summary = []


class SummaryModes:
    def __init__(self, mode_ids: int = None, unique_payouts: int = None):
        self.mode_ids = mode_ids
        self.unique_payouts = unique_payouts


def render_compute_params(state: AppState):
    summaryObj = SummaryGame(None, None)

    if len(state.lut_file) > 0 and not (state.lut_read_complete):
        state.all_payout_ints = read_csv(state.lut_file)
        max_payout_float = round(max(state.all_payout_ints) / 100, 2)
        state.max_payout = max_payout_float
        state.lut_read_complete = True

        assigned, unassigned = [], []
        for c in state.criteria_list:
            (unassigned if c.hr is None else assigned).append(c)

        if len(unassigned) > 1:
            st.error(
                "Only one criteria can have unassigned hit-rate; all others must have 2 of 3 parameters applied"
            )
            return

        used_prob = sum(1.0 / c.hr for c in assigned)
        remaining_prob = 1.0 - used_prob

        if remaining_prob < 0:
            st.error("Assigned hit-rates exceed total probability")
            return

        if unassigned:
            new_hr = 1.0 / remaining_prob
            c = unassigned[0]
            c.hr = new_hr

        state.zero_prob -= sum(1.0 / c.hr for c in state.criteria_list)

    for i, o in enumerate(state.dist_objects):
        # extract book ids
        if not o.book_ids:
            o.book_ids, o.payouts, state.lookup_length, state.zero_ids = extract_ids(state, o.criteria)
            if summaryObj.zero_id_len is None or summaryObj.lookup_length is None:
                summaryObj.zero_id_len = len(state.zero_ids)
                summaryObj.lookup_length = state.lookup_length

            o.unique_payouts = sorted(list(set(o.payouts)))
            if len(o.unique_payouts) == 1:
                with st.container():
                    st.warning(
                        f"Criteria [{o.criteria}] has 1 unique payout - distribution fits not avalaliable. \n\nProbabilty will be automatically assigned based on hit-rate and RTP contribution."
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

    with st.expander("Criteria Summary"):
        for i, c in enumerate(state.criteria_list):
            st.write(
                f"{c.name}:\n\nRTP:{c.rtp} -- Av Win: {c.av} -- Hit-Rate:{c.hr} \n\nNum IDs: {len(state.dist_objects[i].book_ids)}"
            )
        if state.zero_prob not in [0, 1]:
            st.write(f"Zero Wins:\n\nHit-Rate:{round(1.0/state.zero_prob,2)} -- Num IDs: {len(state.zero_ids)}")

    with st.expander("Game/Mode Split Summary"):
        if state.set_params:
            st.write(
                f"Game Info: \n\nLookup length: {summaryObj.lookup_length} \n\nNum Zero-IDs: {summaryObj.zero_id_len}"
            )
            st.warning(f"{len(remaining_sim_ids)} non-zero (unused) ids without matching criteria")
        for i, o in enumerate(state.dist_objects):
            st.space()
            with st.container():
                st.write(
                    f"Mode Info ({o.criteria}): \n\n{summaryObj.mode_summary[i].mode_ids} mode sumulations found"
                )
                st.write(f"{summaryObj.mode_summary[i].unique_payouts} unique payouts in mode")

    if state.set_params:
        if len(state.criteria_list) == 1 and not state.mode_contains_zero_criteria:
            state.criteria_list[0].hr = 1.0


def merge_dist_pdf(pdf1, pdf2, mix_factor, criteria_scale=1.0):
    final_pdf = []
    for x, y in zip(pdf1, pdf2):
        final_pdf.append((mix_factor * x) + ((1 - mix_factor) * y))

    final_pdf = np.asarray(final_pdf)
    final_pdf = np.clip(final_pdf, 0, None)
    final_pdf /= final_pdf.sum()

    return (final_pdf * criteria_scale).tolist()


def merge_multi_dist(
    payouts: list,
    pdf_dict: dict[int, list],
    mix_factors: list[float],
    criteria_scale: float,
):
    assert len(pdf_dict) == len(mix_factors), "weight array length does not match number of input pdfs"

    final_pdf = []

    for idx in range(len(payouts)):
        pdf_val = 0.0
        for k, w in enumerate(mix_factors):
            pdf_val += w * pdf_dict[k][idx]
        final_pdf.append(pdf_val)

    pdf = np.asarray(final_pdf)
    pdf = np.clip(pdf, 0, None)

    pdf /= pdf.sum()

    return (pdf * criteria_scale).tolist()


def ensure_dist_params(c, d):
    dist_type = c.dist_type[d]
    cls = DIST_PARAM_CLASSES[dist_type]

    attr = f"dist{d}_params"
    params = getattr(c, attr, None)
    if params is None:
        params = cls()
        setattr(c, attr, params)
        return params

    if isinstance(params, cls):
        return params

    new_params = cls()

    for k, v in vars(params).items():
        if hasattr(new_params, k):
            setattr(new_params, k, v)

    setattr(c, attr, new_params)
    return new_params


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
                st.subheader(f"{c.name}")
                if f"checkbox_{i}" not in st.session_state:
                    st.session_state[f"checkbox_{i}"] = c.multi_dist

                c.multi_dist = st.checkbox(
                    "Use Multiple Distributions",
                    key=f"checkbox_{i}",
                )

                if c.multi_dist:
                    c.num_dists = st.number_input(
                        "Distribution Count", 2, 5, 2, 1, width=150, key=f"dists_used_{i}"
                    )
                    if c.num_dists == 2:
                        c.dist1_mix = st.number_input(
                            "Dist 1 Weight Factor", 0.0, 1.0, value=0.5, step=0.1, key=f"dist1_mix_{i}", width=150
                        )
                        c.dist2_mix = 1.0 - c.dist1_mix
                        st.write(f"Dist 2 Weight: {c.dist2_mix}")
                    elif c.num_dists > 2:
                        key = f"multi_dist_weights_{i}"
                        st.session_state[key] = [0.5] * c.num_dists

                        c.multi_dist_weights = st.session_state[key]
                        for start in range(0, c.num_dists, 3):
                            cols = st.columns(3)
                            for j in range(3):
                                idx = start + j
                                if idx >= c.num_dists:
                                    break

                                with cols[j]:
                                    c.multi_dist_weights[idx] = st.number_input(
                                        f"Dist {idx+1} weight",
                                        min_value=0.0,
                                        max_value=100.0,
                                        value=0.5,
                                        width=100,
                                        key=f"dist_weight_{i}_{idx}",
                                    )
                        tw = sum(c.multi_dist_weights)
                        c.multi_dist_weights = [x / tw for x in c.multi_dist_weights]
                    else:
                        c.num_dists = 1

                for d in range(c.num_dists):
                    c.dist_type[d] = st.selectbox(
                        "Function Type",
                        list(DIST_PARAM_CLASSES.keys()),
                        key=f"dist_type_{i}_{d}",
                        on_change=reset_optimizer_and_merge,
                        args=(state,),
                        width=150,
                    )
                    dist_params = ensure_dist_params(c, d)

                    ythe, yact = [], []
                    with st.container(border=True):
                        st.write(f"Criteria: {c.name}")

                        match c.dist_type[d]:
                            case "Log-Normal":
                                ythe, yact = assign_log_normal(state, dist_params, c, i, d)
                            case "Gaussian":
                                ythe, yact = assign_gaussian(state, dist_params, c, i, d)
                            case "Exponential":
                                ythe, yact = assign_exponential(state, dist_params, c, i, d)
                            case "Quadratic":
                                ythe, yact = assign_quadratic(state, dist_params, c, i, d)
                            case "Linear":
                                ythe, yact = assign_linear(state, dist_params, c, i, d)
                            case "Rect":
                                ythe, yact = assign_rect(state, dist_params, c, i, d)

                        # setattr(c, f"dist{d}_params", dist_params)
                    c.dist_values[d] = DistributionInput(c.dist_type[d], c.xthe, c.xact, ythe, yact)

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
                elif c.num_dists > 2:
                    ythe, yact = {}, {}
                    for n in range(c.num_dists):
                        ythe[int(n)] = c.dist_values[n].ythe
                        yact[int(n)] = c.dist_values[n].yact

                    ymergedact = merge_multi_dist(c.xact, yact, c.multi_dist_weights, 1.0 / c.hr)
                    ymergedthe = merge_multi_dist(c.xthe, ythe, c.multi_dist_weights, 1.0 / c.hr)
                    c.effective_rtp, c.effective_pdf = calculate_act_expectation(c.xact, ymergedact, state.cost)
                    c.merged_dist = ymergedact
                    c.merged_dist_the = ymergedthe

                st.write(f"Actual Effective RTP Target: {round((1.0/c.hr) * c.effective_rtp,5)}")
            else:
                st.write(
                    f"Criteria [{c.name}] has only 1 average win amount [{state.dist_objects[i].unique_payouts[0]}]"
                )
                prob = 1.0 / c.hr
                c.dist_values[0] = DistributionInput("fixed_amt", [c.av], [c.av], prob, prob)
                c.yact, c.ythe = [prob], [prob]

        while len(state.opt_settings) < len(state.criteria_list):
            state.opt_settings.append(ConvexOptSetup(1.0, 1.0, [], []))
        state.opt_settings[i] = ConvexOptSetup(
            1.0,
            1.0,
            c.xact,
            list(np.ones(len(x))),
        )
