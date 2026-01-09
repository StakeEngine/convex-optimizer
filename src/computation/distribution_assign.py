import streamlit as st
from src.class_setup.state import AppState, CriteraParams
from src.computation.math_functions import (
    get_log_normal_pdf,
    calculate_mu_from_mode,
    calculate_theoretical_expectation,
    get_gaussian_pdf,
    get_exp_pdf,
    get_quadratic_pdf,
    get_linear_pdf,
    get_rect_pdf,
)
from src.class_setup.models import (
    LogNormalParams,
    GaussianParams,
    ExponentialParams,
    ParabolaParams,
    LinearParams,
    RectParams,
)


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


def assign_log_normal(
    state: AppState, dist_params: LogNormalParams, criteria: CriteraParams, i: int, d: int
) -> list[list]:
    def_mode = 1.0
    def_var = 1.0
    if f"log_mode_{i}_{d}" in st.session_state:
        def_mode = st.session_state[f"log_mode_{i}_{d}"]
    if f"log_var_{i}_{d}" in st.session_state:
        def_var = st.session_state[f"log_var_{i}_{d}"]
    dist_params.mode = st.number_input(
        "Distribution Mode",
        0.0,
        1000.0 * state.cost,
        def_mode,
        0.1 * state.cost,
        key=f"log_mode_{i}_{d}",
        on_change=reset_optimizer_and_merge,
        args=(state,),
    )
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

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        dist_params.xmin = st.number_input("xmin", 0.0, max(criteria.xact), 0.0, key=f"log_xmin{i}_{d}")
    with cc2:
        dist_params.xmax = st.number_input(
            "xmax", 0.0, max(criteria.xact), max(criteria.xact), key=f"log_xmax_{i}_{d}"
        )
    with cc3:
        dist_params.linear_offset = st.number_input(
            "offset", 0.0, max(criteria.xact), 0.0, key=f"log_offset_{i}_{d}"
        )

    dist_params.mean = calculate_mu_from_mode(dist_params.mode, dist_params.std)
    dist_params.the_exp = calculate_theoretical_expectation(dist_params.mode, dist_params.std)

    st.text(f"Target Mean: {round(dist_params.the_exp *(1.0 / criteria.hr) ,3)}")

    # compute the probability distribution
    ythe = get_log_normal_pdf(
        criteria.xthe,
        dist_params.mode,
        dist_params.std,
        1.0 / criteria.hr,
        dist_params.xmin,
        dist_params.xmax,
        dist_params.linear_offset,
    )
    yact = get_log_normal_pdf(
        criteria.xact,
        dist_params.mode,
        dist_params.std,
        1.0 / criteria.hr,
        dist_params.xmin,
        dist_params.xmax,
        dist_params.linear_offset,
    )

    return ythe, yact


def assign_gaussian(
    state: AppState, dist_params: GaussianParams, criteria: CriteraParams, i: int, d: int
) -> list[list]:
    def_mean = state.cost
    def_std = 0.1
    if f"gauss_var_{i}_{d}" in st.session_state:
        def_std = st.session_state[f"gauss_var_{i}_{d}"]
    if f"gauss_mu_{i}_{d}" in st.session_state:
        def_mean = st.session_state[f"gauss_mu_{i}_{d}"]
    dist_params.mean = st.number_input(
        "Distribution Mode",
        -10.0 * state.cost,
        1000.0 * state.cost,
        def_mean,
        0.1 * state.cost,
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
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        dist_params.xmin = st.number_input("xmin", 0.0, max(criteria.xact), 0.0, key=f"gauss_xmin{i}_{d}")
    with cc2:
        dist_params.xmax = st.number_input(
            "xmax", 0.0, max(criteria.xact), max(criteria.xact), key=f"gauss_xmax_{i}_{d}"
        )
    with cc3:
        dist_params.linear_offset = st.number_input(
            "offset", -max(criteria.xact), max(criteria.xact), 0.0, key=f"gauss_offset_{i}_{d}"
        )

    ythe = get_gaussian_pdf(
        criteria.xthe,
        dist_params.mean,
        dist_params.std,
        1.0 / criteria.hr,
        dist_params.xmin,
        dist_params.xmax,
        dist_params.linear_offset,
    )
    yact = get_gaussian_pdf(
        criteria.xact,
        dist_params.mean,
        dist_params.std,
        1.0 / criteria.hr,
        dist_params.xmin,
        dist_params.xmax,
        dist_params.linear_offset,
    )

    return ythe, yact


def assign_exponential(
    state: AppState, dist_params: ExponentialParams, criteria: CriteraParams, i: int, d: int
) -> list[list]:

    def_power = 1.0
    if f"exp_mode_{i}_{d}" in st.session_state:
        def_power = st.session_state[f"exp_mode_{i}_{d}"]
    dist_params.power = st.number_input(
        "Exponential Power",
        0.01,
        10.0,
        def_power,
        0.1,
        key=f"exp_mode_{i}_{d}",
        format="%f",
        on_change=reset_optimizer_and_merge,
        args=(state,),
    )
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        dist_params.xmin = st.number_input("xmin", 0.0, max(criteria.xact), 0.0, key=f"exp_xmin{i}_{d}")
    with cc2:
        dist_params.xmax = st.number_input(
            "xmax", 0.0, max(criteria.xact), max(criteria.xact), key=f"exp_xmax_{i}_{d}"
        )
    with cc3:
        dist_params.linear_offset = st.number_input(
            "offset", -max(criteria.xact), max(criteria.xact), 0.0, key=f"exp_offset_{i}_{d}"
        )

    ythe = get_exp_pdf(
        criteria.xthe,
        dist_params.power,
        1.0 / criteria.hr,
        dist_params.xmin,
        dist_params.xmax,
        dist_params.linear_offset,
    )
    yact = get_exp_pdf(
        criteria.xact,
        dist_params.power,
        1.0 / criteria.hr,
        dist_params.xmin,
        dist_params.xmax,
        dist_params.linear_offset,
    )

    return ythe, yact


def assign_quadratic(
    state: AppState, dist_params: ParabolaParams, criteria: CriteraParams, i: int, d: int
) -> list[list]:
    def_quad = 1.0
    def_lin = 1.0
    if f"quadratic_quad_{i}_{d}" in st.session_state:
        def_quad = st.session_state[f"quadratic_quad_{i}_{d}"]
    if f"quadratic_lin_{i}_{d}" in st.session_state:
        def_lin = st.session_state[f"quadratic_lin_{i}_{d}"]

    normalize = st.checkbox("Normalize", False, key="quadratic_norm")
    if not normalize:
        dist_params.quad_coef = st.number_input(
            "Square Coefficient",
            -100 * state.cost,
            100.0 * state.cost,
            def_quad,
            0.1 * state.cost,
            key=f"quadratic_quad_{i}_{d}",
            format="%e",
            on_change=reset_optimizer_and_merge,
            args=(state,),
        )
        dist_params.lin_coef = st.number_input(
            "Linear Coefficient",
            -100.0 * state.cost,
            100.0 * state.cost,
            def_lin,
            0.1,
            key=f"quadratic_lin_{i}_{d}",
            format="%e",
            on_change=reset_optimizer_and_merge,
            args=(state,),
        )
    cc1, cc2, cc3 = st.columns(3)  # save these to state to reload
    with cc1:
        dist_params.xmin = st.number_input("xmin", 0.0, max(criteria.xact), 0.0, key=f"quadratic_xmin{i}_{d}")
    with cc2:
        dist_params.xmax = st.number_input(
            "xmax", 0.0, max(criteria.xact), max(criteria.xact), key=f"quadratic_xmax_{i}_{d}"
        )
    with cc3:
        dist_params.linear_offset = st.number_input(
            "offset", -max(criteria.xact), max(criteria.xact), 0.0, key=f"quad_offset_{i}_{d}"
        )

    ythe = get_quadratic_pdf(
        criteria.xthe,
        1.0 / criteria.hr,
        dist_params.quad_coef,
        dist_params.lin_coef,
        dist_params.xmin,
        dist_params.xmax,
        dist_params.linear_offset,
    )
    yact = get_quadratic_pdf(
        criteria.xact,
        1.0 / criteria.hr,
        dist_params.quad_coef,
        dist_params.lin_coef,
        dist_params.xmin,
        dist_params.xmax,
        dist_params.linear_offset,
    )

    return ythe, yact


def assign_linear(
    state: AppState, dist_params: LinearParams, criteria: CriteraParams, i: int, d: int
) -> list[list]:
    def_lin = 1.0
    if f"lin_{i}_{d}" in st.session_state:
        def_lin = st.session_state[f"lin_{i}_{d}"]
    normalize = st.checkbox("Normalize", key="linear_normalize")
    if not normalize:
        dist_params.lin_coef = st.number_input(
            "Linear Coefficient",
            -100.0 * state.cost,
            100.0 * state.cost,
            def_lin,
            0.1,
            key=f"lin_{i}_{d}",
            format="%e",
            on_change=reset_optimizer_and_merge,
            args=(state,),
        )
    cc1, cc2, cc3 = st.columns(3)  # save these to state to reload
    with cc1:
        dist_params.xmin = st.number_input("xmin", 0.0, max(criteria.xact), 0.0, key=f"lin_xmin{i}_{d}")
    with cc2:
        dist_params.xmax = st.number_input(
            "xmax", 0.0, max(criteria.xact), max(criteria.xact), key=f"lin_xmax_{i}_{d}"
        )
    with cc3:
        dist_params.linear_offset = st.number_input(
            "offset", -max(criteria.xact), max(criteria.xact), 0.0, key=f"quad_offset_{i}_{d}"
        )

    ythe = get_linear_pdf(
        criteria.xthe,
        1.0 / criteria.hr,
        dist_params.lin_coef,
        dist_params.xmin,
        dist_params.xmax,
        dist_params.linear_offset,
        normalize,
    )
    yact = get_linear_pdf(
        criteria.xact,
        1.0 / criteria.hr,
        dist_params.lin_coef,
        dist_params.xmin,
        dist_params.xmax,
        dist_params.linear_offset,
        normalize,
    )

    return ythe, yact


def assign_rect(dist_params: RectParams, criteria: CriteraParams, i: int, d: int) -> list[list]:

    cc1, cc2 = st.columns(2)  # save these to state to reload
    with cc1:
        dist_params.xmin = st.number_input("xmin", 0.0, max(criteria.xact), 0.0, key=f"rect_xmin{i}_{d}")
    with cc2:
        dist_params.xmax = st.number_input(
            "xmax", 0.0, max(criteria.xact), max(criteria.xact), key=f"rect_xmax_{i}_{d}"
        )

    yact = get_rect_pdf(criteria.xact, 1.0 / criteria.hr, dist_params.xmin, dist_params.xmax)
    ythe = get_rect_pdf(criteria.xthe, 1.0 / criteria.hr, dist_params.xmin, dist_params.xmax)

    return ythe, yact
