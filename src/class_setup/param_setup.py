import streamlit as st
from src.class_setup.state import AppState, CriteraParams
from src.class_setup.state import Distribution
from src.util.utils import calculate_params


def render_mode_editor(state: AppState):

    mde, cst, step = st.columns(3)
    with mde:
        state.mode = st.text_input("Game Mode", key="gmode", value="base", width=150)
    with cst:
        state.cost = st.number_input("Mode Cost:", value=1.0, width=100)
    with step:
        state.win_step_size = st.number_input("Win Step Size", 0.0, 100.0, 0.1, width=80)

    indir, outdir = st.columns(2)
    with indir:
        state.in_dir = st.text_input("Input Dir:", key="inputDir", value="input_files")
    with outdir:
        state.out_dir = st.text_input("Output Dir:", key="outDir", value="output_files")

    state.lookup_name = st.text_input("Lookup Table", key="lookupName", value=f"lookUpTable_{state.mode}.csv")
    state.segmented_name = st.text_input(
        "Segmented Lookup Table", key="segmentedName", value=f"lookUpTableSegmented_{state.mode}.csv"
    )

    state.lut_file = str.join("/", [state.root_dir, state.in_dir, state.lookup_name])
    state.segmented_file = str.join("/", [state.root_dir, state.in_dir, state.segmented_name])
    st.write(f"Target lookup file: {state.lut_file}")
    st.write((f"Target segmented file: {state.segmented_file}"))


def render_criteria_editor(state: AppState):
    criteria_input = st.text_input(
        "input criteria",
        width=200,
        value="basegame",
        key="criteria_input",
    )

    if st.button("Append", key="append_criteria", width=80):
        name = criteria_input.strip().lower()
        existing_names = {c.name.strip().lower() for c in state.criteria_list}
        if name not in existing_names:
            state.criteria_list.append(CriteraParams(name=name))
        else:
            st.warning("Criteria already exists")


def render_criteria_params(state: AppState):
    for i, criteria in enumerate(state.criteria_list):

        col1, col2, col3 = st.columns(3)

        criteria.rtp = col1.number_input(
            "RTP",
            key=f"rtp_{i}",
            value=criteria.rtp,
            step=0.01,
        )

        criteria.av = col2.number_input(
            "Avg Win",
            key=f"av_{i}",
            value=criteria.av,
            step=0.01,
        )

        criteria.hr = col3.number_input(
            "Hit Rate",
            key=f"hr_{i}",
            value=criteria.hr,
            step=0.01,
        )

        if st.button(
            f"Compute missing value for '{criteria.name}'",
            key=f"compute_{i}",
        ):
            for nme, val in zip(("hr", "av", "rtp"), (criteria.hr, criteria.av, criteria.rtp)):
                if val is None:
                    setattr(criteria, nme, val)

            criteria.rtp, criteria.hr, criteria.av = calculate_params(
                criteria.rtp, criteria.hr, criteria.av, state.cost
            )
            state.dist_objects.append(
                Distribution(criteria=criteria.name, rtp=criteria.rtp, hr=criteria.hr, av_win=criteria.av)
            )

            state.zero_prob -= 1.0 / criteria.hr

            st.success(
                f"Solved missing value for '{criteria.name}'\n RTP:{criteria.rtp}, Av Win: {criteria.av}, hr: {criteria.hr}"
            )
        state.criteria_list[i].plot_log_scale = st.checkbox(
            f"Plot semi log-scale (x) for {state.criteria_list[i].name}", key=f"plot_log_scale_{i}"
        )
