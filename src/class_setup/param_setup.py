import streamlit as st
from src.class_setup.state import AppState, CriteraParams
from src.class_setup.state import Distribution
from src.util.utils import calculate_params


def render_mode_editor(state: AppState):

    mde, cst, step, zero_criteria = st.columns(4)
    with mde:
        state.mode = st.text_input("Game Mode", key="gmode", value="base", width=150)
    with cst:
        state.cost = st.number_input("Mode Cost:", value=1.0, width=100)
    with step:
        state.win_step_size = st.number_input("Win Step Size", 0.0, 100.0, 0.1, width=80)
    with zero_criteria:
        state.mode_contains_zero_criteria = st.checkbox("'0' Criteria?", True, "contains_zero_criteria")

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
    criteria_input = str(
        st.text_input(
            "input criteria",
            width=200,
            value="basegame",
            key="criteria_input",
        )
    )

    if st.button("Append", key="append_criteria", width=80):
        name = criteria_input.strip().lower()
        existing_names = {c.name.strip().lower() for c in state.criteria_list}
        if name not in existing_names:
            state.criteria_list.append(CriteraParams(name=name))
        else:
            st.warning("Criteria already exists")


def ensure_criteria_state(criteria, i):
    for name in ["rtp", "av", "hr"]:
        key = f"{name}_{i}"
        if key not in st.session_state:
            st.session_state[key] = getattr(criteria, name)


def compute_missing(i: int, state: AppState, rtp, hr, av):
    criteria = state.criteria_list[i]
    rtp, hr, av = calculate_params(
        rtp,
        hr,
        av,
        state.cost,
    )

    st.session_state[f"rtp_{i}"] = rtp
    st.session_state[f"hr_{i}"] = hr
    st.session_state[f"av_{i}"] = av

    criteria.rtp = rtp
    criteria.hr = hr
    criteria.av = av

    st.session_state[f"computed_{i}"] = True


def render_criteria_params(state: AppState):

    for i, criteria in enumerate(state.criteria_list):
        ensure_criteria_state(criteria, i)

        col1, col2, col3 = st.columns(3)

        col1.number_input(
            "RTP",
            key=f"rtp_{i}",
            value=st.session_state.get(f"rtp_{i}", criteria.rtp),
            step=0.001,
        )

        col2.number_input(
            "Avg Win",
            key=f"av_{i}",
            value=st.session_state.get(f"av_{i}", criteria.av),
            step=0.01,
        )

        col3.number_input(
            "Hit Rate",
            key=f"hr_{i}",
            value=st.session_state.get(f"hr_{i}", criteria.hr),
            step=0.01,
        )

        if st.button(
            f"Compute missing value for '{criteria.name}'",
            key=f"compute_{i}",
            on_click=compute_missing,
            args=(
                i,
                state,
                st.session_state.get(f"rtp_{i}", criteria.rtp),
                st.session_state.get(f"hr_{i}", criteria.hr),
                st.session_state.get(f"av_{i}", criteria.av),
            ),
        ):

            state.dist_objects.append(
                Distribution(criteria=criteria.name, rtp=criteria.rtp, hr=criteria.hr, av_win=criteria.av)
            )

            st.success(
                f"Solved missing value for '{criteria.name}'\n RTP:{criteria.rtp}, Av Win: {criteria.av}, hr: {criteria.hr}",
            )
