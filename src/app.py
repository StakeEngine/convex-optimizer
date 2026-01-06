import streamlit as st
from src.class_setup.state import AppState
from src.class_setup.param_setup import render_mode_editor, render_criteria_editor, render_criteria_params
from src.computation.compute_params import render_compute_params, render_target_dist_params
from src.computation.optimizer import (
    render_optimizer_settings,
    solve_optimizer,
    merge_solutions,
    render_optimizer_results,
)
from src.util.utils import write_optimized_lookup, print_optimized_hr_table, save_mode_solution, load_mode_solution
from src.data.plotting import render_plots


def get_state() -> AppState:
    if "app" not in st.session_state:
        st.session_state.app = AppState()

    return st.session_state.app


state = get_state()


with st.expander("(optional) Load Existing Solution"):
    mde = st.text_input("Mode Name", "")
    soln = int(st.number_input("Mode Solution Value", 1))
    loader = st.button("Load Pickled Data")
    if loader:
        load_mode_solution(state, mde, soln)

with st.container(border=True):
    st.subheader("Game Setup")
    render_mode_editor(state)

with st.container(border=True):
    st.subheader("Define optimization criteria")
    render_criteria_editor(state)
    render_criteria_params(state)


st.subheader("Compute Parameters")
if st.button("Set Criteria Params"):
    state.set_params = True

if state.set_params:
    with st.container(border=True):
        render_compute_params(state)

    with st.container(border=True):
        render_target_dist_params(state)

        plt_container = st.container()
        render_plots(state, plt_container)

    with st.sidebar:
        st.subheader("Solution preferences")
        render_optimizer_settings(state)

        if st.button("Run Convex Optimizer"):
            state.run_optimizer = True

if state.run_optimizer and not state.optimization_success:
    solve_optimizer(state)
    state.run_optimizer = False
    st.rerun()

if state.optimization_success:
    render_optimizer_results(state)
    if st.button("Merge Optimized Solutions"):
        state.merge_solutions = True

if state.merge_solutions:
    merge_solutions(state)
    write_data = st.button("Write Lookup and HR Data")
    if write_data:
        state.write_data = True

    write_raw_data = st.button("Save Complete Class Data")
    if write_raw_data:
        state.pickle_data = True

if len(state.hr_ranges) > 0:
    with st.expander("Final Hit-Rate Table"):
        print_optimized_hr_table(state)

if state.write_data:
    write_optimized_lookup(state)

if state.pickle_data:
    save_mode_solution(state)
