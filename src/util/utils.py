import json
import pickle
from pathlib import Path
from typing import Literal, Type
import streamlit as st
from src.class_setup.state import AppState, ModeData, CriteraParams, DistributionInput
from src.class_setup.models import (
    LogNormalParams,
    GaussianParams,
    ExponentialParams,
    ParabolaParams,
    LinearParams,
    RectParams,
)

DistType = Literal[
    "Log-Normal",
    "Gaussian",
    "Exponential",
    "Quadratic",
    "Linear",
    "Rect",
]

DIST_PARAM_CLASSES: dict[DistType, Type] = {
    "Log-Normal": LogNormalParams,
    "Gaussian": GaussianParams,
    "Exponential": ExponentialParams,
    "Quadratic": ParabolaParams,
    "Linear": LinearParams,
    "Rect": RectParams,
}


def extract_ids(state: AppState, target_string: str):
    ids = []
    pays = []
    zero_ids = []
    total_lookup_length = 0
    with open(state.segmented_file, "r", encoding="utf-8") as f:
        for line in f:
            total_lookup_length += 1
            book, criteria, a, b = line.strip().split(",")
            criteria = str(criteria)
            tot = min(round(float(a) + float(b), 2), state.max_payout)
            if (criteria.lower() == target_string.lower()) or (tot == 0 and not state.mode_contains_zero_criteria):
                ids.append(int(book))
                pays.append(tot)
            elif tot == 0 and state.mode_contains_zero_criteria:
                zero_ids.append(int(book))

    return ids, pays, total_lookup_length, zero_ids


def read_csv(fname: str) -> list:
    payouts = []
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            _, _, payout = line.strip().split(",")
            payouts.append(float(payout))

    return payouts


def get_unique_payouts(payouts: list) -> list:
    unique_payouts = []
    for p in payouts:
        if p not in unique_payouts:
            unique_payouts.append(p)
    return unique_payouts


def get_uniuqe_payouts_from_lut(state: AppState, books: list) -> list:
    payouts = []
    with open(state.lut_file, "r", encoding="utf-8") as f:
        for line in f:
            idx, _, p = line.strip().split(",")
            if int(idx) in books:
                payouts.append(round(float(p) / 100, 2))
    return payouts


def hit_rates_ranges(payouts: list, weights: list) -> dict:
    ranges = [
        (0.0, 0.1),
        (0.1, 1.0),
        (1.0, 2.0),
        (2.0, 5.0),
        (5.0, 10.0),
        (10.0, 20.0),
        (20.0, 50.0),
        (50.0, 100.0),
        (100, 200),
        (200, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 5000),
        (5000, 10000),
        (10000, 25000),
        (25000, 100000),
    ]
    hr_dict = {}
    max_payout = max(payouts)
    for r in ranges:
        if r[0] <= max_payout:
            hr_dict[r] = 0
        else:
            break

    total_weight = sum(weights)

    for p, w in zip(payouts, weights):
        for r in ranges:
            if p >= r[0] and p < r[1]:
                hr_dict[r] += w / total_weight
                break

    for r, prob in hr_dict.items():
        if prob > 0:
            hr_dict[r] = 1 / prob

    return hr_dict


def calculate_params(rtp: float | None, hr: float | None, av_win: float | None, cost: float | None) -> list[float]:
    if [rtp, hr, av_win].count(None) == 1:
        if rtp is None:
            rtp = av_win / (hr * cost)
        elif hr is None:
            hr = av_win / (rtp * cost)
        elif av_win is None:
            av_win = rtp * hr * cost

    return rtp, hr, av_win


def get_optimizer_name(state: AppState) -> list[str]:
    output_dir = Path(state.root_dir) / state.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{state.optimized_prefix}_{state.mode}_"

    indices = []
    for p in output_dir.iterdir():
        if p.is_file() and p.name.startswith(prefix) and p.suffix == ".csv":
            try:
                idx = int(p.stem.replace(prefix, ""))
                indices.append(idx)
            except ValueError:
                pass

    next_idx = max(indices, default=0) + 1

    output_lut = output_dir / f"{prefix}{next_idx}.csv"
    output_hr = output_dir / f"combined_hitrates_{state.mode}_{next_idx}.json"

    return output_lut, output_hr


def write_optimized_lookup(state: AppState) -> None:
    solnpath, hrpath = get_optimizer_name(state)
    with open(solnpath, "w", encoding="utf-8") as f:
        for ele in state.final_optimized_lookup:
            int_weight = int((2**state.weight_scale) * ele[1])
            f.write(f"{ele[0]},{int_weight},{ele[2]}\n")

    with open(hrpath, "w", encoding="utf-8") as f:
        hrwrite = {}
        for r, hr in state.hr_ranges.items():
            hrwrite[str(r)] = hr
        f.write(json.dumps(hrwrite, indent=4))

    # state.merge_solutions = False
    state.write_data = False
    st.write("Write Successful.")


def print_optimized_hr_table(state: AppState) -> None:
    with st.container(border=True):
        print_hr = {}
        for raange, h in state.hr_ranges.items():
            st.write(f"{raange}: {h}")
            print_hr[str(raange)] = round(h, 3)


def save_mode_solution(state: AppState) -> None:
    modeData = ModeData(
        name=state.mode,
        cost=state.cost,
        criteria=state.criteria_list,
        dist=state.dist_objects,
        zero_ids=state.zero_ids,
        zero_prob=state.zero_prob,
        optimizer=state.opt_settings,
        solved_lut=state.final_optimized_lookup,
        solved_hr=state.hr_ranges,
        plot_params=state.plot_params,
    )

    save_path = Path(state.root_dir) / state.data_save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    prefix = f"data_output_{state.mode}_"
    indicies = []
    for p in save_path.iterdir():
        if p.is_file() and p.name.startswith(prefix) and p.suffix == ".pkl":
            idx = p.stem.replace(prefix, "")
            indicies.append(int(idx))
    next_idx = max(indicies, default=0) + 1
    name = save_path / f"{prefix}{next_idx}.pkl"
    with open(name, "wb") as f:
        pickle.dump(modeData, f)

    state.pickle_data = False


def change_dist_params(
    state: AppState, criteria: CriteraParams, dist_type: str = "Log-Normal", dist_num: int = 0
) -> None:
    try:
        param_cls = DIST_PARAM_CLASSES[dist_type]
        attr_name = f"dist{dist_num}_params"
        setattr(criteria, attr_name, param_cls())
    except:
        raise RuntimeError("Function class not reconised")

    state.run_optimizer = False


def load_mode_solution(state: AppState, mode: str, soln: int) -> None:
    file = Path(state.root_dir) / state.data_save_dir / f"data_output_{mode}_{soln}.pkl"
    with open(file, "rb") as f:
        mode_data = pickle.load(f)

    state.mode = mode_data.name
    state.cost = mode_data.cost
    state.criteria_list = mode_data.criteria
    state.dist_objects = mode_data.dist
    state.opt_settings = mode_data.optimizer
    state.final_optimized_lookup = mode_data.solved_lut
    state.hr_ranges = mode_data.solved_hr
    state.zero_ids = mode_data.zero_ids
    state.zero_prob = mode_data.zero_prob
    state.optimization_success = True
    state.set_params = True
    state.plot_params = mode_data.plot_params

    for i, c in enumerate(state.criteria_list):
        st.session_state[f"checkbox_{i}"] = c.is_2_dist
        st.session_state[f"dist1_mix_{i}"] = c.dist1_mix

        for d in range(c.num_dists):
            dist_type = c.dist_type[d]
            params = getattr(c, f"dist{d}_params")
            print(params)
            match dist_type:
                case "Log-Normal":
                    st.session_state[f"log_mode_{i}_{d}"] = params.mode
                    st.session_state[f"log_var_{i}_{d}"] = params.std
                    st.session_state[f"log_mu_{i}_{d}"] = params.scale
                    st.session_state[f"log_offset_{i}_{d}"] = params.linear_offset
                    st.session_state[f"log_xmin{i}_{d}"] = params.xmin
                    st.session_state[f"log_xmax{i}_{d}"] = params.xmax
                case "Gaussian":
                    st.session_state[f"gauss_mode_{i}_{d}"] = params.mean
                    st.session_state[f"gauss_var_{i}_{d}"] = params.std
                    st.session_state[f"gauss_mu_{i}_{d}"] = params.scale
                    st.session_state[f"gauss_offset_{i}_{d}"] = params.linear_offset
                    st.session_state[f"gauss_xmin{i}_{d}"] = params.xmin
                    st.session_state[f"gauss_xmax{i}_{d}"] = params.xmax
                case "Exponential":
                    st.session_state[f"exp_mode_{i}_{d}"] = params.power
                    st.session_state[f"exp_mu_{i}_{d}"] = params.scale
                    st.session_state[f"exp_offset_{i}_{d}"] = params.linear_offset
                    st.session_state[f"exp_xmin{i}_{d}"] = params.xmin
                    st.session_state[f"exp_xmax{i}_{d}"] = params.xmax
                case "Quadratic":
                    st.session_state[f"quadratic_quad_{i}_{d}"] = params.quad_coef
                    st.session_state[f"quadratic_lin_{i}_{d}"] = params.lin_coef
                    st.session_state[f"quadratic_offset_{i}_{d}"] = params.linear_offset
                    st.session_state[f"quadratic_xmin{i}_{d}"] = params.xmin
                    st.session_state[f"quadratic_xmax{i}_{d}"] = params.xmax
                case "Linear":
                    st.session_state[f"lin_{i}_{d}"] = params.lin_coef
                    st.session_state[f"lin_offset_{i}_{d}"] = params.linear_offset
                    st.session_state[f"lin_xmin{i}_{d}"] = params.xmin
                    st.session_state[f"lin_xmax{i}_{d}"] = params.xmax
                case "Rect":
                    st.session_state[f"rect_xmin{i}_{d}"] = params.xmin
                    st.session_state[f"rect_xmax{i}_{d}"] = params.xmax
                    st.session_state[f"rect_scale{i}_{d}"] = params.height
                    st.session_state["rect_norm"] = params.normalized
    st.rerun()
