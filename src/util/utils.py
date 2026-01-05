import json
import pickle
from pathlib import Path
import streamlit as st
from src.class_setup.state import AppState, ModeData, CriteraParams
from src.class_setup.models import LogNormalParams, GaussianParams, ExponentialParams


def extract_ids(state: AppState, target_string, exclustion_payouts):
    ids = []
    pays = []
    exc_payouts = []
    zero_ids = []
    total_lookup_length = 0
    for p in exclustion_payouts:
        exc_payouts.append(int(p) * 100)
    with open(state.segmented_file, "r", encoding="utf-8") as f:
        for line in f:
            total_lookup_length += 1
            book, criteria, a, b = line.strip().split(",")
            tot = round(float(a) + float(b), 2)
            if tot == 0:
                zero_ids.append(int(book))
            if criteria.lower() == target_string.lower() and (tot not in exc_payouts):
                ids.append(int(book))
                pays.append(tot)
    return ids, pays, total_lookup_length, zero_ids


def read_csv(fname):
    payouts = []
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            _, _, payout = line.strip().split(",")
            payouts.append(float(payout))

    return payouts


def get_unique_payouts(payouts):
    unique_payouts = []
    for p in payouts:
        if p not in unique_payouts:
            unique_payouts.append(p)
    return unique_payouts


def get_uniuqe_payouts_from_lut(state: AppState, books):
    payouts = []
    with open(state.lut_file, "r", encoding="utf-8") as f:
        for line in f:
            idx, _, p = line.strip().split(",")
            if int(idx) in books:
                payouts.append(round(float(p) / 100, 2))
    return payouts


def hit_rates_ranges(payouts, weights):
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


def calculate_params(rtp, hr, av_win, cost):
    lst = [rtp, hr, av_win]
    assert sum([1 for x in lst if x is None]) == 1

    if rtp is None:
        rtp = (cost * av_win) / hr
    elif hr is None:
        hr = (av_win * cost) / rtp
    elif av_win is None:
        av_win = (rtp * hr) / cost

    return rtp, hr, av_win


def get_optimizer_name(state: AppState):
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


def write_optimized_lookup(state: AppState):
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


def print_optimized_hr_table(state: AppState):
    with st.container(border=True):
        print_hr = {}
        for raange, h in state.hr_ranges.items():
            st.write(f"{raange}: {h}")
            print_hr[str(raange)] = round(h, 4)


def save_mode_solution(state: AppState):
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


def change_dist_params(state: AppState, criteria: CriteraParams, dist_type: str = "Log-Normal", dist_num: int = 0):
    match dist_type:
        case "Log-Normal":
            if dist_num == 0:
                criteria.dist1_params = LogNormalParams()
            elif dist_num == 1:
                criteria.dist2_params = LogNormalParams()
        case "Gaussian":
            if dist_num == 0:
                criteria.dist1_params = GaussianParams()
            elif dist_num == 1:
                criteria.dist2_params = GaussianParams()
        case "Exponential":
            if dist_num == 0:
                criteria.dist1_params = ExponentialParams()
            elif dist_num == 1:
                criteria.dist2_params = ExponentialParams()
    state.run_optimizer = False


def load_mode_solution(state: AppState, mode: str, soln: int):
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
            params = c.dist1_params if d == 0 else c.dist2_params
            st.session_state[f"dist_type_{i}_{d}"] = dist_type
            if dist_type == "Log-Normal":
                st.session_state[f"log_mode_{i}_{d}"] = params.mode
                st.session_state[f"log_var_{i}_{d}"] = params.std
                st.session_state[f"log_mu_{i}_{d}"] = params.scale

            elif dist_type == "Gaussian":
                st.session_state[f"gauss_mode_{i}_{d}"] = params.mean
                st.session_state[f"gauss_var_{i}_{d}"] = params.std
                st.session_state[f"gauss_mu_{i}_{d}"] = params.scale

            elif dist_type == "Exponential":
                st.session_state[f"exp_mode_{i}_{d}"] = params.power
                st.session_state[f"exp_mu_{i}_{d}"] = params.scale
            else:
                raise RuntimeError

    st.rerun()
