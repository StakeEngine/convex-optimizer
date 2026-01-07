from dataclasses import dataclass, field
from src.class_setup.models import LogNormalParams


class DistributionInput:
    def __init__(self, name, xthe, xact, ythe, yact):
        self.name = name
        self.xthe = xthe
        self.ythe = ythe
        self.xact = xact
        self.yact = yact


class Distribution:
    def __init__(self, criteria, rtp, av_win, hr):
        self.criteria = criteria
        self.rtp = rtp
        self.av_win = av_win
        self.hr = hr
        self.book_ids = []
        self.payouts = []
        self.unique_payouts = []
        self.sub_dist = []

        self.log_mode = None
        self.log_std = None
        self.log_mu = None
        self.log_the_expectation = None

        self.lookup_segment_length = 0
        self.optimized_unique_distribution = {}
        self.unique_payout_counter = {}
        self.optimized_final_distribution = {}


@dataclass
class CriteraParams:
    name: str = ""
    rtp: float | None = None
    hr: float | None = None
    av: float | None = None

    xact: list[float] = field(default_factory=list)
    yact: list[float] = field(default_factory=list)

    xthe: list[float] = field(default_factory=list)
    ythe: list[float] = field(default_factory=list)

    solved_weights: list[float] = field(default_factory=list)
    optimized_unique_distribution: list = field(default_factory=list)
    unique_payout_counter: dict = field(default_factory=dict)
    optimized_final_distribution: dict = field(default_factory=dict)
    solution_metrics: dict = field(default_factory=dict)
    criteria_hr_dict: dict = field(default_factory=dict)

    effective_rtp: float = 0.0
    effective_pdf: list[float] = field(default_factory=list)
    plot_log_scale: bool = False
    dist_type: list[str] = field(default_factory=lambda: ["Log-Normal", "Gaussian", "Exponential"])
    num_dists = 1
    is_2_dist: bool = False
    dist_values: dict = field(default_factory=dict)
    dist1_mix: float = field(default=0.5)
    dist2_mix: float = field(default=0.5)
    dist1_params: LogNormalParams = field(default_factory=LogNormalParams)
    dist2_params: LogNormalParams = field(default_factory=LogNormalParams)
    merged_dist: list = field(default_factory=list)
    merged_dist_the: list = field(default_factory=list)


@dataclass
class ConvexOptSetup:
    kl_divergence: float
    smoothness: float
    payouts: list[float]
    init_weights: list[float]


@dataclass
class PlotSettings:
    criteria_name: str = ""
    xmin: float = 0.0
    xmax: float = 10.0
    show_the_curve: bool = True
    show_solution: bool = False
    normalize_all: bool = False
    base_curves: bool = True
    log_scale: bool = False


@dataclass
class ModeData:
    name: str = field(default_factory=str)
    cost: float = field(default_factory=float)
    criteria: list[CriteraParams] = field(default_factory=list)
    dist: list[Distribution] = field(default_factory=list)
    optimizer: list[ConvexOptSetup] = field(default_factory=list)
    solved_lut: list[tuple] = field(default_factory=list)
    solved_hr: dict = field(default_factory=dict)
    zero_prob: float = field(default_factory=float)
    zero_ids: list[int] = field(default_factory=list)
    dist1_params: LogNormalParams = field(default_factory=LogNormalParams)
    dist2_params: LogNormalParams = field(default_factory=LogNormalParams)
    plot_params: list[PlotSettings] = field(default_factory=list)


@dataclass
class AppState:
    mode: str = ""
    cost: float = 1.0
    criteria_list: list[CriteraParams] = field(default_factory=list)
    dist_objects: list[Distribution] = field(default_factory=list)
    log_normal_params: list[LogNormalParams] = field(default_factory=list)
    plot_params: list[PlotSettings] = field(default_factory=list)
    zero_ids: list[int] = field(default_factory=list)
    mode_contains_zero_criteria: bool = True
    all_payout_ints: list[int] = field(default_factory=list)
    zero_prob: float = 1.0
    lookup_length: int = 0
    book_offset = 1
    max_payout: float = 0.0
    weight_scale: int = field(default=50)

    set_params: bool = False
    merge_solutions: bool = False
    compute_weights: bool = False
    show_plot_settings: bool = False
    show_optimizer_settings: bool = False
    run_optimizer: bool = False
    combine_optimized_solutions: bool = False
    optimization_success: bool = False
    write_data: bool = False
    pickle_data: bool = False

    lookup_name: str = ""
    lut_file: str = ""
    segmented_name: str = ""
    segmented_file: str = ""
    opt_settings: list[ConvexOptSetup] = field(default_factory=list)
    lut_read_complete: bool = field(default=False)

    # to do - assign these in gui
    win_step_size: float = field(default=0.1)
    excluded_payouts = [0]
    disused_sims: list[int] = field(default_factory=list)
    disused_int_payouts: list[int] = field(default_factory=list)
    root_dir: str = "src"
    out_dir: str = "output_files"
    in_dir: str = "input_files"
    final_optimized_lookup: list = field(default_factory=list)
    optimized_lookup_name = "final_optimized_lookup.csv"
    optimized_prefix: str = "optimized"
    final_total_payouts: list = field(default_factory=list)
    final_weights: list = field(default_factory=list)
    hr_ranges: dict = field(default_factory=dict)

    # pickled data
    data_save_dir = "saved_solns"
