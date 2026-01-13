import math
import numpy as np


def get_log_normal(payouts, mode=50, mean=300):
    sigma2 = (2.0 / 3.0) * (math.log(mean) - math.log(mode))
    sigma = math.sqrt(sigma2)
    mu = math.log(mode) + sigma2
    x = np.array(payouts)
    pdf = (1.0 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2))

    return pdf


def get_log_normal_pdf(payouts, mode, std, scale, xmin, xmax, offset):
    # need to filter 0 payouts
    x = np.array([p - offset for p in payouts])
    x[x <= 0] = 1e-9
    pdf = (1 / (x * std * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x / mode) - std**2) ** 2) / (2 * std**2))

    mask = (x >= xmin) & (x <= xmax)
    pdf = pdf * mask
    total = pdf.sum()
    pdf /= total
    pdf_norm = []
    for p in pdf:
        pdf_norm.append(p * scale)  # I dont like doing this
    return pdf_norm


def get_gaussian_pdf(payouts, mean, std, scale, xmin, xmax, offset):
    x = np.asarray([p - offset for p in payouts], dtype=float)
    if std <= 0:
        raise ValueError("Standard deviation must be > 0")

    pdf = 1.0 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    mask = (x >= xmin) & (x <= xmax)
    pdf = pdf * mask
    total = pdf.sum()
    pdf /= total

    return (pdf * scale).tolist()


def get_exp_pdf(payouts, power, scale, xmin, xmax, offset):
    x = np.asarray([p - offset for p in payouts], dtype=float)
    pdf = np.exp(-np.power(x, power))

    mask = (x >= xmin) & (x <= xmax)
    pdf *= mask
    total = pdf.sum()
    pdf /= total

    return (pdf * scale).tolist()


def calculate_mu_from_mode(mode, std):
    return math.log(mode) + std**2


def calculate_expectation(payouts, mu, std, xmin, xmax, offset):
    pdf = get_log_normal_pdf(payouts, mu, std, 1.0, xmin, xmax, offset)
    return np.sum(np.array(payouts) * pdf)


def calculate_theoretical_expectation(mu, std):
    return math.exp(mu + std**2)


def calculate_actual_expectation(x, pdf, actual_payouts, cost=1):
    weights = []
    for p in actual_payouts:
        i = np.argmin(np.abs(np.array(x) - p))
        weights.append(pdf[i])

    weights = np.array(weights)
    if weights.sum() == 0:
        return float("nan")

    probs = weights / weights.sum()

    rtp = np.sum(np.array(actual_payouts) * probs) / cost

    return rtp, probs


def calculate_mode(mu, std):
    return math.exp(mu - std**2)


def calculate_act_expectation(actual_payouts, pdf, cost=1):
    weights = pdf
    weights = np.array(weights)
    if weights.sum() == 0:
        return float("nan")

    probs = weights / weights.sum()

    rtp = np.sum(np.array(actual_payouts) * probs) / cost

    return rtp, probs


def get_quadratic_pdf(payouts, scale, quad, lin, xmin, xmax, offset, normalize_curve=False):
    x = np.asarray(payouts, float)

    pdf = (quad * ((x - offset) ** 2)) + (lin * (x - offset))  # + offset

    mask = (x >= xmin) & (x <= xmax)
    pdf *= mask
    if normalize_curve:
        pdf = np.clip(pdf, 1e-12, None)
        pdf /= pdf.sum()
        return (pdf * scale).tolist()
    else:
        return pdf.tolist()


def get_linear_pdf(payouts, scale, lin, xmin, xmax, x_offset, normalize_curve=False):
    x = np.asarray(payouts, float)

    pdf = lin * (x - x_offset)
    pdf = np.where((x >= xmin) & (x <= xmax), pdf, 0.0)

    if normalize_curve:
        pdf = np.clip(pdf, 1e-12, None)
        pdf = pdf * (scale / pdf.sum())

    return pdf.tolist()


def get_rect_pdf(payouts, scale, height, xmin, xmax, normalize_curve=False):
    x = np.asarray(payouts, float)
    pdf = ((x >= xmin) & (x <= xmax)).astype(float)
    pdf *= height
    if normalize_curve:
        pdf = np.clip(pdf, 1e-12, None)
        pdf /= pdf.sum()
        return (scale * pdf).tolist()
    else:
        return pdf.tolist()
