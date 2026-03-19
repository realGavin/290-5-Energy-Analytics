import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


WORKBOOK_PATH = Path("/Users/gavinzeng/Desktop/Prices.xlsx")
OUTPUT_DIR = Path("/Users/gavinzeng/Desktop")


CONFIGS = {
    1: {
        "min_mw": 0.0,
        "max_mw": 0.0,
        "su_dollar": 0.0,
        "su_fuel": 0.0,
        "ml_dollar": 0.0,
        "ml_fuel": 0.0,
        "vom": 0.0,
        "segments": [],
        "ihr": [],
    },
    2: {
        "min_mw": 57.0,
        "max_mw": 190.0,
        "su_dollar": 7250.0,
        "su_fuel": 220.0,
        "ml_dollar": 285.0,
        "ml_fuel": 717.687,
        "vom": 5.0,
        "segments": [57.0, 114.0, 152.0, 190.0],
        "ihr": [10.642, 10.275, 10.133],
    },
    3: {
        "min_mw": 114.0,
        "max_mw": 380.0,
        "su_dollar": 7250.0,
        "su_fuel": 220.0,
        "ml_dollar": 570.0,
        "ml_fuel": 1435.374,
        "vom": 5.0,
        "segments": [114.0, 228.0, 304.0, 380.0],
        "ihr": [10.642, 10.275, 10.133],
    },
    4: {
        "min_mw": 150.0,
        "max_mw": 340.0,
        "su_dollar": 16500.0,
        "su_fuel": 850.0,
        "ml_dollar": 375.0,
        "ml_fuel": 1154.25,
        "vom": 2.5,
        "segments": [150.0, 204.0, 272.0, 340.0],
        "ihr": [7.358, 7.149, 7.048],
    },
    5: {
        "min_mw": 312.0,
        "max_mw": 610.0,
        "su_dollar": 23750.0,
        "su_fuel": 1070.0,
        "ml_dollar": 624.0,
        "ml_fuel": 2221.752,
        "vom": 2.0,
        "segments": [312.0, 366.0, 488.0, 610.0],
        "ihr": [6.999, 6.839, 6.762],
    },
}

TARGET_START = pd.Timestamp("2022-03-21")
TARGET_END = pd.Timestamp("2022-03-27")


def normalize_name(name):
    text = str(name).strip().upper()
    for char in [" ", "-", "/", "(", ")", "$", ".", ":", "\n"]:
        text = text.replace(char, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def find_sheet_name(excel_file, include_words):
    normalized = {sheet: normalize_name(sheet) for sheet in excel_file.sheet_names}
    for sheet, norm in normalized.items():
        if all(word in norm for word in include_words):
            return sheet
    for sheet, norm in normalized.items():
        if any(word in norm for word in include_words):
            return sheet
    raise ValueError(f"Could not find a sheet matching words {include_words}. Available sheets: {excel_file.sheet_names}")


def find_column(columns, include_words, exclude_words=None):
    exclude_words = exclude_words or []
    normalized = {col: normalize_name(col) for col in columns}
    for col, norm in normalized.items():
        if all(word in norm for word in include_words) and not any(word in norm for word in exclude_words):
            return col
    for col, norm in normalized.items():
        if any(word in norm for word in include_words) and not any(word in norm for word in exclude_words):
            return col
    raise ValueError(f"Could not find a column matching words {include_words} in columns {list(columns)}")


def find_value_column(df, preferred_words, exclude_words=None):
    exclude_words = exclude_words or []
    normalized = {col: normalize_name(col) for col in df.columns}
    for col, norm in normalized.items():
        if any(word in norm for word in preferred_words) and not any(word in norm for word in exclude_words):
            return col

    numeric_candidates = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0 and col not in exclude_words:
            numeric_candidates.append(col)

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    if numeric_candidates:
        return numeric_candidates[-1]
    raise ValueError(f"Could not identify a numeric value column from columns {list(df.columns)}")


def parse_operating_date(series):
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(series.astype(str).str.strip(), errors="coerce")
    if parsed.isna().any():
        raise ValueError("Could not parse OPERATING_DATE values.")
    return parsed.dt.normalize()


def to_hour_ending(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        hour = int(digits)
        if 1 <= hour <= 24:
            return hour
    return np.nan


def reshape_hourly_if_needed(df):
    cols_norm = {col: normalize_name(col) for col in df.columns}
    has_hour_col = any("HOUR" in norm for norm in cols_norm.values())
    if has_hour_col:
        return df.copy()

    date_col = find_column(df.columns, ["DATE"])
    value_vars = []
    for col, norm in cols_norm.items():
        if norm in {"HE", "HOUR", "HOUR_ENDING"}:
            value_vars.append(col)
            continue
        if norm.startswith("HE_"):
            value_vars.append(col)
            continue
        if norm.startswith("HOUR_"):
            value_vars.append(col)
            continue
        if norm.isdigit() and 1 <= int(norm) <= 24:
            value_vars.append(col)
            continue
    if not value_vars:
        raise ValueError("Could not identify hourly columns to reshape from wide to long format.")

    long_df = df.melt(id_vars=[date_col], value_vars=value_vars, var_name="HOUR_LABEL", value_name="PRICE_ELECTRIC")
    long_df["HOUR_ENDING"] = long_df["HOUR_LABEL"].map(to_hour_ending)
    long_df = long_df.drop(columns=["HOUR_LABEL"])
    long_df = long_df.rename(columns={date_col: "OPERATING_DATE"})
    return long_df


def load_working_dataframe():
    excel_file = pd.ExcelFile(WORKBOOK_PATH, engine="openpyxl")
    print("Workbook sheet names:", excel_file.sheet_names)

    electric_sheet = find_sheet_name(excel_file, ["PRICE", "ELECTRIC"])
    gas_sheet = find_sheet_name(excel_file, ["PRICE", "GAS"])

    power_raw = pd.read_excel(excel_file, sheet_name=electric_sheet)
    gas_raw = pd.read_excel(excel_file, sheet_name=gas_sheet)

    power_raw = reshape_hourly_if_needed(power_raw)

    power_date_col = find_column(power_raw.columns, ["DATE"])
    power_hour_col = find_column(power_raw.columns, ["HOUR"])
    power_price_col = find_value_column(power_raw, ["PRICE", "LMP", "DALMP", "NP15", "SP15", "ZP26"], exclude_words=["DATE", "HOUR", "GAS", "CO2"])

    gas_date_col = find_column(gas_raw.columns, ["DATE"])
    gas_price_col = find_value_column(gas_raw, ["GAS", "CITYGATE", "PRICE"], exclude_words=["DATE", "HOUR", "CO2"])

    print("Columns used:")
    print({
        "electric_sheet": electric_sheet,
        "power_date_col": power_date_col,
        "power_hour_col": power_hour_col,
        "power_price_col": power_price_col,
        "gas_sheet": gas_sheet,
        "gas_date_col": gas_date_col,
        "gas_price_col": gas_price_col,
    })

    power_df = power_raw[[power_date_col, power_hour_col, power_price_col]].copy()
    power_df.columns = ["OPERATING_DATE", "HOUR_ENDING", "PRICE_ELECTRIC"]
    power_df["OPERATING_DATE"] = parse_operating_date(power_df["OPERATING_DATE"])
    power_df["HOUR_ENDING"] = power_df["HOUR_ENDING"].map(to_hour_ending)
    power_df["PRICE_ELECTRIC"] = pd.to_numeric(power_df["PRICE_ELECTRIC"], errors="coerce")

    gas_df = gas_raw[[gas_date_col, gas_price_col]].copy()
    gas_df.columns = ["OPERATING_DATE", "PRICE_GAS"]
    gas_df["OPERATING_DATE"] = parse_operating_date(gas_df["OPERATING_DATE"])
    gas_df["PRICE_GAS"] = pd.to_numeric(gas_df["PRICE_GAS"], errors="coerce")

    working = power_df.merge(gas_df, on="OPERATING_DATE", how="left")
    working = working.dropna(subset=["OPERATING_DATE", "HOUR_ENDING", "PRICE_ELECTRIC", "PRICE_GAS"])
    working = working[(working["OPERATING_DATE"] >= TARGET_START) & (working["OPERATING_DATE"] <= TARGET_END)].copy()
    working["HOUR_ENDING"] = working["HOUR_ENDING"].astype(int)
    working = working.sort_values(["OPERATING_DATE", "HOUR_ENDING"]).reset_index(drop=True)

    if len(working) != 168:
        raise ValueError(f"Expected 168 hourly rows for March 21-27, 2022, found {len(working)} rows.")

    print("\nFirst few rows of the working dataframe:")
    print(working.head())
    return working


def build_variable_index(num_periods):
    index = {}
    names = []
    cursor = 0

    def add(name):
        nonlocal cursor
        index[name] = cursor
        names.append(name)
        cursor += 1

    for t in range(num_periods):
        for c in CONFIGS:
            add(("y", c, t))
    for t in range(num_periods):
        for c in CONFIGS:
            add(("s", c, t))
    for t in range(num_periods):
        for c in CONFIGS:
            if c == 1:
                continue
            for k in range(3):
                add(("x", c, k, t))
    for t in range(num_periods):
        add(("p", t))
    return index, names, cursor


def add_sparse_row(rows, lows, highs, entries, low, high):
    row = {}
    for idx, coeff in entries:
        row[idx] = row.get(idx, 0.0) + coeff
    rows.append(row)
    lows.append(low)
    highs.append(high)


def initial_state(config_id):
    return 1.0 if config_id == 1 else 0.0


def build_model_inputs(df):
    num_periods = len(df)
    index, _, num_vars = build_variable_index(num_periods)

    c = np.zeros(num_vars)
    lb = np.zeros(num_vars)
    ub = np.full(num_vars, np.inf)
    integrality = np.zeros(num_vars, dtype=int)

    price_e = df["PRICE_ELECTRIC"].to_numpy()
    price_g = df["PRICE_GAS"].to_numpy()

    for t in range(num_periods):
        c[index[("p", t)]] = -price_e[t]

        for config_id, config in CONFIGS.items():
            y_idx = index[("y", config_id, t)]
            s_idx = index[("s", config_id, t)]
            integrality[y_idx] = 1
            integrality[s_idx] = 1
            ub[y_idx] = 1.0
            ub[s_idx] = 1.0

            startup_cost = config["su_dollar"] + price_g[t] * config["su_fuel"]
            min_load_cost = config["ml_dollar"] + price_g[t] * config["ml_fuel"]

            c[s_idx] += startup_cost
            if config_id != 1:
                c[y_idx] += min_load_cost

                segment_points = config["segments"]
                segment_lengths = np.diff(segment_points)
                for k, (seg_len, ihr) in enumerate(zip(segment_lengths, config["ihr"])):
                    x_idx = index[("x", config_id, k, t)]
                    ub[x_idx] = seg_len
                    incremental_cost = ihr * price_g[t] + config["vom"]
                    c[x_idx] += incremental_cost

    rows = []
    lows = []
    highs = []

    for t in range(num_periods):
        add_sparse_row(
            rows,
            lows,
            highs,
            [(index[("y", c_id, t)], 1.0) for c_id in CONFIGS],
            1.0,
            1.0,
        )

    for t in range(num_periods):
        entries = [(index[("p", t)], 1.0)]
        for config_id, config in CONFIGS.items():
            if config_id == 1:
                continue
            entries.append((index[("y", config_id, t)], -config["min_mw"]))
            for k in range(3):
                entries.append((index[("x", config_id, k, t)], -1.0))
        add_sparse_row(rows, lows, highs, entries, 0.0, 0.0)

    for t in range(num_periods):
        for config_id, config in CONFIGS.items():
            if config_id == 1:
                continue
            segment_lengths = np.diff(config["segments"])
            for k, seg_len in enumerate(segment_lengths):
                add_sparse_row(
                    rows,
                    lows,
                    highs,
                    [
                        (index[("x", config_id, k, t)], 1.0),
                        (index[("y", config_id, t)], -seg_len),
                    ],
                    -np.inf,
                    0.0,
                )

    # Startup logic is modeled tightly so s[c,t] is 1 only for a real transition
    # into configuration c. The pre-horizon state is fixed as OFF (config 1).
    for t in range(num_periods):
        for config_id in CONFIGS:
            prev_y = initial_state(config_id) if t == 0 else None
            prev_y_idx = None if t == 0 else index[("y", config_id, t - 1)]

            entries = [(index[("s", config_id, t)], -1.0), (index[("y", config_id, t)], 1.0)]
            if t == 0:
                add_sparse_row(rows, lows, highs, entries, -np.inf, prev_y)
            else:
                entries.append((prev_y_idx, -1.0))
                add_sparse_row(rows, lows, highs, entries, -np.inf, 0.0)

            add_sparse_row(
                rows,
                lows,
                highs,
                [
                    (index[("s", config_id, t)], 1.0),
                    (index[("y", config_id, t)], -1.0),
                ],
                -np.inf,
                0.0,
            )

            if t == 0:
                add_sparse_row(
                    rows,
                    lows,
                    highs,
                    [(index[("s", config_id, t)], 1.0)],
                    -np.inf,
                    1.0 - initial_state(config_id),
                )
            else:
                add_sparse_row(
                    rows,
                    lows,
                    highs,
                    [
                        (index[("s", config_id, t)], 1.0),
                        (index[("y", config_id, t - 1)], 1.0),
                    ],
                    -np.inf,
                    1.0,
                )

    # Hour 1 starts from a fixed OFF pre-horizon state, so OFF -> 3 and OFF -> 5
    # are forbidden immediately at the start of the horizon.
    for forbidden_config in [3, 5]:
        add_sparse_row(
            rows,
            lows,
            highs,
            [(index[("y", forbidden_config, 0)], 1.0)],
            -np.inf,
            0.0,
        )

    # Keep the required simplified transition restrictions for all hourly transitions.
    forbidden_pairs = [(1, 3), (1, 5), (2, 5)]
    for t in range(1, num_periods):
        for prev_config, next_config in forbidden_pairs:
            add_sparse_row(
                rows,
                lows,
                highs,
                [
                    (index[("y", prev_config, t - 1)], 1.0),
                    (index[("y", next_config, t)], 1.0),
                ],
                -np.inf,
                1.0,
            )

    row_idx = []
    col_idx = []
    data = []
    for i, row in enumerate(rows):
        for j, value in row.items():
            row_idx.append(i)
            col_idx.append(j)
            data.append(value)

    constraint_matrix = np.zeros((len(rows), num_vars))
    constraint_matrix[row_idx, col_idx] = data

    constraints = LinearConstraint(constraint_matrix, np.array(lows), np.array(highs))
    bounds = Bounds(lb, ub)

    return c, integrality, bounds, constraints, index


def solve_dispatch(df):
    c, integrality, bounds, constraints, index = build_model_inputs(df)
    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    return result, index


def get_active_config(solution, index, t):
    return max(CONFIGS, key=lambda c_id: solution[index[("y", c_id, t)]])


def build_validation(active_configs):
    first_12 = active_configs[:12]
    hour_1_ok = first_12[0] not in {3, 5}
    no_25 = all(not (active_configs[t - 1] == 2 and active_configs[t] == 5) for t in range(1, len(active_configs)))
    no_off_3 = all(not (active_configs[t - 1] == 1 and active_configs[t] == 3) for t in range(1, len(active_configs)))
    no_off_5 = all(not (active_configs[t - 1] == 1 and active_configs[t] == 5) for t in range(1, len(active_configs)))
    return {
        "first_12_configs": first_12,
        "hour_1_not_3_or_5": hour_1_ok,
        "no_2_to_5": no_25,
        "no_off_to_3": no_off_3,
        "no_off_to_5": no_off_5,
    }


def clean_mw(value, tol=1e-6):
    return 0.0 if abs(value) <= tol else float(value)


def extract_results(df, result, index):
    if result.x is None:
        raise RuntimeError("Solver did not return a solution vector.")

    solution = result.x
    records = []
    hourly_revenue = 0.0
    total_startup_cost = 0.0
    total_min_load_cost = 0.0
    total_incremental_cost = 0.0
    startup_fuel_cost = 0.0
    min_load_fuel_cost = 0.0
    incremental_fuel_cost = 0.0
    total_config_starts = 0
    off_to_online_starts = 0
    active_configs = [get_active_config(solution, index, t) for t in range(len(df))]

    for t, row in df.iterrows():
        active_config = active_configs[t]
        generation = clean_mw(solution[index[("p", t)]])
        price_g = row["PRICE_GAS"]
        price_e = row["PRICE_ELECTRIC"]

        hourly_revenue += price_e * generation

        for config_id, config in CONFIGS.items():
            s_val = solution[index[("s", config_id, t)]]
            y_val = solution[index[("y", config_id, t)]]
            if config_id != 1 and s_val > 0.5:
                total_config_starts += 1
                startup_cost = config["su_dollar"] + price_g * config["su_fuel"]
                total_startup_cost += startup_cost
                startup_fuel_cost += price_g * config["su_fuel"]

            if y_val > 0.5 and config_id != 1:
                min_load_cost = config["ml_dollar"] + price_g * config["ml_fuel"]
                total_min_load_cost += min_load_cost
                min_load_fuel_cost += price_g * config["ml_fuel"]
                for k, ihr in enumerate(config["ihr"]):
                    x_val = solution[index[("x", config_id, k, t)]]
                    total_incremental_cost += x_val * (ihr * price_g + config["vom"])
                    incremental_fuel_cost += x_val * (ihr * price_g)

        prev_config = 1 if t == 0 else active_configs[t - 1]
        if prev_config == 1 and active_config != 1:
            off_to_online_starts += 1

        records.append(
            {
                "OPERATING_DATE": row["OPERATING_DATE"].date().isoformat(),
                "HOUR_ENDING": int(row["HOUR_ENDING"]),
                "PRICE_ELECTRIC": float(price_e),
                "PRICE_GAS": float(price_g),
                "CONFIGURATION_ACTIVE": int(active_config),
                "MW_GENERATION": generation,
            }
        )

    result_df = pd.DataFrame(records)
    total_costs = total_startup_cost + total_min_load_cost + total_incremental_cost
    fuel_costs = startup_fuel_cost + min_load_fuel_cost + incremental_fuel_cost
    total_mwh = result_df["MW_GENERATION"].sum()

    summary = {
        "objective_value": hourly_revenue - total_costs,
        "gross_margin_per_kw": (hourly_revenue - total_costs) / 610000.0,
        "capacity_factor_pct": total_mwh / (610.0 * 168.0) * 100.0,
        "total_revenue": hourly_revenue,
        "total_costs": total_costs,
        "fuel_costs": fuel_costs,
        "number_of_starts_off_to_online": off_to_online_starts,
        "number_of_configuration_starts": total_config_starts,
    }

    validation = build_validation(active_configs)
    return result_df, summary, validation


def create_plots(result_df):
    hours = np.arange(1, len(result_df) + 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hours, result_df["MW_GENERATION"], color="tab:blue", linewidth=1.8)
    ax.set_title("CCGT MW Generation: March 21-27, 2022")
    ax.set_xlabel("Hour Index")
    ax.set_ylabel("MW")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot_mw_generation.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(hours, result_df["CONFIGURATION_ACTIVE"], where="mid", color="tab:orange", linewidth=1.8)
    ax.set_title("CCGT Active Configuration: March 21-27, 2022")
    ax.set_xlabel("Hour Index")
    ax.set_ylabel("Configuration")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot_configuration.png", dpi=200)
    plt.close(fig)


def write_summary(summary, validation, solver_status):
    lines = [
        "Homework Task 2c(ii)-2c(vi) Summary",
        "",
        f"Solver status: {solver_status}",
        f"Objective value ($): {summary['objective_value']:.2f}",
        f"Gross Margin ($/kW): {summary['gross_margin_per_kw']:.6f}",
        f"Capacity Factor (%): {summary['capacity_factor_pct']:.2f}",
        f"Total Revenue ($): {summary['total_revenue']:.2f}",
        f"Total Costs ($): {summary['total_costs']:.2f}",
        f"Fuel Costs ($): {summary['fuel_costs']:.2f}",
        f"OFF-to-online starts: {summary['number_of_starts_off_to_online']}",
        f"Total configuration startups: {summary['number_of_configuration_starts']}",
        "",
        "Simplifying assumptions used:",
        "- Used SciPy MILP solver because pulp was not available locally and package install was blocked by no network access.",
        "- Corrected initial condition: pre-horizon state fixed at config 1 (OFF), with configs 2-5 inactive before hour 1.",
        "- Forbidden transitions enforced: OFF->3, OFF->5, and 2->5.",
        "- Ramping constraints were omitted.",
        "- Minimum up/down time constraints were omitted.",
        "- CO2 cost was ignored.",
        "- Startup variables were tightened with lower and upper bounds so they only indicate real entries into configurations.",
        "",
        "Validation checks:",
        f"- First 12 hourly configurations: {validation['first_12_configs']}",
        f"- Hour 1 is not config 3 or 5: {validation['hour_1_not_3_or_5']}",
        f"- No 2->5 transitions: {validation['no_2_to_5']}",
        f"- No OFF->3 transitions: {validation['no_off_to_3']}",
        f"- No OFF->5 transitions: {validation['no_off_to_5']}",
    ]
    (OUTPUT_DIR / "summary_2c.txt").write_text("\n".join(lines))


def main():
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK_PATH}")

    df = load_working_dataframe()
    result, index = solve_dispatch(df)

    status_map = {
        0: "Optimization terminated successfully",
        1: "Iteration or time limit reached",
        2: "Problem appears infeasible",
        3: "Problem appears unbounded",
        4: "Other solver issue",
    }
    solver_status = status_map.get(result.status, str(result.status))
    print("\nSolver status:", solver_status)

    if result.status != 0:
        raise RuntimeError(f"Optimization failed with status {solver_status}. Message: {result.message}")

    result_df, summary, validation = extract_results(df, result, index)
    result_df.to_csv(OUTPUT_DIR / "CCGT_CAISO.csv", index=False)
    create_plots(result_df)
    write_summary(summary, validation, solver_status)

    print("\nFirst 12 rows of the output dataframe:")
    print(result_df.head(12))

    print("\nValidation:")
    print("first_12_configs:", validation["first_12_configs"])
    print("hour_1_not_3_or_5:", validation["hour_1_not_3_or_5"])
    print("no_2_to_5:", validation["no_2_to_5"])
    print("no_off_to_3:", validation["no_off_to_3"])
    print("no_off_to_5:", validation["no_off_to_5"])

    print("\nSummary metrics:")
    for key, value in summary.items():
        if isinstance(value, (int, np.integer)):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")

    print("\nRegenerated output files:")
    print(OUTPUT_DIR / "solve_2c.py")
    print(OUTPUT_DIR / "CCGT_CAISO.csv")
    print(OUTPUT_DIR / "plot_mw_generation.png")
    print(OUTPUT_DIR / "plot_configuration.png")
    print(OUTPUT_DIR / "summary_2c.txt")


if __name__ == "__main__":
    main()
