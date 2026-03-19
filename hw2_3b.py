from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


WORKBOOK_CANDIDATES = [
    Path("/Users/gavinzeng/Desktop/UCB/Class/290 Energy Analytics/Homework 2/Prices.xlsx"),
    Path("/Users/gavinzeng/Desktop/Prices.xlsx"),
]
OUTPUT_DIR = Path("/Users/gavinzeng/Desktop/UCB/Class/290 Energy Analytics/Homework 2")

TARGET_START = pd.Timestamp("2022-03-21")
TARGET_END = pd.Timestamp("2022-03-27")

UNITS = {
    "Unit1": {
        "capacity_mw": 340.0,
        "min_mw": 150.0,
        "su_dollar": 16500.0,
        "su_fuel": 850.0,
        "ml_dollar": 375.0,
        "ml_fuel": 1154.25,
        "vom": 2.50,
        "endpoints": [150.0, 204.0, 272.0, 340.0],
        "ihr": [6.421, 6.525, 6.640],
        "gross_margin_kw_denom": 340000.0,
        "capacity_factor_mwh_denom": 340.0 * 168.0,
    },
    "Unit2": {
        "capacity_mw": 270.0,
        "min_mw": 162.0,
        "su_dollar": 7250.0,
        "su_fuel": 220.0,
        "ml_dollar": 249.0,
        "ml_fuel": 1067.502,
        "vom": 2.00,
        "endpoints": [162.0, 216.0, 243.0, 270.0],
        "ihr": [6.292, 6.361, 6.452],
        "gross_margin_kw_denom": 270000.0,
        "capacity_factor_mwh_denom": 270.0 * 168.0,
    },
}


def get_workbook_path():
    for path in WORKBOOK_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find workbook in any of: {WORKBOOK_CANDIDATES}")


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
        if series.notna().sum() > 0 and normalize_name(col) not in exclude_words:
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
    if any("HOUR" in norm for norm in cols_norm.values()):
        return df.copy()

    date_col = find_column(df.columns, ["DATE"])
    value_vars = []
    for col, norm in cols_norm.items():
        if norm in {"HE", "HOUR", "HOUR_ENDING"}:
            value_vars.append(col)
        elif norm.startswith("HE_") or norm.startswith("HOUR_"):
            value_vars.append(col)
        elif norm.isdigit() and 1 <= int(norm) <= 24:
            value_vars.append(col)
    if not value_vars:
        raise ValueError("Could not identify hourly columns to reshape from wide to long format.")

    long_df = df.melt(id_vars=[date_col], value_vars=value_vars, var_name="HOUR_LABEL", value_name="PRICE_ELECTRIC")
    long_df["HOUR_ENDING"] = long_df["HOUR_LABEL"].map(to_hour_ending)
    long_df = long_df.drop(columns=["HOUR_LABEL"])
    long_df = long_df.rename(columns={date_col: "OPERATING_DATE"})
    return long_df


def load_working_dataframe():
    workbook_path = get_workbook_path()
    excel_file = pd.ExcelFile(workbook_path, engine="openpyxl")
    print("Workbook used:", workbook_path)
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


def clean_mw(value, tol=1e-6):
    return 0.0 if abs(value) <= tol else float(value)


def build_single_unit_index(num_periods, num_segments):
    index = {}
    cursor = 0

    def add(name):
        nonlocal cursor
        index[name] = cursor
        cursor += 1

    for t in range(num_periods):
        add(("y", t))
    for t in range(num_periods):
        add(("s", t))
    for t in range(num_periods):
        for k in range(num_segments):
            add(("x", k, t))
    for t in range(num_periods):
        add(("p", t))
    return index, cursor


def add_sparse_row(rows, lows, highs, entries, low, high):
    row = {}
    for idx, coeff in entries:
        row[idx] = row.get(idx, 0.0) + coeff
    rows.append(row)
    lows.append(low)
    highs.append(high)


def solve_unit(df, unit_name, unit):
    num_periods = len(df)
    num_segments = len(unit["ihr"])
    segment_lengths = np.diff(unit["endpoints"])
    price_e = df["PRICE_ELECTRIC"].to_numpy()
    price_g = df["PRICE_GAS"].to_numpy()

    index, num_vars = build_single_unit_index(num_periods, num_segments)
    c = np.zeros(num_vars)
    lb = np.zeros(num_vars)
    ub = np.full(num_vars, np.inf)
    integrality = np.zeros(num_vars, dtype=int)

    for t in range(num_periods):
        y_idx = index[("y", t)]
        s_idx = index[("s", t)]
        p_idx = index[("p", t)]

        integrality[y_idx] = 1
        integrality[s_idx] = 1
        ub[y_idx] = 1.0
        ub[s_idx] = 1.0

        c[p_idx] = -price_e[t]
        c[y_idx] += unit["ml_dollar"] + price_g[t] * unit["ml_fuel"]
        c[s_idx] += unit["su_dollar"] + price_g[t] * unit["su_fuel"]

        for k, (seg_len, ihr) in enumerate(zip(segment_lengths, unit["ihr"])):
            x_idx = index[("x", k, t)]
            ub[x_idx] = seg_len
            c[x_idx] += ihr * price_g[t] + unit["vom"]

    rows = []
    lows = []
    highs = []

    for t in range(num_periods):
        add_sparse_row(
            rows,
            lows,
            highs,
            [
                (index[("p", t)], 1.0),
                (index[("y", t)], -unit["min_mw"]),
                *[(index[("x", k, t)], -1.0) for k in range(num_segments)],
            ],
            0.0,
            0.0,
        )

    for t in range(num_periods):
        for k, seg_len in enumerate(segment_lengths):
            add_sparse_row(
                rows,
                lows,
                highs,
                [
                    (index[("x", k, t)], 1.0),
                    (index[("y", t)], -seg_len),
                ],
                -np.inf,
                0.0,
            )

    # Tight startup logic with fixed initial OFF condition before hour 1.
    for t in range(num_periods):
        if t == 0:
            add_sparse_row(
                rows,
                lows,
                highs,
                [
                    (index[("s", t)], -1.0),
                    (index[("y", t)], 1.0),
                ],
                -np.inf,
                0.0,
            )
            add_sparse_row(
                rows,
                lows,
                highs,
                [
                    (index[("s", t)], 1.0),
                    (index[("y", t)], -1.0),
                ],
                -np.inf,
                0.0,
            )
            add_sparse_row(
                rows,
                lows,
                highs,
                [(index[("s", t)], 1.0)],
                -np.inf,
                1.0,
            )
        else:
            add_sparse_row(
                rows,
                lows,
                highs,
                [
                    (index[("s", t)], -1.0),
                    (index[("y", t)], 1.0),
                    (index[("y", t - 1)], -1.0),
                ],
                -np.inf,
                0.0,
            )
            add_sparse_row(
                rows,
                lows,
                highs,
                [
                    (index[("s", t)], 1.0),
                    (index[("y", t)], -1.0),
                ],
                -np.inf,
                0.0,
            )
            add_sparse_row(
                rows,
                lows,
                highs,
                [
                    (index[("s", t)], 1.0),
                    (index[("y", t - 1)], 1.0),
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

    matrix = np.zeros((len(rows), num_vars))
    matrix[row_idx, col_idx] = data

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix, np.array(lows), np.array(highs)),
    )

    status_map = {
        0: "Optimization terminated successfully",
        1: "Iteration or time limit reached",
        2: "Problem appears infeasible",
        3: "Problem appears unbounded",
        4: "Other solver issue",
    }
    solver_status = status_map.get(result.status, str(result.status))
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"{unit_name} optimization failed: {solver_status}. Message: {result.message}")

    solution = result.x
    generation = [clean_mw(solution[index[("p", t)]]) for t in range(num_periods)]
    on_values = [1 if solution[index[("y", t)]] > 0.5 else 0 for t in range(num_periods)]
    start_values = [1 if solution[index[("s", t)]] > 0.5 else 0 for t in range(num_periods)]

    total_revenue = 0.0
    total_startup_cost = 0.0
    total_min_load_cost = 0.0
    total_incremental_cost = 0.0
    startup_fuel_cost = 0.0
    min_load_fuel_cost = 0.0
    incremental_fuel_cost = 0.0

    for t in range(num_periods):
        total_revenue += price_e[t] * generation[t]
        if start_values[t] == 1:
            total_startup_cost += unit["su_dollar"] + price_g[t] * unit["su_fuel"]
            startup_fuel_cost += price_g[t] * unit["su_fuel"]
        if on_values[t] == 1:
            total_min_load_cost += unit["ml_dollar"] + price_g[t] * unit["ml_fuel"]
            min_load_fuel_cost += price_g[t] * unit["ml_fuel"]
            for k, ihr in enumerate(unit["ihr"]):
                x_val = solution[index[("x", k, t)]]
                total_incremental_cost += x_val * (ihr * price_g[t] + unit["vom"])
                incremental_fuel_cost += x_val * (ihr * price_g[t])

    total_costs = total_startup_cost + total_min_load_cost + total_incremental_cost
    total_mwh = float(np.sum(generation))
    fuel_costs = startup_fuel_cost + min_load_fuel_cost + incremental_fuel_cost

    summary = {
        "solver_status": solver_status,
        "gross_margin_per_kw": (total_revenue - total_costs) / unit["gross_margin_kw_denom"],
        "capacity_factor_pct": total_mwh / unit["capacity_factor_mwh_denom"] * 100.0,
        "total_revenue": total_revenue,
        "total_costs": total_costs,
        "fuel_costs": fuel_costs,
        "number_of_starts": int(sum(start_values)),
    }

    return generation, summary


def write_summary(unit_summaries, workbook_path):
    lines = [
        "Homework Task 3b Summary",
        "",
        f"Workbook used: {workbook_path}",
        "Pseudo-units were optimized independently as instructed.",
        "",
    ]

    for unit_name in ["Unit1", "Unit2"]:
        summary = unit_summaries[unit_name]
        lines.extend(
            [
                f"{unit_name}:",
                f"  Solver status: {summary['solver_status']}",
                f"  Gross Margin ($/kW): {summary['gross_margin_per_kw']:.6f}",
                f"  Capacity Factor (%): {summary['capacity_factor_pct']:.2f}",
                f"  Total Revenue ($): {summary['total_revenue']:.2f}",
                f"  Total Costs ($): {summary['total_costs']:.2f}",
                f"  Fuel Costs ($): {summary['fuel_costs']:.2f}",
                f"  Number of Starts: {summary['number_of_starts']}",
                "",
            ]
        )

    lines.extend(
        [
            "Simplifying assumptions used:",
            "- Used the same March 21-27, 2022 electricity and gas price extraction logic as Task 2.",
            "- CO2 cost was ignored.",
            "- Each pseudo-unit was optimized independently with no cross-unit dependency.",
            "- Initial condition for each pseudo-unit was OFF before hour 1.",
            "- Ramping constraints were omitted.",
            "- Minimum up/down time constraints were omitted.",
        ]
    )

    (OUTPUT_DIR / "summary_3b.txt").write_text("\n".join(lines))


def create_plot(result_df):
    hours = np.arange(1, len(result_df) + 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hours, result_df["MW_GENERATION_Unit1"], label="Unit 1", linewidth=1.8)
    ax.plot(hours, result_df["MW_GENERATION_Unit2"], label="Unit 2", linewidth=1.8)
    ax.set_title("Pseudo-Unit MW Generation: March 21-27, 2022")
    ax.set_xlabel("Hour Index")
    ax.set_ylabel("MW")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot_pseudo_mw.png", dpi=200)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    workbook_path = get_workbook_path()
    df = load_working_dataframe()

    unit_results = {}
    unit_summaries = {}
    for unit_name, unit in UNITS.items():
        generation, summary = solve_unit(df, unit_name, unit)
        unit_results[unit_name] = generation
        unit_summaries[unit_name] = summary

    result_df = df.copy()
    result_df["OPERATING_DATE"] = result_df["OPERATING_DATE"].dt.date.astype(str)
    result_df["MW_GENERATION_Unit1"] = unit_results["Unit1"]
    result_df["MW_GENERATION_Unit2"] = unit_results["Unit2"]
    result_df = result_df[
        [
            "OPERATING_DATE",
            "HOUR_ENDING",
            "PRICE_ELECTRIC",
            "PRICE_GAS",
            "MW_GENERATION_Unit1",
            "MW_GENERATION_Unit2",
        ]
    ]

    result_df.to_csv(OUTPUT_DIR / "CCGT_PSEUDO.csv", index=False)
    create_plot(result_df)
    write_summary(unit_summaries, workbook_path)

    print("\nFirst 12 rows of the final CSV:")
    print(result_df.head(12))

    print("\nSummary metrics:")
    for unit_name in ["Unit1", "Unit2"]:
        print(unit_name)
        for key, value in unit_summaries[unit_name].items():
            if key == "solver_status":
                print(f"  {key}: {value}")
            elif isinstance(value, (int, np.integer)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value:.6f}")

    print("\nOutput files:")
    print(OUTPUT_DIR / "solve_3b.py")
    print(OUTPUT_DIR / "CCGT_PSEUDO.csv")
    print(OUTPUT_DIR / "plot_pseudo_mw.png")
    print(OUTPUT_DIR / "summary_3b.txt")


if __name__ == "__main__":
    main()
