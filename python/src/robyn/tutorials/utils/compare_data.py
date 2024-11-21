import pandas as pd
import numpy as np
from tabulate import tabulate
from utils.data_mapper import load_data_from_json, import_data


def compare_featurized_mmm_data(r_python_data_path, python_only_data):
    loaded_data = load_data_from_json(r_python_data_path)
    imported_data = import_data(loaded_data)
    r_python_featurized_mmm_data = imported_data["featurized_mmm_data"]

    dt_mod_diff = compare_dataframes(
        r_python_featurized_mmm_data.dt_mod, python_only_data.dt_mod, "dt_mod"
    )
    dt_modRollWind_diff = compare_dataframes(
        r_python_featurized_mmm_data.dt_modRollWind,
        python_only_data.dt_modRollWind,
        "dt_modRollWind",
    )
    # modNLS_diff = compare_modNLS(r_python_featurized_mmm_data.modNLS, python_only_data.modNLS, "modNLS")
    modNLS_diff = None
    return dt_mod_diff, dt_modRollWind_diff, modNLS_diff


def compare_dataframes(df1, df2, name):
    result = [f"\n{name} DataFrame Comparison:"]

    if df1.empty and df2.empty:
        result.append(f"{name} DataFrames are both empty.")
        return "\n".join(result)

    shape_table = [
        ["", "R/Python", "New Python"],
        ["Rows", df1.shape[0], df2.shape[0]],
        ["Columns", df1.shape[1], df2.shape[1]],
    ]
    result.append("Shape Comparison:")
    result.append(tabulate(shape_table, headers="firstrow", tablefmt="grid"))

    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    common_cols = cols1.intersection(cols2)
    only_in_df1 = cols1 - cols2
    only_in_df2 = cols2 - cols1

    result.append("\nColumn Comparison:")
    col_table = [
        ["Common Columns", "Only in R/Python", "Only in New Python"],
        [
            ", ".join(sorted(common_cols)),
            ", ".join(sorted(only_in_df1)),
            ", ".join(sorted(only_in_df2)),
        ],
    ]
    result.append(tabulate(col_table, headers="firstrow", tablefmt="grid"))

    dtype_table = [["Column", "R/Python Type", "New Python Type"]]
    for col in common_cols:
        if df1[col].dtype != df2[col].dtype:
            dtype_table.append([col, df1[col].dtype, df2[col].dtype])

    if len(dtype_table) > 1:
        result.append("\nData Type Differences:")
        result.append(tabulate(dtype_table, headers="firstrow", tablefmt="grid"))
    else:
        result.append("\nAll data types are identical.")

    stats_diff = []
    for col in common_cols:
        if df1[col].dtype != df2[col].dtype:
            stats_diff.append([col, "Skipped (different data types)", ""])
        else:
            stats1 = calculate_summary_stats(df1[col])
            stats2 = calculate_summary_stats(df2[col])
            diff = compare_summary_stats(stats1, stats2)
            if diff:
                for stat, values in diff.items():
                    stats_diff.append(
                        [f"{col} ({stat})", values["R/Python"], values["New Python"]]
                    )

    if stats_diff:
        result.append("\nSummary Statistics Differences:")
        result.append(
            tabulate(
                stats_diff,
                headers=["Column (Statistic)", "R/Python", "New Python"],
                tablefmt="grid",
            )
        )
    else:
        result.append("\nAll summary statistics are identical within tolerance.")

    return "\n".join(result)


def calculate_summary_stats(series):
    if pd.api.types.is_numeric_dtype(series):
        return {
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
        }
    elif pd.api.types.is_datetime64_any_dtype(series):
        return {
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "median": series.median(),
        }
    else:
        return {
            "unique_count": series.nunique(),
            "most_common": series.value_counts().index[0] if not series.empty else None,
        }


def compare_summary_stats(stats1, stats2, tolerance=1e-5):
    diff = {}
    for key in stats1.keys():
        if key in ["min", "max", "mean", "median"]:
            if isinstance(stats1[key], (pd.Timestamp, np.datetime64)):
                if stats1[key] != stats2[key]:
                    diff[key] = {"R/Python": stats1[key], "New Python": stats2[key]}
            elif not np.isclose(
                stats1[key], stats2[key], rtol=tolerance, atol=tolerance, equal_nan=True
            ):
                diff[key] = {"R/Python": stats1[key], "New Python": stats2[key]}
        elif key == "std":
            if not np.isclose(
                stats1[key], stats2[key], rtol=tolerance, atol=tolerance, equal_nan=True
            ):
                diff[key] = {"R/Python": stats1[key], "New Python": stats2[key]}
        else:  # unique_count, most_common
            if stats1[key] != stats2[key]:
                diff[key] = {"R/Python": stats1[key], "New Python": stats2[key]}
    return diff


def compare_modNLS(dict1, dict2, name):
    result = [f"\n{name} Dictionary Comparison:"]

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    common_keys = keys1.intersection(keys2)
    only_in_dict1 = keys1 - keys2
    only_in_dict2 = keys2 - keys1

    result.append("Key Comparison:")
    key_table = [
        ["Common Keys", "Only in R/Python", "Only in New Python"],
        [
            ", ".join(sorted(common_keys)),
            ", ".join(sorted(only_in_dict1)),
            ", ".join(sorted(only_in_dict2)),
        ],
    ]
    result.append(tabulate(key_table, headers="firstrow", tablefmt="grid"))

    result.append("\nCommon Keys Analysis:")
    for key in common_keys:
        result.append(f"\nKey: {key}")
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            sub_diff = compare_nested_dict(dict1[key], dict2[key], f"{name}.{key}")
            if sub_diff:
                result.append("Differences in nested dictionary:")
                result.append(
                    tabulate(
                        sub_diff,
                        headers=["Subkey", "R/Python", "New Python"],
                        tablefmt="grid",
                    )
                )
            else:
                result.append("Nested dictionaries are identical within tolerance.")
        elif isinstance(dict1[key], pd.DataFrame) and isinstance(
            dict2[key], pd.DataFrame
        ):
            df_diff = compare_dataframes(dict1[key], dict2[key], f"{name}.{key}")
            result.append(df_diff)
        elif isinstance(dict1[key], (int, float, str, bool)):
            if np.isclose(dict1[key], dict2[key], equal_nan=True, rtol=1e-5, atol=1e-8):
                result.append(f"Values are identical within tolerance: {dict1[key]}")
            else:
                result.append(
                    f"Values differ: R/Python = {dict1[key]}, New Python = {dict2[key]}"
                )
        elif isinstance(dict1[key], np.ndarray) and isinstance(dict2[key], np.ndarray):
            if np.allclose(
                dict1[key], dict2[key], equal_nan=True, rtol=1e-5, atol=1e-8
            ):
                result.append("Arrays are identical within tolerance.")
                result.append(f"Shape: {dict1[key].shape}")
                result.append("Summary statistics:")
                stats = calculate_array_stats(dict1[key])
                result.append(
                    tabulate(
                        [["Statistic", "Value"]] + [[k, v] for k, v in stats.items()],
                        headers="firstrow",
                        tablefmt="grid",
                    )
                )
            else:
                result.append("Arrays are different.")
                result.append("Summary statistics for R/Python array:")
                stats1 = calculate_array_stats(dict1[key])
                result.append(
                    tabulate(
                        [["Statistic", "Value"]] + [[k, v] for k, v in stats1.items()],
                        headers="firstrow",
                        tablefmt="grid",
                    )
                )
                result.append("Summary statistics for New Python array:")
                stats2 = calculate_array_stats(dict2[key])
                result.append(
                    tabulate(
                        [["Statistic", "Value"]] + [[k, v] for k, v in stats2.items()],
                        headers="firstrow",
                        tablefmt="grid",
                    )
                )
        else:
            result.append(f"Unable to compare {type(dict1[key])} objects")

    return "\n".join(result)


def calculate_array_stats(arr):
    return {
        "min": np.min(arr),
        "max": np.max(arr),
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
    }


def compare_nested_dict(dict1, dict2, name):
    differences = []
    for key in set(dict1.keys()) | set(dict2.keys()):
        if key not in dict1:
            differences.append([f"{key}", "Missing", dict2[key]])
        elif key not in dict2:
            differences.append([f"{key}", dict1[key], "Missing"])
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            sub_diff = compare_nested_dict(dict1[key], dict2[key], f"{name}.{key}")
            differences.extend(sub_diff)
        elif isinstance(dict1[key], (int, float)) and isinstance(
            dict2[key], (int, float)
        ):
            if not np.isclose(
                dict1[key], dict2[key], equal_nan=True, rtol=1e-5, atol=1e-8
            ):
                differences.append([f"{key}", dict1[key], dict2[key]])
        elif dict1[key] != dict2[key]:
            differences.append([f"{key}", dict1[key], dict2[key]])
    return differences
