# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, bootstrap


def divide_list_into_chunks(lst: list, n: int) -> list[list]:
    """
    Divide a Sequence into n approximately equal-sized chunks.

    Args:
        lst (list[Any]): The list to be divided.
        n (int): The number of chunks to create.

    Returns:
        list[list[Any]]: A list of n chunks, where each chunk is a sublist of lst.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def int_to_letters(n: int) -> str:
    """
    Convert int to letters.
    Useful for converting chain index to label_asym_id.

    Args:
        n (int): int number
    Returns:
        str: letters. e.g. 1 -> A, 2 -> B, 27 -> AA, 28 -> AB
    """
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def nested_dict_to_sorted_list(data: dict | Any) -> list | Any:
    """
    Convert a nested dictionary into a sorted list.

    This function takes a nested dictionary and converts it into a sorted list.
    If the input is a dictionary, it sorts the keys and recursively processes the values.
    If the input is not a dictionary, it returns the value directly.

    Args:
        data (dict or Any): The input data, which can be a dictionary or any other type.

    Returns:
        list or Any: The sorted list or the original value if the input is not a dictionary.
    """
    if isinstance(data, dict):
        # If the input is a dictionary, sort the keys and recursively process the values
        try:
            for i in data.keys():
                int(i)
            key_type = int
        except ValueError:
            key_type = str
        return [
            nested_dict_to_sorted_list(data[key])
            for key in sorted(data.keys(), key=key_type)
        ]
    else:
        # If the input is not a dictionary, return the value directly
        return data


def get_infer_cif_path(
    infer_output_dir: Path, model: str, entry_id: str, seed: str, sample: str
) -> Path:
    """
    Get the path to the inferred CIF file based on the model name.

    Args:
        infer_output_dir (Path): The directory where inference outputs are stored.
        model (str): The name of the model used for inference.
        entry_id (str): The identifier for the entry.
        seed (str): The seed value used in the inference process.
        sample (str): The sample identifier.

    Returns:
        Path: The path to the inferred CIF file.

    Raises:
        NotImplementedError: If the provided model name is not recognized.
    """
    if model == "protenix":
        cif_path = (
            infer_output_dir
            / entry_id
            / entry_id
            / f"seed_{seed}"
            / "predictions"
            / f"{entry_id}_sample_{sample}.cif"
        )
    elif model == "chai":
        cif_path = infer_output_dir / entry_id / seed / f"pred.model_idx_{sample}.cif"
    elif model == "boltz":
        cif_path = (
            infer_output_dir
            / entry_id
            / f"seed_{seed}"
            / f"boltz_results_{entry_id}"
            / "predictions"
            / entry_id
            / f"{entry_id}_model_{sample}.cif"
        )
    else:
        raise NotImplementedError(f"Unknown model: {model}")
    return cif_path


def get_eval_result_json_path(
    eval_result_dir: Path, entry_id: str, seed: str, sample: str
) -> Path:
    """
    Get the path to the evaluation result JSON file.

    This function constructs the path to the JSON file that contains the evaluation results
    based on the provided evaluation result directory, entry ID, seed, and sample identifier.

    Args:
        eval_result_dir (Path): The directory where evaluation results are stored.
        entry_id (str): The identifier for the entry.
        seed (str): The seed value used in the evaluation process.
        sample (str): The sample identifier.

    Returns:
        Path: The path to the evaluation result JSON file.
    """
    return eval_result_dir / entry_id / str(seed) / f"sample_{sample}_metrics.json"


def get_bootstrap_ci(
    data: list[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n: int = 10000,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the mean of a distribution.

    Args:
        data (list[float]): The data to bootstrap.
        statistic (Callable[[np.ndarray], float], optional): The statistic to calculate. Defaults to np.mean.
        n (int, optional): The number of bootstrap samples to generate. Defaults to 10000.

    Returns:
        tuple[float, float]: The lower and upper bounds of the confidence interval.
    """
    if len(data) == 0:
        logging.warning(
            "Data is empty, cannot calculate confidence \
                interval for bootstrap. return (0, 0)"
        )
        ci_lower, ci_upper = 0.0, 0.0
    elif len(data) == 1:
        logging.warning(
            "Data has only one element, cannot calculate confidence \
                interval for bootstrap. return (data[0], data[0])"
        )
        ci_lower, ci_upper = data[0], data[0]
    else:
        data = (data,)
        bootstrap_result = bootstrap(data, statistic, n_resamples=n)

        ci_lower, ci_upper = bootstrap_result.confidence_interval
    return round(ci_lower, 4), round(ci_upper, 4)


def get_binomial_ci(total_num: int, success_num: int) -> tuple[float, float]:
    """
    Calculate the Clopper-Pearson interval (exact binomial confidence interval)
    for a binomial distribution.

    Args:
        total_num (int): The total number of trials.
        success_num (int): The number of successful trials.

    Returns:
        tuple[float, float]: The lower and upper bounds of the confidence interval.
    """
    binomtest_result = binomtest(success_num, total_num).proportion_ci(0.95)
    ci_lower, ci_upper = binomtest_result
    return round(ci_lower, 4), round(ci_upper, 4)


def fmt_bytes(n: int) -> str:
    """Format a byte count into a human-readable string.

    Args:
        n: Number of bytes.

    Returns:
        A string formatted with a unit suffix (B, KB, MB, GB, TB, PB).
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def shrink_dataframe(
    df: pd.DataFrame,
    *,
    cat_threshold: int = 256,
    cat_ratio: float = 0.5,
    object_to_string: bool = True,
    downcast_float: bool = True,
    downcast_int: bool = True,
    use_nullable_int: bool = True,
    bool_cast: bool = True,
    exclude: Iterable[str] = (),
    report_topk: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Downcast and recode columns to minimize memory and file size.

    This function applies a series of safe transformations to reduce memory
    footprint without changing column semantics:
    - Booleans: convert 0/1 or True/False to bool/boolean dtypes.
    - Floats: downcast float64 to float32 (optional).
    - Integers: downcast int64 to the smallest fitting integer dtype.
    - Object columns with integer-like values: convert to nullable Int* dtypes.
    - Low-cardinality text: convert to ``category``.
    - Other object text: convert to pandas ``string`` (Arrow/Parquet friendly).

    Args:
        df: Input DataFrame.
        cat_threshold: If the number of unique non-null values is ≤ this value,
            convert to ``category``.
        cat_ratio: Alternatively, if ``nunique / len(df)`` ≤ this ratio,
            convert to ``category``.
        object_to_string: Convert remaining ``object`` text columns to
            pandas ``string`` dtype.
        downcast_float: If True, downcast ``float64`` to ``float32``.
        downcast_int: If True, downcast ``int64`` to the smallest fitting
            signed integer dtype.
        use_nullable_int: If True, convert integer-like ``object`` columns
            (possibly with missing values) to nullable ``Int8/16/32/64``.
        bool_cast: If True, convert 0/1 (non-float) or boolean-like columns to
            ``bool`` / nullable ``boolean``.
        exclude: Column names to skip from any transformation.
        report_topk: Number of columns to include in the per-column memory
            savings summary.

    Returns:
        A 2-tuple ``(df_out, report)`` where:
        - ``df_out``: The transformed DataFrame with reduced memory usage.
        - ``report``: A dict containing summary metrics:
            - ``mem_before_bytes`` / ``mem_after_bytes`` / ``mem_saved_bytes``
            - human-readable counterparts (``*_readable``)
            - ``shrink_ratio`` (before/after)
            - ``changed_cols``: mapping of column -> (old_dtype, new_dtype)
            - ``top_saving_cols``: memory saved by column (top-K)

    Notes:
        - Conversions are conservative and try to preserve semantics.
        - For columns critical to numeric precision (e.g., scores), add them to
          ``exclude`` to prevent downcasting.
        - The function does not modify the input ``df`` in place.

    Examples:
        >>> df_small, rpt = shrink_dataframe(df, exclude=["critical_score"])
        >>> rpt["mem_before_readable"], rpt["mem_after_readable"]
        ('1.20 GB', '420.00 MB')
    """
    src_mem = df.memory_usage(deep=True).sum()
    out = df.copy()

    excl = set(exclude)
    changes: dict[str, tuple[str, str]] = {}

    # Iterate over columns and apply dtype reductions.
    for col in out.columns:
        if col in excl:
            continue

        s = out[col]
        old_dtype = s.dtype

        # 1) Boolean casting: recognize 0/1 (non-float) or existing bools.
        if bool_cast:
            if pd.api.types.is_bool_dtype(s):
                pass  # already boolean
            elif set(np.unique(s.dropna().values)).issubset(
                {0, 1}
            ) and not pd.api.types.is_float_dtype(s):
                out[col] = s.astype("boolean") if s.isna().any() else s.astype(bool)
                changes[col] = (old_dtype, out[col].dtype)
                continue

        # 2) Numeric downcasting.
        if pd.api.types.is_float_dtype(s):
            if downcast_float and s.dtype == "float64":
                out[col] = pd.to_numeric(s, downcast="float")
                changes[col] = (old_dtype, out[col].dtype)
                continue

        elif pd.api.types.is_integer_dtype(s):
            if downcast_int and s.dtype == "int64":
                out[col] = pd.to_numeric(s, downcast="integer")
                changes[col] = (old_dtype, out[col].dtype)
                continue

        # 3) Nullable integers for integer-like object columns.
        elif use_nullable_int and s.dtype == "object":
            sample = s.sample(min(len(s), 5000), random_state=0)
            try_parse = pd.to_numeric(sample, errors="coerce", downcast="integer")
            if try_parse.notna().mean() > 0.98 and (try_parse % 1 == 0).all():
                parsed = pd.to_numeric(s, errors="coerce", downcast="integer")
                if parsed.isna().any():
                    # Choose the smallest nullable integer dtype that can hold the range.
                    iinfo = pd.Series(parsed.dropna().astype("int64"))
                    minv, maxv = iinfo.min(), iinfo.max()
                    if -128 <= minv and maxv <= 127:
                        out[col] = parsed.astype("Int8")
                    elif -32768 <= minv and maxv <= 32767:
                        out[col] = parsed.astype("Int16")
                    elif -2147483648 <= minv and maxv <= 2147483647:
                        out[col] = parsed.astype("Int32")
                    else:
                        out[col] = parsed.astype("Int64")
                else:
                    out[col] = pd.to_numeric(parsed, downcast="integer")
                changes[col] = (old_dtype, out[col].dtype)
                continue

        # 4) Text handling: low-cardinality -> category; otherwise -> string.
        if s.dtype == "object":
            nunq = s.nunique(dropna=True)
            if nunq <= cat_threshold or nunq <= len(s) * cat_ratio:
                out[col] = s.astype("category")
            elif object_to_string:
                out[col] = s.astype("string")
            if out[col].dtype != old_dtype:
                changes[col] = (old_dtype, out[col].dtype)

    # Build memory report.
    dst_mem = out.memory_usage(deep=True).sum()
    delta = src_mem - dst_mem
    ratio = (src_mem / max(dst_mem, 1)) if dst_mem else np.inf

    mem_before = df.memory_usage(deep=True)
    mem_after = out.memory_usage(deep=True)
    diff = (mem_before - mem_after).sort_values(ascending=False)

    report: dict[str, Any] = {
        "mem_before_bytes": int(src_mem),
        "mem_after_bytes": int(dst_mem),
        "mem_saved_bytes": int(delta),
        "mem_before_readable": fmt_bytes(src_mem),
        "mem_after_readable": fmt_bytes(dst_mem),
        "mem_saved_readable": fmt_bytes(delta),
        "shrink_ratio": float(ratio),
        "changed_cols": {k: (str(v0), str(v1)) for k, (v0, v1) in changes.items()},
        "top_saving_cols": diff.head(report_topk).to_dict(),
    }
    return out, report
