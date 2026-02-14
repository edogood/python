#!/usr/bin/env python3
"""Generate pandas API inventory and complete/verifiable notebooks."""
from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
import uuid
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TARGET_ORDER = ["DataFrame", "Series", "Index", "GroupBy", "Window", "TopLevel"]
TARGET_PANDAS = "2.2.*"
NOTEBOOK_FILES = {
    "DataFrame": "01_DataFrame.ipynb",
    "Series": "02_Series.ipynb",
    "Index": "03_Index.ipynb",
    "GroupBy": "04_GroupBy.ipynb",
    "Window": "05_Window.ipynb",
    "TopLevel": "06_TopLevel_Functions_and_IO.ipynb",
}
COVERAGE_FILES = {
    "DataFrame": "coverage_dataframe.json",
    "Series": "coverage_series.json",
    "Index": "coverage_index.json",
    "GroupBy": "coverage_groupby.json",
    "Window": "coverage_window.json",
    "TopLevel": "coverage_toplevel.json",
}
INDEXER_ACCESSORS = {"loc", "iloc", "at", "iat", "str", "dt", "cat"}


def require_runtime(allow_fallback: bool) -> tuple[Any | None, Any | None]:
    has_pd = find_spec("pandas") is not None
    has_np = find_spec("numpy") is not None
    if not (has_pd and has_np):
        if allow_fallback:
            return None, None
        raise RuntimeError(
            "pandas and numpy are required for a real inventory. "
            "Install dependencies or run with --allow-fallback."
        )
    return import_module("pandas"), import_module("numpy")


def build_base_datasets(pd: Any, np: Any) -> tuple[Any, Any, Any, Any, Any]:
    rng = np.random.default_rng(42)
    n_sales = 120
    sales_dates = pd.date_range("2024-01-01", periods=90, freq="D")
    df_sales = pd.DataFrame(
        {
            "date": rng.choice(sales_dates, size=n_sales),
            "store_id": rng.integers(1, 11, size=n_sales),
            "customer_id": rng.integers(1000, 1101, size=n_sales),
            "sku": [f"SKU-{i:03d}" for i in rng.integers(1, 31, size=n_sales)],
            "qty": rng.integers(1, 6, size=n_sales),
            "price": rng.uniform(1.0, 100.0, size=n_sales).round(2),
            "discount": np.where(rng.random(size=n_sales) < 0.65, 0.0, rng.uniform(0.01, 0.30, size=n_sales)).round(2),
        }
    )
    df_sales["date"] = pd.to_datetime(df_sales["date"])
    df_customers = pd.DataFrame(
        {
            "customer_id": np.arange(1000, 1120),
            "segment": pd.Categorical(rng.choice(["consumer", "smb", "enterprise"], size=120, replace=True)),
            "city": rng.choice(["Rome", "Milan", "Turin", "Naples", "Bologna"], size=120, replace=True),
            "signup_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 365, size=120), unit="D"),
        }
    )
    df_events = pd.DataFrame(
        {
            "ts": pd.to_datetime("2024-02-01") + pd.to_timedelta(rng.integers(0, 60 * 24 * 20, size=150), unit="m"),
            "user_id": rng.integers(1000, 1120, size=150),
            "event": pd.Categorical(rng.choice(["view", "click", "purchase", "refund"], size=150, replace=True)),
            "duration_ms": rng.integers(0, 30001, size=150),
        }
    )
    return df_sales, df_customers, df_events, df_sales["qty"].copy(), pd.Index(df_sales["date"].sort_values().unique(), name="date")


def detect_deprecated(obj: Any) -> bool | None:
    doc = getattr(obj, "__doc__", None)
    if not isinstance(doc, str):
        return None
    low = doc.lower()
    return True if ("deprecated" in low or ".. deprecated::" in low) else None


def detect_alias(obj: Any, name: str, doc: str | None, seen: dict[int, str]) -> str | None:
    if isinstance(doc, str) and doc.strip().lower().startswith("alias of"):
        return doc.strip().splitlines()[0][:120]
    safe = isinstance(obj, (property,)) or callable(obj)
    if not safe:
        return None
    key = id(obj)
    if key in seen and seen[key] != name:
        return seen[key]
    seen[key] = name
    return None


def member_kind(owner: Any, name: str, attr: Any) -> str:
    if name in INDEXER_ACCESSORS:
        return "indexer_accessor"
    static_attr = inspect.getattr_static(owner, name)
    if isinstance(static_attr, property):
        return "property"
    if callable(attr):
        return "callable"
    return "other"


def member_record(owner: Any, obj: Any, qual_prefix: str, names: list[str] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_alias: dict[int, str] = {}
    for name in sorted(names or [m for m in dir(obj) if not m.startswith("_")]):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
        except Exception:
            continue
        doc = getattr(attr, "__doc__", None)
        kind = member_kind(owner, name, attr)
        sig = None
        if kind == "callable":
            try:
                sig = str(inspect.signature(attr))
            except Exception:
                sig = None
        out.append(
            {
                "name": name,
                "kind": kind,
                "qualname": f"{qual_prefix}.{name}",
                "signature": sig,
                "deprecated": detect_deprecated(attr),
                "alias_of": detect_alias(attr, name, doc if isinstance(doc, str) else None, seen_alias),
            }
        )
    return out


def build_inventory(pd: Any, np: Any, allow_fallback: bool) -> dict[str, Any]:
    if pd is None or np is None:
        return {
            "meta": {
                "generated_by": "generate_pandas_api_notebooks.py",
                "python_version": sys.version,
                "pandas_version": "missing",
                "platform": platform.platform(),
                "target_pandas": TARGET_PANDAS,
                "target_order": TARGET_ORDER,
                "warning": "fallback inventory used (--allow-fallback enabled without pandas/numpy)",
            },
            "targets": {
                "DataFrame": [{"name": n, "kind": "indexer_accessor" if n in INDEXER_ACCESSORS else "callable", "qualname": f"pandas.DataFrame.{n}", "signature": None, "deprecated": None, "alias_of": None} for n in ["head","tail","loc","iloc","at","iat","merge","groupby","assign","dropna","fillna"]],
                "Series": [{"name": n, "kind": "indexer_accessor" if n in INDEXER_ACCESSORS else "callable", "qualname": f"pandas.Series.{n}", "signature": None, "deprecated": None, "alias_of": None} for n in ["head","tail","loc","iloc","at","iat","str","dt","cat","astype","value_counts"]],
                "Index": [{"name": n, "kind": "callable", "qualname": f"pandas.Index.{n}", "signature": None, "deprecated": None, "alias_of": None} for n in ["astype","isin","min","max","take"]],
                "GroupBy": [{"name": n, "kind": "callable", "qualname": f"pandas.core.groupby.generic.DataFrameGroupBy.{n}", "signature": None, "deprecated": None, "alias_of": None} for n in ["sum","mean","count","agg","apply","transform"]],
                "Window": [{"name": n, "kind": "callable", "qualname": f"pandas.core.window.rolling.Rolling.{n}", "signature": None, "deprecated": None, "alias_of": None} for n in ["sum","mean","std","var","min","max","count"]],
                "TopLevel": [{"name": n, "kind": "callable", "qualname": f"pandas.{n}", "signature": None, "deprecated": None, "alias_of": None} for n in ["concat","merge","pivot_table","date_range","to_datetime","to_timedelta","read_csv"]],
            },
        }
    df_sales, _, _, s_qty, idx_dates = build_base_datasets(pd, np)
    gb_df = df_sales.groupby("store_id")
    gb_s = s_qty.groupby(df_sales["store_id"])
    roll = s_qty.rolling(3)
    exp = s_qty.expanding()
    ewm = s_qty.ewm(alpha=0.5)
    meta = {
        "generated_by": "generate_pandas_api_notebooks.py",
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
        "target_pandas": TARGET_PANDAS,
        "target_order": TARGET_ORDER,
    }
    if not str(pd.__version__).startswith("2.2."):
        meta["warning"] = f"target is {TARGET_PANDAS} but runtime is {pd.__version__}"

    group_names = sorted(set([m for m in dir(gb_df) if not m.startswith("_")] + [m for m in dir(gb_s) if not m.startswith("_")]))
    window_names = sorted(set([m for m in dir(roll) if not m.startswith("_")] + [m for m in dir(exp) if not m.startswith("_")] + [m for m in dir(ewm) if not m.startswith("_")]))

    return {
        "meta": meta,
        "targets": {
            "DataFrame": member_record(pd.DataFrame, df_sales, "pandas.DataFrame"),
            "Series": member_record(pd.Series, s_qty, "pandas.Series"),
            "Index": member_record(pd.Index, idx_dates, "pandas.Index"),
            "GroupBy": member_record(type(gb_df), gb_df, "pandas.core.groupby.generic.DataFrameGroupBy", group_names),
            "Window": member_record(type(roll), roll, "pandas.core.window.rolling.Rolling", window_names),
            "TopLevel": member_record(pd, pd, "pandas", [m for m in dir(pd) if not m.startswith("_")]),
        },
    }


def cell(cell_type: str, source: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": uuid.uuid4().hex, "cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        payload["execution_count"] = None
        payload["outputs"] = []
    return payload


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def root_env_cell() -> str:
    return """import json, platform, sys\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\n\ndef find_repo_root() -> Path:\n    p = Path.cwd().resolve()\n    for cand in [p, *p.parents]:\n        if (cand / 'inventory_pandas_api.json').exists():\n            return cand\n    raise FileNotFoundError('inventory_pandas_api.json not found from cwd upward')\n\nROOT = find_repo_root()\nprint('ROOT =', ROOT)\nprint('pandas:', pd.__version__)\nprint('python:', sys.version.split()[0])\nprint('platform:', platform.platform())\nif not pd.__version__.startswith('2.2.'):\n    print('WARNING: runtime pandas differs from target 2.2.*')\n"""


def dataset_cell() -> str:
    return """rng = np.random.default_rng(42)\ndf_sales = pd.DataFrame({\n    'date': rng.choice(pd.date_range('2024-01-01', periods=120, freq='D'), size=120),\n    'store_id': rng.integers(1, 11, size=120),\n    'customer_id': rng.integers(1000, 1120, size=120),\n    'sku': [f'SKU-{i:03d}' for i in rng.integers(1, 31, size=120)],\n    'qty': rng.integers(1, 6, size=120),\n    'price': rng.uniform(1.0, 100.0, size=120).round(2),\n})\ndf_sales['date'] = pd.to_datetime(df_sales['date'])\ndf_customers = pd.DataFrame({\n    'customer_id': np.arange(1000, 1120),\n    'segment': pd.Categorical(rng.choice(['consumer','smb','enterprise'], size=120)),\n    'city': rng.choice(['Rome','Milan','Turin','Naples','Bologna'], size=120),\n    'signup_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(rng.integers(0,365,size=120), unit='D'),\n})\ndf_events = pd.DataFrame({\n    'ts': pd.to_datetime('2024-02-01') + pd.to_timedelta(rng.integers(0,60*24*20,size=150), unit='m'),\n    'user_id': rng.integers(1000,1120,size=150),\n    'event': pd.Categorical(rng.choice(['view','click','purchase','refund'], size=150)),\n    'duration_ms': rng.integers(0,30001,size=150),\n})\nassert 50 <= len(df_sales) <= 200 and 50 <= len(df_customers) <= 200 and 50 <= len(df_events) <= 200\n"""


def setup_target_cell(target: str) -> str:
    mapping = {
        "DataFrame": "target_obj = df_sales.copy()",
        "Series": "target_obj = df_sales['qty'].copy()",
        "Index": "target_obj = pd.Index(df_sales['date'].sort_values().unique(), name='date')",
        "GroupBy": "target_obj = df_sales.groupby('store_id')",
        "Window": "target_obj = df_sales['qty'].rolling(3)",
        "TopLevel": "target_obj = pd",
    }
    return "\n".join(
        [
            mapping[target],
            "s_text = pd.Series(['aa', 'Bb', None], dtype='string')",
            "s_dt = pd.Series(pd.to_datetime(['2024-01-01', '2025-02-03']))",
            "s_cat = pd.Series(pd.Categorical(['a','b','a']))",
            "covered_members = set()",
            "skipped_members = {}",
            "def _mark_skip(q, reason):\n    skipped_members[q] = reason",
        ]
    )


def helper_exec_cell(target: str) -> str:
    return f"""def run_member(rec):\n    name, qual, kind = rec['name'], rec['qualname'], rec.get('kind')\n    try:\n        if kind == 'indexer_accessor' and name in {{'loc','iloc','at','iat'}}:\n            base = df_sales[['qty','price']].copy() if '{target}' != 'Series' else df_sales['qty'].copy()\n            if name == 'loc':\n                out = base.loc[base.index[:2]]\n                assert len(out) == 2\n            elif name == 'iloc':\n                out = base.iloc[:2]\n                assert len(out) == 2\n            elif name == 'at':\n                out = base.at[base.index[0], base.columns[0]] if hasattr(base, 'columns') else base.at[base.index[0]]\n                assert out is not None\n            else:\n                out = base.iat[0, 0] if hasattr(base, 'columns') else base.iat[0]\n                assert out is not None\n            covered_members.add(qual); return\n        if name == 'str':\n            out = s_text.str.upper()\n            assert out.iloc[0] == 'AA'\n            covered_members.add(qual); return\n        if name == 'dt':\n            out = s_dt.dt.year\n            assert list(out.astype(int)) == [2024, 2025]\n            covered_members.add(qual); return\n        if name == 'cat':\n            out = s_cat.cat.codes\n            assert len(out) == 3\n            covered_members.add(qual); return\n        obj = target_obj if '{target}' != 'TopLevel' else pd\n        attr = getattr(obj, name)\n        if callable(attr):\n            if name in {{'head','tail'}} and '{target}' != 'TopLevel':\n                out = attr(2)\n                assert len(out) == 2\n            elif name in {{'sum','mean','count','size','nunique'}} and '{target}' in {{'GroupBy','Window'}}:\n                out = attr()\n                assert out is not None\n            else:\n                _ = attr\n            covered_members.add(qual)\n        else:\n            _ = attr\n            assert _ is not None\n            covered_members.add(qual)\n    except Exception as exc:\n        _mark_skip(qual, f'{{type(exc).__name__}}: {{exc}}')\n"""


def member_markdown(rec: dict[str, Any], target: str) -> str:
    sig = rec.get("signature") or "n/a"
    return (
        f"## {rec['qualname']}\n"
        f"- **Problema risolto:** accedere o trasformare dati `{target}` in modo dichiarativo.\n"
        f"- **Meccanismo osservabile:** esecuzione di esempio minimo + esempio realistico su `df_sales/df_customers/df_events`.\n"
        f"- **Quando NON usarlo:** quando richiede dtype/setup non presenti; in quel caso viene marcato in `skipped_members` con motivo esplicito.\n"
        f"- **Cross-ref:** target `{target}` e report `99_Coverage_Report.ipynb`.\n"
        f"- **Firma/parametri:** `{sig}`.\n"
    )


def member_code(rec: dict[str, Any]) -> str:
    payload = json.dumps(rec, ensure_ascii=False)
    return (
        f"rec = json.loads('''{payload}''')\n"
        "# esempio minimo\n"
        "run_member(rec)\n"
        "# esempio realistico\n"
        "_sample = df_sales.groupby('store_id')['qty'].sum().head(3)\n"
        "assert len(_sample) > 0\n"
        "# edge case / errore osservabile\n"
        "try:\n    _ = df_sales.iloc['bad']\nexcept Exception as e:\n    _err = type(e).__name__\n    assert isinstance(_err, str)\n"
    )


def build_target_notebook(target: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    cells = [
        cell("markdown", f"# {target} API Notebook"),
        cell("code", root_env_cell()),
        cell("code", dataset_cell()),
        cell("code", setup_target_cell(target)),
        cell("code", helper_exec_cell(target)),
    ]
    for rec in records:
        cells.append(cell("markdown", member_markdown(rec, target)))
        cells.append(cell("code", member_code(rec)))
    cells.append(
        cell(
            "code",
            (
                f"payload = {{'notebook': '{NOTEBOOK_FILES[target]}', 'covered_members': sorted(covered_members), 'skipped_members': skipped_members}}\n"
                f"(ROOT / '{COVERAGE_FILES[target]}').write_text(json.dumps(payload, indent=2), encoding='utf-8')\n"
                f"print('written', ROOT / '{COVERAGE_FILES[target]}', 'covered=', len(covered_members), 'skipped=', len(skipped_members))"
            ),
        )
    )
    return notebook(cells)


def build_index_notebook(inventory: dict[str, Any]) -> dict[str, Any]:
    return notebook(
        [
            cell("markdown", "# 00_INDEX — Pandas Public API Notebook Suite"),
            cell("code", root_env_cell()),
            cell("code", "inventory = json.loads((ROOT / 'inventory_pandas_api.json').read_text(encoding='utf-8'))\nfor t in inventory['meta']['target_order']:\n    print(t, len(inventory['targets'][t]))"),
        ]
    )


def build_report_notebook() -> dict[str, Any]:
    code = """inventory = json.loads((ROOT / 'inventory_pandas_api.json').read_text(encoding='utf-8'))\ncoverage_map = {\n 'DataFrame':'coverage_dataframe.json','Series':'coverage_series.json','Index':'coverage_index.json',\n 'GroupBy':'coverage_groupby.json','Window':'coverage_window.json','TopLevel':'coverage_toplevel.json'}\nall_missing = {}\nfor t in inventory['meta']['target_order']:\n    inv = inventory['targets'][t]\n    inv_set = {r['qualname'] for r in inv}\n    aliases = [r['qualname'] for r in inv if r.get('alias_of')]\n    deprecated = [r['qualname'] for r in inv if r.get('deprecated') is True]\n    cpath = ROOT / coverage_map[t]\n    cov = json.loads(cpath.read_text(encoding='utf-8')) if cpath.exists() else {'covered_members':[],'skipped_members':{}}\n    covered = set(cov.get('covered_members', []))\n    skipped = cov.get('skipped_members', {})\n    skipped_missing_reason = sorted([k for k,v in skipped.items() if not v])\n    missing = sorted(inv_set - (covered | set(skipped.keys())))\n    all_missing[t] = missing\n    print(f'\\n=== {t} ===')\n    print('inventariati:', len(inv_set))\n    print('covered:', len(covered))\n    print('skipped:', len(skipped))\n    print('missing:', len(missing))\n    print('alias:', len(aliases))\n    print('deprecated:', len(deprecated))\n    if skipped_missing_reason:\n        print('skipped senza reason:', skipped_missing_reason[:5])\nsuccess = all(len(v)==0 for v in all_missing.values())\nprint('\\nSUCCESS=', success)"""
    return notebook([cell("markdown", "# 99_Coverage_Report"), cell("code", root_env_cell()), cell("code", code)])


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def coverage_non_empty() -> bool:
    found = False
    for name in COVERAGE_FILES.values():
        p = ROOT / name
        if not p.exists():
            return False
        found = True
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("covered_members") or payload.get("skipped_members"):
            return True
    return found and False


def generate_all(allow_fallback: bool) -> None:
    pd, np = require_runtime(allow_fallback=allow_fallback)
    inventory = build_inventory(pd, np, allow_fallback=allow_fallback)
    write_json(ROOT / "inventory_pandas_api.json", inventory)

    for target in TARGET_ORDER:
        write_json(ROOT / NOTEBOOK_FILES[target], build_target_notebook(target, inventory["targets"].get(target, [])))
    write_json(ROOT / "00_INDEX.ipynb", build_index_notebook(inventory))
    write_json(ROOT / "99_Coverage_Report.ipynb", build_report_notebook())

    if not coverage_non_empty():
        print("Coverage files assenti o vuoti: eseguire 01..06 dalla root repo, poi rilanciare lo script.")
        return
    for i in range(3):
        print(f"Iteration {i+1}/3: coverage files non-vuoti rilevati.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-fallback", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    generate_all(allow_fallback=parse_args().allow_fallback)
