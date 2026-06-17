from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")
SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
CSV_SHEET_NAME = "CSV"


@dataclass(frozen=True)
class RawDataFile:
    path: Path
    name: str
    suffix: str
    size_kb: float


def _resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def list_raw_files(raw_dir: Path | str = RAW_DATA_DIR) -> list[RawDataFile]:
    directory = Path(raw_dir)
    if not directory.exists():
        return []

    files: list[RawDataFile] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        files.append(
            RawDataFile(
                path=path,
                name=path.name,
                suffix=path.suffix.lower(),
                size_kb=round(path.stat().st_size / 1024.0, 1),
            )
        )
    return files


def is_excel_file(path: Path | str) -> bool:
    return Path(path).suffix.lower() in {".xls", ".xlsx"}


def is_csv_file(path: Path | str) -> bool:
    return Path(path).suffix.lower() == ".csv"


def get_sheet_names(path: Path | str) -> list[str]:
    file_path = _resolve_path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return [CSV_SHEET_NAME]
    if suffix in {".xls", ".xlsx"}:
        with pd.ExcelFile(file_path) as workbook:
            return list(workbook.sheet_names)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def load_table(
    path: Path | str,
    sheet_name: str | None = None,
    header_row: int | None = 0,
    nrows: int | None = None,
) -> pd.DataFrame:
    file_path = _resolve_path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path, header=header_row, nrows=nrows)
    if suffix in {".xls", ".xlsx"}:
        selected_sheet = 0 if sheet_name in {None, CSV_SHEET_NAME} else sheet_name
        return pd.read_excel(
            file_path,
            sheet_name=selected_sheet,
            header=header_row,
            nrows=nrows,
        )
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def preview_table(
    path: Path | str,
    sheet_name: str | None = None,
    header_row: int | None = 0,
    rows: int = 25,
) -> pd.DataFrame:
    return load_table(
        path,
        sheet_name=sheet_name,
        header_row=header_row,
        nrows=rows,
    )


def get_sheet_columns(
    path: Path | str,
    sheet_name: str | None = None,
    header_row: int | None = 0,
) -> list[str]:
    preview = load_table(
        path,
        sheet_name=sheet_name,
        header_row=header_row,
        nrows=0,
    )
    return [str(column) for column in preview.columns]


def summarize_dataframe(df: pd.DataFrame) -> dict[str, object]:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    datetime_columns = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    empty_columns = [column for column in df.columns if df[column].isna().all()]

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": [str(column) for column in numeric_columns],
        "datetime_columns": [str(column) for column in datetime_columns],
        "empty_columns": [str(column) for column in empty_columns],
    }


def describe_raw_file(path: Path | str) -> dict[str, object]:
    file_path = _resolve_path(path)
    sheet_names = get_sheet_names(file_path)
    return {
        "name": file_path.name,
        "path": str(file_path),
        "suffix": file_path.suffix.lower(),
        "size_kb": round(file_path.stat().st_size / 1024.0, 1),
        "sheets": sheet_names,
        "sheet_count": len(sheet_names),
    }
