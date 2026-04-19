from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


FEATURE_COLUMNS = ["Al", "Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W"]
TARGET_COLUMNS = ["yield_strength_1000C", "fracture_strain_RT"]


@dataclass(frozen=True)
class ModelSpec:
	name: str
	build_estimator: Callable[[int], RegressorMixin]
	param_grid: dict


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compare 9 regression models on alloy property prediction with repeated "
			"80/20 split, 5-fold CV tuning, and test-set metric aggregation."
		)
	)
	parser.add_argument("--data-path", default="datest.csv", help="Path to the CSV dataset.")
	parser.add_argument("--outdir", default="compare_outputs", help="Output directory for CSVs and figures.")
	parser.add_argument("--repeats", type=int, default=100, help="Number of repeated random train/test splits.")
	parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio for each split.")
	parser.add_argument("--base-seed", type=int, default=42, help="Base random seed. split_seed = base_seed + repeat_index.")
	parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV.")
	parser.add_argument(
		"--smoke",
		action="store_true",
		help="Quick validation mode. Uses 5 repeats and default output dir compare_outputs_smoke.",
	)
	return parser.parse_args()


def metric_rmse(y_exp: np.ndarray, y_pre: np.ndarray) -> float:
	return float(np.sqrt(np.mean((y_exp - y_pre) ** 2)))


def metric_mae(y_exp: np.ndarray, y_pre: np.ndarray) -> float:
	return float(np.mean(np.abs(y_exp - y_pre)))


def metric_r2_paper(y_exp: np.ndarray, y_pre: np.ndarray) -> float:
	"""
	Paper formula (as provided by user):
	r2 = sum((y_exp - mean(y_exp)) * (y_pre - mean(y_pre))) /
		 sqrt(sum((y_exp - mean(y_exp))^2) * sum((y_pre - mean(y_pre))^2))
	"""
	y_exp_centered = y_exp - np.mean(y_exp)
	y_pre_centered = y_pre - np.mean(y_pre)
	denominator = np.sqrt(np.sum(y_exp_centered**2) * np.sum(y_pre_centered**2))
	if denominator == 0.0:
		return float("nan")
	return float(np.sum(y_exp_centered * y_pre_centered) / denominator)


def scaled_pipeline(model: RegressorMixin) -> Pipeline:
	return Pipeline([("scaler", StandardScaler()), ("model", model)])


def build_model_specs() -> list[ModelSpec]:
	return [
		ModelSpec(
			name="Linear Regression",
			build_estimator=lambda seed: scaled_pipeline(LinearRegression()),
			param_grid={},
		),
		ModelSpec(
			name="Ridge Regression",
			build_estimator=lambda seed: scaled_pipeline(Ridge(random_state=seed)),
			param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0]},
		),
		ModelSpec(
			name="Lasso Regression",
			build_estimator=lambda seed: scaled_pipeline(
				Lasso(max_iter=50000, random_state=seed)
			),
			param_grid={"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1]},
		),
		ModelSpec(
			name="Elastic Net",
			build_estimator=lambda seed: scaled_pipeline(
				ElasticNet(max_iter=50000, random_state=seed)
			),
			param_grid={
				"model__alpha": [1e-3, 1e-2, 1e-1],
				"model__l1_ratio": [0.2, 0.5, 0.8],
			},
		),
		ModelSpec(
			name="Kernel Ridge Regression",
			build_estimator=lambda seed: scaled_pipeline(KernelRidge(kernel="rbf")),
			param_grid={
				"model__alpha": [1e-2, 1e-1, 1.0],
				"model__gamma": [0.01, 0.1, 1.0],
			},
		),
		ModelSpec(
			name="SVR (Linear Kernel)",
			build_estimator=lambda seed: scaled_pipeline(SVR(kernel="linear")),
			param_grid={
				"model__C": [0.1, 1.0, 10.0],
				"model__epsilon": [0.01, 0.1],
			},
		),
		ModelSpec(
			name="SVR (RBF Kernel)",
			build_estimator=lambda seed: scaled_pipeline(SVR(kernel="rbf")),
			param_grid={
				"model__C": [1.0, 10.0, 100.0],
				"model__epsilon": [0.01, 0.1],
				"model__gamma": [0.01, 0.1, 1.0],
			},
		),
		ModelSpec(
			name="Gaussian Process Regression",
			build_estimator=lambda seed: scaled_pipeline(
				GaussianProcessRegressor(
					kernel=ConstantKernel(1.0, (1e-2, 1e3))
					* RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)),
					normalize_y=True,
					random_state=seed,
				)
			),
			param_grid={"model__alpha": [1e-10, 1e-6, 1e-3]},
		),
		ModelSpec(
			name="Random Forest",
			build_estimator=lambda seed: RandomForestRegressor(random_state=seed, n_jobs=1),
			param_grid={
				"n_estimators": [200, 500],
				"max_features": ["sqrt", 1.0],
				"min_samples_leaf": [1, 2],
			},
		),
	]


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
	required_columns = FEATURE_COLUMNS + TARGET_COLUMNS
	missing = [col for col in required_columns if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	numeric_df = df.copy()
	for col in required_columns:
		numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

	clean_df = numeric_df.dropna(subset=required_columns)
	if clean_df.empty:
		raise ValueError("No valid rows remain after dropping NaN values in required columns.")

	n_train_min = int(np.floor((1.0 - 0.2) * len(clean_df)))
	if n_train_min < 5:
		raise ValueError("Dataset too small for 5-fold CV after 80/20 split.")

	return clean_df


def evaluate_target(
	dataset: pd.DataFrame,
	target_col: str,
	model_specs: list[ModelSpec],
	repeats: int,
	test_size: float,
	base_seed: int,
	n_jobs: int,
) -> tuple[list[dict], pd.DataFrame]:
	X = dataset[FEATURE_COLUMNS]
	y = dataset[target_col]

	detail_rows: list[dict] = []
	first_repeat_predictions: pd.DataFrame | None = None

	for repeat_idx in range(repeats):
		split_seed = base_seed + repeat_idx
		X_train, X_test, y_train, y_test = train_test_split(
			X,
			y,
			test_size=test_size,
			random_state=split_seed,
			shuffle=True,
		)

		cv = KFold(n_splits=5, shuffle=True, random_state=split_seed)

		if repeat_idx % 10 == 0 or repeat_idx == repeats - 1:
			print(
				f"Target={target_col} | repeat {repeat_idx + 1}/{repeats} | split_seed={split_seed}"
			)

		if repeat_idx == 0:
			first_repeat_predictions = pd.DataFrame(
				{
					"SampleIndex": X_test.index.to_numpy(),
					"Actual": y_test.to_numpy(dtype=float),
				}
			)

		for spec in model_specs:
			search = GridSearchCV(
				estimator=spec.build_estimator(split_seed),
				param_grid=spec.param_grid,
				cv=cv,
				scoring="neg_mean_squared_error",
				n_jobs=n_jobs,
				refit=True,
			)
			search.fit(X_train, y_train)
			y_pred = search.best_estimator_.predict(X_test)

			y_true_np = y_test.to_numpy(dtype=float)
			y_pred_np = np.asarray(y_pred, dtype=float)

			rmse = metric_rmse(y_true_np, y_pred_np)
			mae = metric_mae(y_true_np, y_pred_np)
			r2 = metric_r2_paper(y_true_np, y_pred_np)

			detail_rows.append(
				{
					"Repeat": repeat_idx + 1,
					"SplitSeed": split_seed,
					"Target": target_col,
					"Model": spec.name,
					"RMSE": rmse,
					"MAE": mae,
					"R2": r2,
					"BestParams": json.dumps(
						search.best_params_,
						ensure_ascii=False,
						sort_keys=True,
						default=str,
					),
				}
			)

			if repeat_idx == 0 and first_repeat_predictions is not None:
				first_repeat_predictions[spec.name] = y_pred_np

	if first_repeat_predictions is None:
		first_repeat_predictions = pd.DataFrame()

	first_repeat_predictions = first_repeat_predictions.sort_values("SampleIndex").reset_index(drop=True)
	return detail_rows, first_repeat_predictions


def build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
	summary = (
		detail_df.groupby(["Target", "Model"], as_index=False)
		.agg(
			RMSE_mean=("RMSE", "mean"),
			RMSE_std=("RMSE", "std"),
			MAE_mean=("MAE", "mean"),
			MAE_std=("MAE", "std"),
			R2_mean=("R2", "mean"),
			R2_std=("R2", "std"),
		)
		.fillna(0.0)
	)

	summary = summary.sort_values(
		by=["Target", "RMSE_mean", "R2_mean"], ascending=[True, True, False]
	).reset_index(drop=True)
	return summary


def select_best_model_per_target(summary_df: pd.DataFrame) -> pd.DataFrame:
	best_rows: list[pd.Series] = []
	for target, subset in summary_df.groupby("Target"):
		best_row = subset.sort_values(
			by=["RMSE_mean", "R2_mean"], ascending=[True, False]
		).iloc[0]
		best_rows.append(best_row)
	return pd.DataFrame(best_rows).reset_index(drop=True)


def plot_target_metrics(summary_df: pd.DataFrame, target: str, outdir: Path) -> None:
	target_df = summary_df[summary_df["Target"] == target].copy()
	target_df = target_df.sort_values(by=["RMSE_mean", "R2_mean"], ascending=[True, False])

	models = target_df["Model"].tolist()
	x = np.arange(len(models))

	fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

	metric_specs = [
		("R2_mean", "R2_std", "R2 (paper formula, higher is better)", "#1f77b4"),
		("RMSE_mean", "RMSE_std", "RMSE (lower is better)", "#d62728"),
		("MAE_mean", "MAE_std", "MAE (lower is better)", "#2ca02c"),
	]

	for ax, (mean_col, std_col, title, color) in zip(axes, metric_specs):
		ax.bar(
			x,
			target_df[mean_col].to_numpy(),
			yerr=target_df[std_col].to_numpy(),
			capsize=4,
			color=color,
			alpha=0.85,
		)
		ax.set_ylabel(mean_col)
		ax.set_title(f"{target} - {title}")
		ax.grid(axis="y", linestyle="--", alpha=0.35)

	axes[-1].set_xticks(x)
	axes[-1].set_xticklabels(models, rotation=28, ha="right")
	axes[-1].set_xlabel("Model")

	fig.tight_layout()
	fig.savefig(outdir / f"metrics_bar_{target}.png", dpi=300)
	fig.savefig(outdir / f"metrics_{target}.png", dpi=300)
	plt.close(fig)


def plot_multi_target_metric(summary_df: pd.DataFrame, outdir: Path) -> None:
	targets = summary_df["Target"].unique().tolist()
	model_order = (
		summary_df.groupby("Model", as_index=False)["RMSE_mean"]
		.mean()
		.sort_values("RMSE_mean")
		["Model"]
		.tolist()
	)
	x = np.arange(len(model_order))
	width = 0.8 / max(len(targets), 1)

	fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
	metric_defs = [
		("R2_mean", "R2 across both targets (higher is better)"),
		("RMSE_mean", "RMSE across both targets (lower is better)"),
	]

	for ax, (metric_col, title) in zip(axes, metric_defs):
		for idx, target in enumerate(targets):
			subset = summary_df[summary_df["Target"] == target].set_index("Model")
			values = [subset.loc[m, metric_col] for m in model_order]
			positions = x + (idx - (len(targets) - 1) / 2.0) * width
			ax.bar(positions, values, width=width, label=target, alpha=0.85)

		ax.set_title(title)
		ax.set_ylabel(metric_col)
		ax.grid(axis="y", linestyle="--", alpha=0.35)

	axes[-1].set_xticks(x)
	axes[-1].set_xticklabels(model_order, rotation=28, ha="right")
	axes[-1].set_xlabel("Model")
	axes[0].legend()

	fig.tight_layout()
	fig.savefig(outdir / "metrics_bar_all_targets.png", dpi=300)
	fig.savefig(outdir / "metrics_all_targets.png", dpi=300)
	plt.close(fig)


def main() -> None:
	args = parse_args()
	if args.smoke:
		args.repeats = 5
		if args.outdir == "compare_outputs":
			args.outdir = "compare_outputs_smoke"

	if not (0.0 < args.test_size < 1.0):
		raise ValueError("test-size must be in (0, 1).")
	if args.repeats < 1:
		raise ValueError("repeats must be >= 1.")

	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	warnings.filterwarnings("ignore", category=ConvergenceWarning)

	dataset = pd.read_csv(args.data_path)
	dataset = validate_dataset(dataset)

	print(f"Loaded dataset from {args.data_path} with {len(dataset)} rows.")
	print(f"Features: {FEATURE_COLUMNS}")
	print(f"Targets: {TARGET_COLUMNS}")
	print(
		"Protocol: 80/20 split, repeated random splitting, 5-fold CV tuning on train, "
		"test-set evaluation (RMSE, MAE, R2-paper)."
	)

	model_specs = build_model_specs()
	all_detail_rows: list[dict] = []

	for target in TARGET_COLUMNS:
		detail_rows, pred_df = evaluate_target(
			dataset=dataset,
			target_col=target,
			model_specs=model_specs,
			repeats=args.repeats,
			test_size=args.test_size,
			base_seed=args.base_seed,
			n_jobs=args.n_jobs,
		)
		all_detail_rows.extend(detail_rows)
		pred_df.to_csv(outdir / f"predictions_{target}.csv", index=False)

	detail_df = pd.DataFrame(all_detail_rows)
	detail_df.to_csv(outdir / "metrics_detail_all_repeats.csv", index=False)

	summary_df = build_summary(detail_df)
	summary_df.to_csv(outdir / "metrics_summary_all_targets.csv", index=False)

	compact_all_df = summary_df[["Target", "Model", "R2_mean", "RMSE_mean"]].rename(
		columns={"R2_mean": "R2", "RMSE_mean": "RMSE"}
	)
	compact_all_df.to_csv(outdir / "metrics_all_targets.csv", index=False)

	for target in TARGET_COLUMNS:
		target_summary = summary_df[summary_df["Target"] == target]
		target_summary.to_csv(outdir / f"metrics_summary_{target}.csv", index=False)
		compact_target_df = target_summary[["Target", "Model", "R2_mean", "RMSE_mean"]].rename(
			columns={"R2_mean": "R2", "RMSE_mean": "RMSE"}
		)
		compact_target_df.to_csv(outdir / f"metrics_{target}.csv", index=False)
		plot_target_metrics(summary_df, target, outdir)

	best_df = select_best_model_per_target(summary_df)
	best_df.to_csv(outdir / "best_model_by_target.csv", index=False)

	plot_multi_target_metric(summary_df, outdir)

	print("Saved files:")
	print(f"- {outdir / 'metrics_detail_all_repeats.csv'}")
	print(f"- {outdir / 'metrics_summary_all_targets.csv'}")
	print(f"- {outdir / 'metrics_all_targets.csv'}")
	print(f"- {outdir / 'best_model_by_target.csv'}")
	for target in TARGET_COLUMNS:
		print(f"- {outdir / f'metrics_summary_{target}.csv'}")
		print(f"- {outdir / f'metrics_{target}.csv'}")
		print(f"- {outdir / f'predictions_{target}.csv'}")
		print(f"- {outdir / f'metrics_bar_{target}.png'}")
		print(f"- {outdir / f'metrics_{target}.png'}")
	print(f"- {outdir / 'metrics_bar_all_targets.png'}")
	print(f"- {outdir / 'metrics_all_targets.png'}")


if __name__ == "__main__":
	main()
