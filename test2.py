"""
RF + Bootstrap-EI + NSGA-II multi-objective alloy search.

This script implements the requested process:
1) Build RF bootstrap ensembles to estimate prediction mean/std.
2) Use EI as two objectives (yield strength and fracture strain).
3) Run NSGA-II with requested defaults (pop=500, gen=20, cx=0.8, mut=0.02).
4) Repeat independent optimization runs (default 100), merge fronts,
   and extract a global Pareto front.
5) Respect search constraints in decision decoding:
   - 10 target elements
   - 4/5/6 component alloys
   - 1 at% discretization step
   - each selected element constrained to 5-35 wt% in feasibility checks.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor


matplotlib.use("Agg")

ELEMENTS = ["Al", "Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W"]
TARGET_YIELD = "yield_strength_1000C"
TARGET_DUCTILITY = "fracture_strain_RT"

# Atomic weights used for at% -> wt% conversion.
ATOMIC_WEIGHTS = np.array(
	[26.9815385, 47.867, 50.9415, 51.9961, 91.224, 92.90637, 95.95, 178.49, 180.94788, 183.84],
	dtype=float,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="RF-bootstrap EI + repeated NSGA-II global Pareto search"
	)
	parser.add_argument("--data-path", default="datest.csv", help="Input dataset CSV path.")
	parser.add_argument("--outdir", default="nsga2_rf_outputs", help="Output directory.")

	parser.add_argument("--bootstrap-models", type=int, default=1000, help="Bootstrap model count per target.")
	parser.add_argument("--rf-n-estimators", type=int, default=200, help="Trees in each RF model.")
	parser.add_argument("--rf-max-features", default="sqrt", help="RF max_features (sqrt/log2/float).")
	parser.add_argument("--rf-min-samples-leaf", type=int, default=1, help="RF min_samples_leaf.")

	parser.add_argument("--pop-size", type=int, default=500, help="NSGA-II population size.")
	parser.add_argument("--n-gen", type=int, default=20, help="NSGA-II generation count.")
	parser.add_argument("--cx-prob", type=float, default=0.8, help="SBX crossover probability.")
	parser.add_argument("--mut-prob", type=float, default=0.02, help="Polynomial mutation probability.")
	parser.add_argument("--n-runs", type=int, default=100, help="Independent NSGA-II run count.")

	parser.add_argument("--xi", type=float, default=0.0, help="EI exploration parameter xi.")
	parser.add_argument("--base-seed", type=int, default=42, help="Base random seed.")

	parser.add_argument(
		"--smoke",
		action="store_true",
		help="Quick debug mode: smaller bootstrap/models/runs for validation.",
	)
	return parser.parse_args()


def parse_max_features(value: str):
	lower = value.lower()
	if lower in {"sqrt", "log2", "auto"}:
		return lower
	try:
		return float(value)
	except ValueError:
		return value


def bounded_composition_count(k: int, total: int = 100, lower: int = 5, upper: int = 35) -> int:
	"""Count integer solutions x1..xk with lower<=xi<=upper and sum xi = total."""
	s = total - k * lower
	u = upper - lower
	count = 0
	for j in range(k + 1):
		n = s - j * (u + 1)
		if n < 0:
			continue
		term = comb(k, j) * comb(n + k - 1, k - 1)
		count += term if j % 2 == 0 else -term
	return int(count)


def estimate_search_space_size() -> tuple[int, dict[int, int]]:
	details: dict[int, int] = {}
	total = 0
	for k in (4, 5, 6):
		count_k = comb(len(ELEMENTS), k) * bounded_composition_count(k)
		details[k] = int(count_k)
		total += count_k
	return int(total), details


def load_dataset(data_path: str) -> pd.DataFrame:
	df = pd.read_csv(data_path)
	required_cols = ELEMENTS + [TARGET_YIELD, TARGET_DUCTILITY]
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	for col in required_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")

	df = df.dropna(subset=required_cols).reset_index(drop=True)
	if df.empty:
		raise ValueError("No valid rows left after dropping NaNs.")
	return df


def train_bootstrap_rf_ensemble(
	X: np.ndarray,
	y: np.ndarray,
	n_models: int,
	rf_params: dict,
	seed: int,
	label: str,
) -> list[RandomForestRegressor]:
	rng = np.random.default_rng(seed)
	n_samples = X.shape[0]
	models: list[RandomForestRegressor] = []

	print(f"Training bootstrap RF ensemble for {label}: {n_models} models")
	for idx in range(n_models):
		sample_idx = rng.integers(0, n_samples, size=n_samples)
		model = RandomForestRegressor(random_state=seed + idx, **rf_params)
		model.fit(X[sample_idx], y[sample_idx])
		models.append(model)

		if (idx + 1) % max(1, n_models // 10) == 0 or idx + 1 == n_models:
			print(f"  {label}: trained {idx + 1}/{n_models}")

	return models


def predict_bootstrap_distribution(models: list[RandomForestRegressor], X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	preds = np.empty((len(models), X_query.shape[0]), dtype=float)
	for i, model in enumerate(models):
		preds[i, :] = model.predict(X_query)
	mu = np.mean(preds, axis=0)
	sigma = np.std(preds, axis=0, ddof=1)
	sigma = np.maximum(sigma, 1e-12)
	return mu, sigma


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_observed: float, xi: float = 0.0) -> np.ndarray:
	improvement = mu - best_observed - xi
	z = improvement / sigma
	ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
	return np.where(sigma <= 1e-12, np.maximum(improvement, 0.0), ei)


def allocate_integer_percent(
	weights: np.ndarray,
	total: int = 100,
	lower: int = 5,
	upper: int = 35,
) -> np.ndarray:
	"""Allocate integer percentages with bounds and fixed total."""
	k = int(weights.size)
	if k * lower > total or k * upper < total:
		raise ValueError("Infeasible bounds for allocation.")

	w = np.asarray(weights, dtype=float)
	w = np.maximum(w, 0.0)
	if not np.any(w > 0):
		w = np.ones_like(w)
	w = w / np.sum(w)

	available = total - k * lower
	capacities = np.full(k, upper - lower, dtype=float)
	alloc = np.zeros(k, dtype=float)
	active = np.ones(k, dtype=bool)
	remaining = float(available)

	for _ in range(20):
		if remaining <= 1e-12 or not np.any(active):
			break
		active_idx = np.where(active)[0]
		w_active = w[active_idx]
		w_sum = np.sum(w_active)
		if w_sum <= 1e-12:
			shares = np.full(active_idx.shape[0], remaining / active_idx.shape[0])
		else:
			shares = remaining * w_active / w_sum

		consumed = 0.0
		for local_i, global_i in enumerate(active_idx):
			room = capacities[global_i] - alloc[global_i]
			add = min(room, shares[local_i])
			if add > 0.0:
				alloc[global_i] += add
				consumed += add

		remaining = available - float(np.sum(alloc))
		active = alloc < (capacities - 1e-12)
		if consumed <= 1e-12:
			break

	continuous = lower + alloc
	integer_vals = np.floor(continuous).astype(int)
	frac = continuous - integer_vals

	diff = total - int(np.sum(integer_vals))
	if diff > 0:
		for idx in np.argsort(-frac):
			if diff == 0:
				break
			if integer_vals[idx] < upper:
				integer_vals[idx] += 1
				diff -= 1
		if diff > 0:
			for idx in np.argsort(-w):
				if diff == 0:
					break
				room = upper - integer_vals[idx]
				if room > 0:
					add = min(room, diff)
					integer_vals[idx] += add
					diff -= add
	elif diff < 0:
		need = -diff
		for idx in np.argsort(frac):
			if need == 0:
				break
			if integer_vals[idx] > lower:
				integer_vals[idx] -= 1
				need -= 1
		if need > 0:
			for idx in np.argsort(w):
				if need == 0:
					break
				room = integer_vals[idx] - lower
				if room > 0:
					sub = min(room, need)
					integer_vals[idx] -= sub
					need -= sub

	final_diff = total - int(np.sum(integer_vals))
	if final_diff != 0:
		direction = 1 if final_diff > 0 else -1
		for _ in range(abs(final_diff) * k + 5):
			if final_diff == 0:
				break
			changed = False
			order = np.argsort(-w if direction > 0 else w)
			for idx in order:
				if direction > 0 and integer_vals[idx] < upper:
					integer_vals[idx] += 1
					final_diff -= 1
					changed = True
				elif direction < 0 and integer_vals[idx] > lower:
					integer_vals[idx] -= 1
					final_diff += 1
					changed = True
				if final_diff == 0:
					break
			if not changed:
				break

	if int(np.sum(integer_vals)) != total:
		raise RuntimeError("Failed to satisfy integer sum constraint.")
	return integer_vals


def decode_decision_to_at_int(row: np.ndarray) -> np.ndarray:
	"""
	Decision vector format:
	- row[0] controls number of active elements: 4/5/6.
	- row[1:11] are weights used to choose active elements and allocate at%.
	"""
	k_raw = int(np.floor(float(row[0]) * 3.0))
	k = 4 + max(0, min(2, k_raw))

	weights = np.asarray(row[1:], dtype=float)
	chosen_idx = np.argsort(weights)[::-1][:k]
	chosen_weights = weights[chosen_idx]
	chosen_at = allocate_integer_percent(chosen_weights, total=100, lower=5, upper=35)

	at_int = np.zeros(len(ELEMENTS), dtype=int)
	at_int[chosen_idx] = chosen_at
	return at_int


def decode_population_to_at_int(X: np.ndarray) -> np.ndarray:
	X = np.atleast_2d(X)
	decoded = np.zeros((X.shape[0], len(ELEMENTS)), dtype=int)
	for i in range(X.shape[0]):
		decoded[i, :] = decode_decision_to_at_int(X[i, :])
	return decoded


def at_int_to_fraction(at_int: np.ndarray) -> np.ndarray:
	return at_int.astype(float) / 100.0


def at_fraction_to_wt_percent(at_fraction: np.ndarray) -> np.ndarray:
	mass = at_fraction * ATOMIC_WEIGHTS.reshape(1, -1)
	total_mass = np.sum(mass, axis=1, keepdims=True)
	return np.divide(mass, total_mass, out=np.zeros_like(mass), where=total_mass > 0.0) * 100.0


def evaluate_at_int_points(
	at_int: np.ndarray,
	models_yield: list[RandomForestRegressor],
	models_ductility: list[RandomForestRegressor],
	best_yield: float,
	best_ductility: float,
	xi: float,
) -> dict[str, np.ndarray]:
	at_fraction = at_int_to_fraction(at_int)
	wt_percent = at_fraction_to_wt_percent(at_fraction)

	mu_yield, std_yield = predict_bootstrap_distribution(models_yield, at_fraction)
	mu_ductility, std_ductility = predict_bootstrap_distribution(models_ductility, at_fraction)

	ei_yield = expected_improvement(mu_yield, std_yield, best_yield, xi)
	ei_ductility = expected_improvement(mu_ductility, std_ductility, best_ductility, xi)

	return {
		"at_fraction": at_fraction,
		"wt_percent": wt_percent,
		"mu_yield": mu_yield,
		"std_yield": std_yield,
		"mu_ductility": mu_ductility,
		"std_ductility": std_ductility,
		"ei_yield": ei_yield,
		"ei_ductility": ei_ductility,
	}


class MultiObjectiveEIProblem(Problem):
	def __init__(
		self,
		models_yield: list[RandomForestRegressor],
		models_ductility: list[RandomForestRegressor],
		best_yield: float,
		best_ductility: float,
		xi: float,
	) -> None:
		super().__init__(n_var=11, n_obj=2, n_constr=5, xl=0.0, xu=1.0)
		self.models_yield = models_yield
		self.models_ductility = models_ductility
		self.best_yield = best_yield
		self.best_ductility = best_ductility
		self.xi = xi

	def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
		at_int = decode_population_to_at_int(X)

		# Evaluate only unique discrete compositions for speed, then map back.
		unique_at_int, inverse_idx = np.unique(at_int, axis=0, return_inverse=True)

		eval_data = evaluate_at_int_points(
			at_int=unique_at_int,
			models_yield=self.models_yield,
			models_ductility=self.models_ductility,
			best_yield=self.best_yield,
			best_ductility=self.best_ductility,
			xi=self.xi,
		)

		nonzero_count = np.sum(unique_at_int > 0, axis=1)
		selected_mask = unique_at_int > 0

		# All constraints are expressed as violation values (<=0 feasible in pymoo,
		# so 0 means feasible, positive means violation).
		g_count_low = np.maximum(0.0, 4 - nonzero_count).astype(float)
		g_count_high = np.maximum(0.0, nonzero_count - 6).astype(float)
		g_at_low = np.sum(np.where(selected_mask, np.maximum(0.0, 5 - unique_at_int), 0.0), axis=1)
		g_at_high = np.sum(np.where(selected_mask, np.maximum(0.0, unique_at_int - 35), 0.0), axis=1)

		wt_percent = eval_data["wt_percent"]
		g_wt = np.sum(
			np.where(selected_mask, np.maximum(0.0, 5.0 - wt_percent) + np.maximum(0.0, wt_percent - 35.0), 0.0),
			axis=1,
		)

		f_unique = np.column_stack([-eval_data["ei_yield"], -eval_data["ei_ductility"]])
		g_unique = np.column_stack([g_count_low, g_count_high, g_at_low, g_at_high, g_wt])

		out["F"] = f_unique[inverse_idx]
		out["G"] = g_unique[inverse_idx]


def build_points_dataframe(
	at_int: np.ndarray,
	eval_data: dict[str, np.ndarray],
	run_id: int,
	seed: int,
) -> pd.DataFrame:
	at_fraction = eval_data["at_fraction"]
	wt_percent = eval_data["wt_percent"]

	rows = []
	for i in range(at_int.shape[0]):
		row = {
			"run_id": run_id,
			"seed": seed,
			"n_elements": int(np.sum(at_int[i, :] > 0)),
			"pred_mean_yield_strength_1000C": float(eval_data["mu_yield"][i]),
			"pred_std_yield_strength_1000C": float(eval_data["std_yield"][i]),
			"pred_mean_fracture_strain_RT": float(eval_data["mu_ductility"][i]),
			"pred_std_fracture_strain_RT": float(eval_data["std_ductility"][i]),
			"EI_yield_strength_1000C": float(eval_data["ei_yield"][i]),
			"EI_fracture_strain_RT": float(eval_data["ei_ductility"][i]),
		}
		row["EI_sum"] = row["EI_yield_strength_1000C"] + row["EI_fracture_strain_RT"]

		for j, elem in enumerate(ELEMENTS):
			row[f"{elem}_atpct"] = int(at_int[i, j])
			row[f"{elem}_atfrac"] = float(at_fraction[i, j])
			row[f"{elem}_wtpct"] = float(wt_percent[i, j])

		rows.append(row)

	return pd.DataFrame(rows)


def drop_duplicate_compositions(df: pd.DataFrame) -> pd.DataFrame:
	at_cols = [f"{elem}_atpct" for elem in ELEMENTS]
	dedup = (
		df.sort_values(["EI_sum", "EI_yield_strength_1000C", "EI_fracture_strain_RT"], ascending=False)
		.drop_duplicates(subset=at_cols, keep="first")
		.reset_index(drop=True)
	)
	return dedup


def get_global_pareto_front(df_unique: pd.DataFrame) -> pd.DataFrame:
	f = np.column_stack(
		[
			-df_unique["EI_yield_strength_1000C"].to_numpy(dtype=float),
			-df_unique["EI_fracture_strain_RT"].to_numpy(dtype=float),
		]
	)
	nd_idx = NonDominatedSorting().do(f, only_non_dominated_front=True)
	pareto_df = df_unique.iloc[nd_idx].copy().reset_index(drop=True)
	pareto_df = pareto_df.sort_values(
		["EI_sum", "EI_yield_strength_1000C", "EI_fracture_strain_RT"],
		ascending=False,
	).reset_index(drop=True)
	return pareto_df


def run_single_nsga2(
	run_id: int,
	args: argparse.Namespace,
	models_yield: list[RandomForestRegressor],
	models_ductility: list[RandomForestRegressor],
	best_yield: float,
	best_ductility: float,
) -> pd.DataFrame:
	seed = args.base_seed + run_id - 1

	problem = MultiObjectiveEIProblem(
		models_yield=models_yield,
		models_ductility=models_ductility,
		best_yield=best_yield,
		best_ductility=best_ductility,
		xi=args.xi,
	)

	algorithm = NSGA2(
		pop_size=args.pop_size,
		crossover=SBX(prob=args.cx_prob, eta=15),
		mutation=PM(prob=args.mut_prob, eta=20),
		eliminate_duplicates=True,
	)

	res = minimize(problem, algorithm, ("n_gen", args.n_gen), seed=seed, verbose=False)

	if res.X is None:
		if res.pop is None:
			return pd.DataFrame()

		pop_x = res.pop.get("X")
		if pop_x is None or len(pop_x) == 0:
			return pd.DataFrame()

		at_int = decode_population_to_at_int(pop_x)
	else:
		at_int = decode_population_to_at_int(np.atleast_2d(res.X))

	eval_data = evaluate_at_int_points(
		at_int=at_int,
		models_yield=models_yield,
		models_ductility=models_ductility,
		best_yield=best_yield,
		best_ductility=best_ductility,
		xi=args.xi,
	)

	run_df = build_points_dataframe(at_int=at_int, eval_data=eval_data, run_id=run_id, seed=seed)
	run_df = drop_duplicate_compositions(run_df)
	return run_df


def plot_pareto_outputs(df_unique: pd.DataFrame, df_global: pd.DataFrame, outdir: Path) -> None:
	plt.figure(figsize=(8, 6))
	plt.scatter(
		df_unique["EI_yield_strength_1000C"],
		df_unique["EI_fracture_strain_RT"],
		s=12,
		alpha=0.25,
		label="Merged points",
	)
	plt.scatter(
		df_global["EI_yield_strength_1000C"],
		df_global["EI_fracture_strain_RT"],
		s=30,
		alpha=0.9,
		label="Global Pareto front",
	)
	plt.xlabel("EI (yield_strength_1000C)")
	plt.ylabel("EI (fracture_strain_RT)")
	plt.title("EI space: merged points vs global Pareto front")
	plt.grid(alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(outdir / "global_pareto_ei.png", dpi=300)
	plt.close()

	plt.figure(figsize=(8, 6))
	plt.scatter(
		df_unique["pred_mean_yield_strength_1000C"],
		df_unique["pred_mean_fracture_strain_RT"],
		s=12,
		alpha=0.25,
		label="Merged points",
	)
	plt.scatter(
		df_global["pred_mean_yield_strength_1000C"],
		df_global["pred_mean_fracture_strain_RT"],
		s=30,
		alpha=0.9,
		label="Global Pareto front",
	)
	plt.xlabel("Predicted mean yield_strength_1000C")
	plt.ylabel("Predicted mean fracture_strain_RT")
	plt.title("Predicted performance space: merged points vs global Pareto front")
	plt.grid(alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(outdir / "global_pareto_predicted_means.png", dpi=300)
	plt.close()


def main() -> None:
	args = parse_args()

	if args.smoke:
		if args.outdir == "nsga2_rf_outputs":
			args.outdir = "nsga2_rf_outputs_smoke"
		args.bootstrap_models = min(args.bootstrap_models, 30)
		args.rf_n_estimators = min(args.rf_n_estimators, 80)
		args.pop_size = min(args.pop_size, 60)
		args.n_gen = min(args.n_gen, 5)
		args.n_runs = min(args.n_runs, 3)

	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	df = load_dataset(args.data_path)
	X = df[ELEMENTS].to_numpy(dtype=float)
	y_yield = df[TARGET_YIELD].to_numpy(dtype=float)
	y_ductility = df[TARGET_DUCTILITY].to_numpy(dtype=float)

	rf_params = {
		"n_estimators": args.rf_n_estimators,
		"max_features": parse_max_features(args.rf_max_features),
		"min_samples_leaf": args.rf_min_samples_leaf,
		"bootstrap": True,
		"n_jobs": 1,
	}

	search_size, search_breakdown = estimate_search_space_size()
	print(f"Approximate combinatorial search size (4/5/6 components, 1 at%, 5-35 at%): {search_size:,}")
	print(f"Breakdown by active element count: {search_breakdown}")

	best_yield = float(np.max(y_yield))
	best_ductility = float(np.max(y_ductility))
	print(f"Best observed in dataset: {TARGET_YIELD}={best_yield:.4f}, {TARGET_DUCTILITY}={best_ductility:.4f}")

	models_yield = train_bootstrap_rf_ensemble(
		X=X,
		y=y_yield,
		n_models=args.bootstrap_models,
		rf_params=rf_params,
		seed=args.base_seed,
		label=TARGET_YIELD,
	)
	models_ductility = train_bootstrap_rf_ensemble(
		X=X,
		y=y_ductility,
		n_models=args.bootstrap_models,
		rf_params=rf_params,
		seed=args.base_seed + 100000,
		label=TARGET_DUCTILITY,
	)

	all_run_fronts: list[pd.DataFrame] = []
	run_summaries: list[dict] = []

	for run_id in range(1, args.n_runs + 1):
		print(f"Running NSGA-II optimization {run_id}/{args.n_runs}")
		run_df = run_single_nsga2(
			run_id=run_id,
			args=args,
			models_yield=models_yield,
			models_ductility=models_ductility,
			best_yield=best_yield,
			best_ductility=best_ductility,
		)

		if run_df.empty:
			run_summaries.append(
				{
					"run_id": run_id,
					"seed": args.base_seed + run_id - 1,
					"n_points": 0,
					"max_EI_yield": float("nan"),
					"max_EI_ductility": float("nan"),
					"max_EI_sum": float("nan"),
				}
			)
			print(f"  run {run_id}: no points returned")
			continue

		run_df.to_csv(outdir / f"pareto_run_{run_id:03d}.csv", index=False)
		all_run_fronts.append(run_df)
		run_summaries.append(
			{
				"run_id": run_id,
				"seed": args.base_seed + run_id - 1,
				"n_points": int(len(run_df)),
				"max_EI_yield": float(run_df["EI_yield_strength_1000C"].max()),
				"max_EI_ductility": float(run_df["EI_fracture_strain_RT"].max()),
				"max_EI_sum": float(run_df["EI_sum"].max()),
			}
		)
		print(f"  run {run_id}: {len(run_df)} Pareto points saved")

	run_summary_df = pd.DataFrame(run_summaries)
	run_summary_df.to_csv(outdir / "run_summary.csv", index=False)

	if not all_run_fronts:
		print("No Pareto points were produced. Please relax constraints or adjust optimization settings.")
		return

	all_points_df = pd.concat(all_run_fronts, ignore_index=True)
	all_points_df.to_csv(outdir / "pareto_points_all_runs.csv", index=False)

	unique_points_df = drop_duplicate_compositions(all_points_df)
	unique_points_df.to_csv(outdir / "pareto_points_unique_merged.csv", index=False)

	global_pareto_df = get_global_pareto_front(unique_points_df)
	global_pareto_df.to_csv(outdir / "global_pareto_front.csv", index=False)

	top_k = min(100, len(global_pareto_df))
	global_pareto_df.head(top_k).to_csv(outdir / "global_pareto_top100.csv", index=False)

	plot_pareto_outputs(unique_points_df, global_pareto_df, outdir)

	config_info = {
		"data_path": args.data_path,
		"outdir": str(outdir),
		"elements": ELEMENTS,
		"targets": [TARGET_YIELD, TARGET_DUCTILITY],
		"search_space_constraints": {
			"active_elements": [4, 5, 6],
			"at_percent_step": 1,
			"selected_element_at_percent_bounds": [5, 35],
			"selected_element_wt_percent_bounds": [5, 35],
		},
		"search_space_combinatorial_estimate": {
			"total": search_size,
			"breakdown": search_breakdown,
		},
		"rf_params": rf_params,
		"bootstrap_models_per_target": args.bootstrap_models,
		"ei": {
			"best_yield": best_yield,
			"best_ductility": best_ductility,
			"xi": args.xi,
			"objective": "maximize EI for both targets",
		},
		"nsga2": {
			"pop_size": args.pop_size,
			"n_gen": args.n_gen,
			"crossover_probability": args.cx_prob,
			"mutation_probability": args.mut_prob,
			"n_runs": args.n_runs,
			"seed_base": args.base_seed,
		},
	}

	with open(outdir / "experiment_config.json", "w", encoding="utf-8") as f:
		json.dump(config_info, f, ensure_ascii=False, indent=2)

	print("Completed RF-EI NSGA-II multi-run search.")
	print(f"Saved: {outdir / 'pareto_points_all_runs.csv'}")
	print(f"Saved: {outdir / 'pareto_points_unique_merged.csv'}")
	print(f"Saved: {outdir / 'global_pareto_front.csv'}")
	print(f"Saved: {outdir / 'global_pareto_top100.csv'}")
	print(f"Saved: {outdir / 'run_summary.csv'}")
	print(f"Saved: {outdir / 'global_pareto_ei.png'}")
	print(f"Saved: {outdir / 'global_pareto_predicted_means.png'}")
	print(f"Saved: {outdir / 'experiment_config.json'}")


if __name__ == "__main__":
	main()

