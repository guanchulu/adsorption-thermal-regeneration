# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "polars==2.0.0", "scipy==1.18.1"]
# ///
# Run: uv run --project .. --no-sync --offline python code/ensemble_probe.py
"""Exploratory fixed-temperature convexification; numerical evidence, not a global certificate."""

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray
from path_model import co2_cp
from scipy.optimize import brentq, fminbound
from scipy.special import expit, logit
from thermal_core import R

ROOT: Final = Path(__file__).resolve().parents[1]
Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Frontier:
    temperature: float
    cluster: int
    restricted: bool
    quadrature_t: Vector
    quadrature_w: Vector

    def admissible(self, energy: float, entropy: float, tolerance: float = 0.0) -> bool:
        return energy >= -tolerance and (
            not self.restricted
            or (energy <= 100 + tolerance and 0.04 - tolerance / 100 <= entropy <= 0.14 + tolerance / 100)
        )

    def state(self, release: float, position: float) -> tuple[float, float, float]:
        residual = (1 - release) * expit(position)
        loaded = residual + release
        factor = (1 / 298.0 - 1 / self.temperature) / R
        energy = ((logit(loaded) - logit(residual)) / self.cluster - np.log(1.5)) / factor
        entropy = R * np.log(0.15) + energy / 298.0 - R / self.cluster * logit(loaded)
        theta = expit(self.cluster * (np.log(0.1) + energy / (R * self.quadrature_t) - entropy / R))
        cost = energy * release + self.quadrature_w @ theta
        return float(cost), float(energy), float(entropy)

    def interval_edges(self, release: float) -> list[float]:
        limit = min(22.0, float(np.log((1 - release) / 2e-14)))
        if limit <= 0:
            return []
        # Entropy has one stationary point; splitting there resolves narrow feasible intervals.
        dt = self.temperature - 298.0
        coefficient = -dt + 2 * self.temperature * release
        discriminant = np.sqrt(coefficient**2 + 4 * dt * self.temperature * release * (1 - release))
        v_min = 2 * self.temperature * release * (1 - release) / (coefficient + discriminant)
        turning = float(logit(np.clip(v_min / (1 - release), 1e-12, 1 - 1e-12)))
        nodes = sorted({-limit, 0.0, limit, float(np.clip(turning, -limit, limit))})
        edges = nodes.copy()
        conditions = [(1, 0.0)]
        if self.restricted:
            conditions += [(1, 100.0), (2, 0.04), (2, 0.14)]
        for index, boundary in conditions:

            def difference(value: float, index: int = index, boundary: float = boundary) -> float:
                return self.state(release, value)[index] - boundary

            for left, right in itertools.pairwise(nodes):
                if difference(left) * difference(right) < 0:
                    root, _details = brentq(difference, left, right, xtol=1e-12, full_output=True)
                    edges.append(float(root))
        return sorted(set(edges))

    def minimum(self, release: float) -> tuple[float, float, float]:
        edges = self.interval_edges(release)
        best = (float("inf"), float("nan"), float("nan"))
        for left, right in itertools.pairwise(edges):
            _, energy, entropy = self.state(release, (left + right) / 2)
            if not self.admissible(energy, entropy):
                continue
            grid = np.linspace(left, right, 17)
            costs = np.array([self.state(release, float(x))[0] for x in grid])
            candidates = [float(grid[0]), float(grid[-1])]

            def endpoint_cost(value: float) -> float:
                return self.state(release, value)[0]

            for i in range(len(grid)):
                if i in (0, len(grid) - 1) or costs[i] <= min(costs[i - 1], costs[i + 1]):
                    location, _cost, _flag, _calls = fminbound(
                        endpoint_cost,
                        grid[max(0, i - 1)],
                        grid[min(len(grid) - 1, i + 1)],
                        xtol=1e-11,
                        full_output=True,
                    )
                    candidates.append(float(location))
            for candidate in candidates:
                cost, energy, entropy = self.state(release, candidate)
                if self.admissible(energy, entropy, 1e-8) and cost < best[0]:
                    best = cost, energy, entropy
        return best


def lower_hull(releases: Vector, costs: Vector) -> list[int]:
    hull: list[int] = []
    for i in range(len(releases)):
        while len(hull) >= 2:
            a, b = hull[-2:]
            first = (costs[b] - costs[a]) / (releases[b] - releases[a])
            second = (costs[i] - costs[b]) / (releases[i] - releases[b])
            if first <= second + 1e-10:
                break
            hull.pop()
        hull.append(i)
    return hull


def budget_case(
    frontier: Frontier, arrays: tuple[Vector, Vector, list[int]], budget: float
) -> tuple[float, float, float, float]:
    delta, cost, hull = arrays
    feasible = np.flatnonzero(cost <= budget)
    if not len(feasible):
        return float("nan"), float("nan"), float("nan"), float("nan")
    index = int(feasible[-1])
    pure = float(delta[index])
    if index + 1 < len(delta):

        def residual(value: float) -> float:
            return frontier.minimum(value)[0] - budget

        root, _details = brentq(
            residual, float(delta[index]), float(delta[index + 1]), xtol=1e-10, full_output=True
        )
        pure = float(root)
    mixture = pure
    left_delta = right_delta = pure
    for a, b in itertools.pairwise(hull):
        if cost[a] <= budget < cost[b]:
            fraction = (budget - cost[a]) / (cost[b] - cost[a])
            value = float(delta[a] + fraction * (delta[b] - delta[a]))
            if value > mixture:
                mixture, left_delta, right_delta = value, float(delta[a]), float(delta[b])
    return pure, mixture, left_delta, right_delta


def main() -> None:
    temperatures = np.r_[298.01, 300.0, np.arange(303.0, 424.0, 5.0)]
    releases = np.unique(
        np.r_[np.geomspace(1e-6, 0.04, 20), np.linspace(0.04, 0.99, 100), 1 - np.geomspace(1e-9, 0.01, 20)]
    )
    nodes, weights = np.polynomial.legendre.leggauss(96)
    curves, comparisons = [], []
    for restricted in (False, True):
        for cluster in (1, 4):
            for temperature in temperatures:
                half = (temperature - 298.0) / 2
                quadrature_t = half * nodes + (temperature + 298.0) / 2
                frontier = Frontier(
                    float(temperature),
                    cluster,
                    restricted,
                    quadrature_t,
                    half * weights * co2_cp(quadrature_t),
                )
                values = [frontier.minimum(float(d)) for d in releases]
                valid = np.isfinite([v[0] for v in values])
                delta = releases[valid]
                cost = np.array([v[0] for v in values])[valid]
                hull = lower_hull(delta, cost)
                curves.extend(
                    {
                        "restricted": restricted,
                        "m": cluster,
                        "Td": float(temperature),
                        "release": float(releases[i]),
                        "variable_heat_per_capacity": values[i][0],
                        "E": values[i][1],
                        "s": values[i][2],
                    }
                    for i in np.flatnonzero(valid)
                )
                for capacity in (2.0, 4.0, 8.0):
                    for dry_cp in (0.5, 1.0, 2.0):
                        for budget in (50.0, 100.0, 200.0):
                            available = (budget - dry_cp * (temperature - 298.0)) / capacity
                            if available <= 0 or len(delta) < 2:
                                continue
                            pure, mixture, left, right = budget_case(
                                frontier, (delta, cost, hull), float(available)
                            )
                            comparisons.append(
                                {
                                    "restricted": restricted,
                                    "m": cluster,
                                    "Td": float(temperature),
                                    "Q": capacity,
                                    "C": dry_cp,
                                    "B": budget,
                                    "available_heat_per_capacity": available,
                                    "continuous_pure_release": pure,
                                    "candidate_ensemble_release": mixture,
                                    "ensemble_gain": mixture - pure,
                                    "support_left": left,
                                    "support_right": right,
                                }
                            )
            print("Completed family", restricted, "cluster", cluster, flush=True)
    curve_table = pl.DataFrame(curves)
    comparison = pl.DataFrame(comparisons)
    curve_table.write_csv(ROOT / "results/ensemble_probe_frontiers.csv")
    comparison.write_csv(ROOT / "results/ensemble_probe_comparisons.csv")
    summary = comparison.group_by("restricted", "m").agg(
        pl.len().alias("rows"),
        (pl.col("ensemble_gain") > 1e-6).sum().alias("positive_gains"),
        pl.col("ensemble_gain").max().alias("max_gain"),
    )
    summary.write_csv(ROOT / "results/ensemble_probe_summary.csv")
    provenance = {
        "status": "exploratory",
        "temperatures": temperatures.tolist(),
        "release_nodes": releases.tolist(),
        "endpoint_search": "logit(v/(1-d)) bound min(22,log((1-d)/2e-14)); E>=0; exact E/s feasibility roots; 17-point local-minimum brackets",
        "restricted_family": "0<=E<=100 kJ/mol,0.04<=s<=0.14 kJ/mol/K",
        "free_family": "E>=0, free entropy; finite numerical endpoint search",
        "baseline": "same-Td continuous homogeneous frontier root, not best grid atom",
        "ensemble": "lower convex hull of feasible frontier witnesses; two capacity-weighted atoms at one shared temperature",
        "claim": "No global continuous path optimum certificate. Positive gain requires refinement; no-gain observations are empirical only.",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summary.to_dicts(),
    }
    (ROOT / "results/ensemble_probe_manifest.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(summary.write_csv())


if __name__ == "__main__":
    main()
