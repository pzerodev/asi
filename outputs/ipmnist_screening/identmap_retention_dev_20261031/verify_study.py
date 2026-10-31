"""Re-derive the fixed study's bindings and summary without rerunning learning."""

import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

from alberta_framework.benchmarks import ipmnist_screening as screening
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, load_mnist_train

ROOT = Path.cwd().resolve()
RUN = Path(__file__).resolve().parent
ARMS = (
    "rls_head_resid_l1_preset005",
    "rls_head_resid_identmap50_r",
    "rls_head_resid_identmap50_r_idet003_s005625",
)
SEEDS = (91010, 91011, 91012)
COMMIT = "0d7f8810199aa191edc7170bbf761518eb653d53"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def close(actual, expected):
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-14), "derived metric differs")


def spread(values):
    return {
        "per_seed": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "standard_error": statistics.stdev(values) / math.sqrt(len(values)),
    }


def main():
    def no_network(event, args):
        if event == "socket.connect":
            raise RuntimeError("verification requires the existing public dataset cache")

    sys.addaudithook(no_network)
    # The execution commit is retained. Documentation/output additions may follow
    # it, but every package and lock byte must still match the measured source.
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            COMMIT,
            "--",
            "alberta_framework",
            "pyproject.toml",
            "uv.lock",
        ],
        check=True,
    )
    x, y = load_mnist_train(ROOT / "outputs/upgd_ipmnist/reconstructed_openml_cache")
    dataset = screening._screening_dataset_provenance(x, y)
    random = np.random.default_rng(91000)
    a, b = (random.permutation(784).astype(np.int32) for _ in range(2))
    indices = np.random.default_rng(91001).choice(60000, size=512, replace=False)
    perms = tuple(screening.ipmnist_permutation_sha256(v) for v in (a, b, a))
    sentinel_hashes = tuple(screening.ipmnist_sentinel_set_sha256(x, y, v, indices) for v in (a, b))
    index_hash = screening._array_bundle_sha256(
        "alberta.ipmnist-screening.sentinel-index-order.v1", {"sentinel_indices": indices}
    )
    driver_hash = hashlib.sha256((RUN / "run_shard.py").read_bytes()).hexdigest()
    rows = {}
    inputs = []
    first = None
    for seed in SEEDS:
        orders = screening.build_recurring_ipmnist_online_indices(
            seed=seed, n_examples=60000, phase_lengths=(5000, 5000, 5000), sentinel_indices=indices
        )
        require(np.array_equal(orders[0], orders[2]), "A orders must repeat exactly")
        require(
            all(not np.isin(order, indices).any() for order in orders),
            "sentinels leaked into training",
        )
        order_hashes = tuple(
            screening._array_bundle_sha256(
                "alberta.ipmnist-screening.online-example-order.v1", {"example_indices": order}
            )
            for order in orders
        )
        for arm in ARMS:
            path = RUN / "shards" / f"{arm}_seed{seed}.json"
            raw = path.read_bytes()
            payload = json.loads(raw)
            require(payload["arm"] == arm and payload["seed"] == seed, "arm/seed mismatch")
            require(payload["source_provenance"]["git_commit"] == COMMIT, "commit mismatch")
            require(payload["source_provenance"]["worktree_clean"] is True, "dirty execution")
            require(payload["dataset_provenance"] == dataset, "dataset mismatch")
            require(payload["driver_sha256"] == driver_hash, "driver mismatch")
            if first is None:
                first = payload
            for key in ("source_provenance", "environment", "dataset_provenance"):
                require(payload[key] == first[key], "mixed execution binding: " + key)
            spec = screening.screening_spec(arm)
            expected_id = screening._recurring_protocol_id(
                spec=spec,
                seed=seed,
                config=IPMNISTConfig(n_tasks=3, task_length=5000),
                phase_lengths=(5000, 5000, 5000),
                permutation_sha256=perms,
                sentinel_indices_sha256=index_hash,
                online_indices_sha256=order_hashes,
                relearning_window=250,
            )
            report = payload["report"]
            for flag in (
                "scientific_promotion_allowed",
                "performance_thresholds_applied",
                "retention_claimed",
                "catastrophic_forgetting_absence_claimed",
            ):
                require(report[flag] is False, "development policy changed")
            protocol = report["protocol"]
            require(protocol["protocol_id"] == expected_id, "protocol commitment mismatch")
            require(
                report["protocol_sha256"] == hashlib.sha256(canonical(protocol)).hexdigest(),
                "protocol hash mismatch",
            )
            for index, binding in enumerate(protocol["sentinel_bindings"]):
                require(binding["permutation_sha256"] == perms[index], "permutation mismatch")
                require(
                    binding["sentinel_set_sha256"] == sentinel_hashes[index], "sentinel mismatch"
                )
                require(binding["sentinel_case_count"] == 512, "sentinel count mismatch")
            scores = report["sentinel_scores"]
            require(len(scores) == 5, "expected five probes")
            for score, need in zip(scores, protocol["required_probe_snapshots"], strict=True):
                for key in (
                    "phase_index",
                    "checkpoint_step",
                    "permutation_id",
                    "exposure_index",
                    "sentinel_set_id",
                    "sentinel_case_count",
                ):
                    require(score[key] == need[key], "probe binding mismatch: " + key)
                require(
                    type(score["correct_count"]) is int and 0 <= score["correct_count"] <= 512,
                    "invalid sentinel count",
                )
                close(score["accuracy"], score["correct_count"] / 512)
            for phase in (1, 2):
                hashes = {
                    score["learner_state_sha256"]
                    for score in scores
                    if score["phase_index"] == phase
                }
                require(len(hashes) == 1, "frozen probe checkpoint changed")
            for phase in report["phase_summaries"]:
                require(phase["observation_count"] == 5000, "phase length changed")
                close(
                    phase["mean_pre_update_online_accuracy"],
                    (5000 - phase["online_mistakes"]) / 5000,
                )
            by = {(v["phase_index"], v["permutation_id"]): v["accuracy"] for v in scores}
            aid = "ipmnist-permutation-" + perms[0] + ".v1"
            acquisition, before, revisit = (by[(i, aid)] for i in (0, 1, 2))
            rec = report["recurrence"]
            for key, value in {
                "acquisition_end_sentinel_accuracy": acquisition,
                "pre_revisit_sentinel_accuracy": before,
                "revisit_end_sentinel_accuracy": revisit,
                "retention_change_from_acquisition": before - acquisition,
                "peak_before_revisit_sentinel_accuracy": max(acquisition, before),
                "peak_to_revisit_forgetting": max(0, acquisition - before),
                "revisit_recovery": revisit - before,
                "first_exposure_leading_pre_update_accuracy": (
                    250 - rec["first_exposure_leading_mistakes"]
                )
                / 250,
                "revisit_leading_pre_update_accuracy": (250 - rec["revisit_leading_mistakes"])
                / 250,
                "relearning_savings_mistakes": rec["first_exposure_leading_mistakes"]
                - rec["revisit_leading_mistakes"],
                "relearning_savings_accuracy": rec["revisit_leading_pre_update_accuracy"]
                - rec["first_exposure_leading_pre_update_accuracy"],
            }.items():
                close(rec[key], value)
            rows[arm, seed] = rec
            inputs.append(
                {"path": str(path.relative_to(RUN)), "sha256": hashlib.sha256(raw).hexdigest()}
            )
    metrics = (
        "acquisition_end_sentinel_accuracy",
        "pre_revisit_sentinel_accuracy",
        "revisit_end_sentinel_accuracy",
        "retention_change_from_acquisition",
        "first_exposure_leading_pre_update_accuracy",
        "revisit_leading_pre_update_accuracy",
        "relearning_savings_accuracy",
    )
    arms = {
        arm: {key: spread([rows[arm, seed][key] for seed in SEEDS]) for key in metrics}
        for arm in ARMS
    }
    diffs = [
        rows[ARMS[1], seed]["pre_revisit_sentinel_accuracy"]
        - rows[ARMS[0], seed]["pre_revisit_sentinel_accuracy"]
        for seed in SEEDS
    ]
    paired = spread(diffs)
    agreed = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    directional = (
        abs(paired["mean"]) >= 0.02
        and agreed
        and abs(paired["mean"]) > 2 * paired["standard_error"]
    )
    summary = {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "execution_commit": COMMIT,
        "seeds": list(SEEDS),
        "n": 3,
        "inputs": inputs,
        "arms": arms,
        "paired_primary": paired,
        "declared_absolute_difference_bar": 0.02,
        "study_decision": (
            "lower_direct_frozen_retention"
            if paired["mean"] < 0
            else "higher_direct_frozen_retention"
        )
        if directional
        else "inconclusive",
        "limits": [
            "n=3 development seeds, one A/B/A protocol",
            "the checkpoint's current remap is frozen for every sentinel",
            "this does not distinguish erased weights from inaccessible mappings",
            "trace hashes bind the producer output but do not attest execution",
            "no retained-knowledge, anti-forgetting, promotion or SOTA claim",
        ],
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = RUN / "SUMMARY.json"
    if output.exists():
        require(output.read_text() == encoded, "retained summary does not reproduce")
    else:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(
        json.dumps(
            {
                "validated_shards": len(inputs),
                "decision": summary["study_decision"],
                "paired_primary": paired,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
