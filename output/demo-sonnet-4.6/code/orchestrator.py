"""Orchestrator — manages pipeline execution with remediation loop."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def run_step(script_path, output_dir, run_id, extra_args=None):
    """Run a step script as subprocess. Returns (returncode, stdout, stderr)."""
    python = sys.executable
    cmd = [python, script_path, "--output-dir", output_dir, "--run-id", run_id]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def load_progress(output_dir):
    path = os.path.join(output_dir, "progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def update_progress(output_dir, updates):
    path = os.path.join(output_dir, "progress.json")
    with open(path) as f:
        progress = json.load(f)
    progress.update(updates)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def step_is_complete(output_dir, step_name, json_file):
    progress = load_progress(output_dir)
    completed = progress.get("completed_steps", [])
    json_path = os.path.join(output_dir, json_file)
    return step_name in completed and os.path.exists(json_path)


def delete_step_outputs(output_dir, step_json_files):
    """Delete step output JSONs to force re-run."""
    for f in step_json_files:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Deleted {f} for re-run")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--split-mode", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclude-features", default="")
    parser.add_argument("--max-lag", default="12")
    parser.add_argument("--force-expansion-models", default="false")
    parser.add_argument("--regularization", default="")
    parser.add_argument("--extra-models", default="")
    parser.add_argument("--seasonal-features", default="false")
    args = parser.parse_args()

    output_dir = args.output_dir
    run_id = args.run_id
    os.makedirs(output_dir, exist_ok=True)

    code_dir = os.path.join(output_dir, "code")

    def step_script(name):
        return os.path.join(code_dir, name)

    # Initialize progress.json
    progress_path = os.path.join(output_dir, "progress.json")
    if not os.path.exists(progress_path):
        with open(progress_path, "w") as f:
            json.dump({
                "run_id": run_id,
                "csv_path": args.csv_path,
                "target_column": args.target_column,
                "status": "running",
                "current_step": "10-csv-read-cleansing",
                "completed_steps": [],
                "errors": [],
            }, f, indent=2)

    def run_with_retry(step_name, script, extra_args=None, max_attempts=3):
        for attempt in range(1, max_attempts + 1):
            print(f"\n{'='*60}")
            print(f"Running {step_name} (attempt {attempt}/{max_attempts})")
            print(f"Script: {script}")
            if extra_args:
                print(f"Extra args: {extra_args}")
            rc, stdout, stderr = run_step(script, output_dir, run_id, extra_args)
            if stdout:
                print(stdout)
            if stderr:
                print(f"STDERR: {stderr}")
            if rc == 0:
                print(f"{step_name} completed successfully")
                return 0
            elif rc == 2:
                print(f"LEAKAGE DETECTED in {step_name} (exit code 2)")
                return 2
            else:
                print(f"{step_name} failed (exit code {rc}), attempt {attempt}")
                if attempt == max_attempts:
                    print(f"FATAL: {step_name} failed after {max_attempts} attempts")
                    update_progress(output_dir, {
                        "status": "error",
                        "errors": [f"{step_name} failed: {stderr[-500:] if stderr else 'unknown error'}"]
                    })
                    return rc
        return -1

    # ─── Step 10 ─────────────────────────────────────────────────────────────
    if not step_is_complete(output_dir, "10-csv-read-cleansing", "step-10-cleanse.json"):
        rc = run_with_retry("step-10", step_script("step_10_cleanse.py"),
                            ["--csv-path", args.csv_path, "--target-column", args.target_column])
        if rc != 0:
            sys.exit(1)
    else:
        print("Step 10 already complete, skipping.")

    # ─── Step 11 ─────────────────────────────────────────────────────────────
    if not step_is_complete(output_dir, "11-data-exploration", "step-11-exploration.json"):
        rc = run_with_retry("step-11", step_script("step_11_exploration.py"))
        if rc != 0:
            sys.exit(1)
    else:
        print("Step 11 already complete, skipping.")

    # ─── Steps 12–17 with Remediation Loop ───────────────────────────────────
    MAX_REMEDIATION = 3
    remediation_iter = 0
    exclude_features_param = args.exclude_features
    split_mode_param = args.split_mode
    max_lag_param = args.max_lag
    force_expansion_param = args.force_expansion_models
    regularization_param = args.regularization
    extra_models_param = args.extra_models
    seasonal_features_param = args.seasonal_features

    while True:
        # Step 12
        s12_complete = step_is_complete(output_dir, "12-feature-extraction", "step-12-features.json")
        if not s12_complete:
            extra12 = ["--split-mode", split_mode_param]
            if exclude_features_param:
                extra12 += ["--exclude-features", exclude_features_param]
            if max_lag_param:
                extra12 += ["--max-lag", max_lag_param]
            if seasonal_features_param:
                extra12 += ["--seasonal-features", seasonal_features_param]
            rc = run_with_retry("step-12", step_script("step_12_features.py"), extra12)
            if rc == 2:
                print("LEAKAGE DETECTED — cannot proceed")
                sys.exit(2)
            if rc != 0:
                sys.exit(1)
        else:
            print("Step 12 already complete, skipping.")

        # Step 13
        s13_complete = step_is_complete(output_dir, "13-model-training", "step-13-training.json")
        if not s13_complete:
            extra13 = ["--force-expansion-models", force_expansion_param]
            if regularization_param:
                extra13 += ["--regularization", regularization_param]
            if extra_models_param:
                extra13 += ["--extra-models", extra_models_param]
            rc = run_with_retry("step-13", step_script("step_13_training.py"), extra13)
            if rc != 0:
                sys.exit(1)
        else:
            print("Step 13 already complete, skipping.")

        # Step 14
        s14_complete = step_is_complete(output_dir, "14-model-evaluation", "step-14-evaluation.json")
        if not s14_complete:
            rc = run_with_retry("step-14", step_script("step_14_evaluation.py"))
            if rc != 0:
                sys.exit(1)
        else:
            print("Step 14 already complete, skipping.")

        # Step 15
        s15_complete = step_is_complete(output_dir, "15-model-selection", "step-15-selection.json")
        if not s15_complete:
            rc = run_with_retry("step-15", step_script("step_15_selection.py"))
            if rc != 0:
                sys.exit(1)
        else:
            print("Step 15 already complete, skipping.")

        # Step 16
        s16_complete = step_is_complete(output_dir, "16-result-presentation", "step-16-report.md")
        if not s16_complete:
            rc = run_with_retry("step-16", step_script("step_16_report.py"))
            if rc != 0:
                sys.exit(1)
        else:
            print("Step 16 already complete, skipping.")

        # Step 17
        s17_complete = step_is_complete(output_dir, "17-critical-self-audit", "step-17-audit.json")
        if not s17_complete:
            rc = run_with_retry("step-17", step_script("step_17_audit.py"))
            if rc != 0:
                sys.exit(1)
        else:
            print("Step 17 already complete, skipping.")

        # Check audit result
        audit_path = os.path.join(output_dir, "step-17-audit.json")
        with open(audit_path) as f:
            audit = json.load(f)

        overall = audit.get("overall_audit_result", "fail")
        print(f"\nAudit result: {overall} (remediation iteration {remediation_iter})")

        if overall == "pass":
            # Audit passed — proceed to Step 18
            update_progress(output_dir, {"final_audit_result": "pass"})
            break
        elif remediation_iter >= MAX_REMEDIATION:
            print(f"Max remediation iterations ({MAX_REMEDIATION}) reached")
            # Write remediation_required.json
            rr = {
                "run_id": run_id,
                "original_audit_result": "fail",
                "remediation_iterations_attempted": remediation_iter,
                "final_audit_result": "fail",
                "pending_manual_actions": [
                    a for a in audit.get("remediation_actions", [])
                    if not a.get("auto_executable", False)
                ],
                "instructions": "Manual intervention required. Review pending_manual_actions and re-run the pipeline with the corrected parameters."
            }
            with open(os.path.join(output_dir, "remediation_required.json"), "w") as f:
                json.dump(rr, f, indent=2)
            update_progress(output_dir, {
                "status": "remediation_required",
                "final_audit_result": "fail",
            })
            sys.exit(1)
        else:
            # Collect AUTO actions
            remediation_actions = audit.get("remediation_actions", [])
            auto_actions = [a for a in remediation_actions if a.get("auto_executable", False)]
            manual_actions = [a for a in remediation_actions if not a.get("auto_executable", False)]

            if not auto_actions:
                print("No AUTO actions available — only MANUAL actions remain")
                rr = {
                    "run_id": run_id,
                    "original_audit_result": "fail",
                    "remediation_iterations_attempted": remediation_iter,
                    "final_audit_result": "fail",
                    "pending_manual_actions": manual_actions,
                    "instructions": "Manual intervention required. Review pending_manual_actions."
                }
                with open(os.path.join(output_dir, "remediation_required.json"), "w") as f:
                    json.dump(rr, f, indent=2)
                update_progress(output_dir, {
                    "status": "remediation_required",
                    "final_audit_result": "fail",
                })
                sys.exit(1)

            # Apply AUTO actions
            print(f"Applying AUTO remediation actions: {[a['action_id'] for a in auto_actions]}")
            steps_to_rerun = set()
            actions_applied = []

            for action in auto_actions:
                action_id = action["action_id"]
                params = action.get("suggested_parameters", {})
                actions_applied.append(action_id)

                if action_id == "remove_monotonic_index_features":
                    exclude_list = params.get("exclude_features", [])
                    if exclude_list:
                        new_excludes = ",".join(exclude_list)
                        if exclude_features_param:
                            exclude_features_param += "," + new_excludes
                        else:
                            exclude_features_param = new_excludes
                    steps_to_rerun.update(["12", "13", "14", "15", "17"])

                elif action_id == "improve_model_performance":
                    force_expansion_param = "true"
                    steps_to_rerun.update(["13", "14", "15", "17"])

                elif action_id == "extend_lag_window":
                    max_lag_param = str(params.get("max_lag", 20))
                    steps_to_rerun.update(["12", "13", "14", "15", "17"])

                elif action_id == "add_seasonal_features":
                    seasonal_features_param = "true"
                    steps_to_rerun.update(["12", "13", "14", "15", "17"])

                elif action_id == "increase_regularization":
                    regularization_param = "ridge_cv"
                    steps_to_rerun.update(["13", "14", "15", "17"])

                elif action_id == "try_alternative_models":
                    extra_models_param = "histgradient,svr"
                    steps_to_rerun.update(["13", "14", "15", "17"])

                elif action_id == "use_time_series_split":
                    split_mode_param = "time_series"
                    steps_to_rerun.update(["12", "13", "14", "15", "17"])

            # Delete output JSONs for steps to re-run
            step_file_map = {
                "12": "step-12-features.json",
                "13": "step-13-training.json",
                "14": "step-14-evaluation.json",
                "15": "step-15-selection.json",
                "16": "step-16-report.md",
                "17": "step-17-audit.json",
            }
            for step_num in sorted(steps_to_rerun):
                if step_num in step_file_map:
                    delete_step_outputs(output_dir, [step_file_map[step_num]])
            # Also delete step 16 if we're re-running 15
            if "15" in steps_to_rerun:
                delete_step_outputs(output_dir, ["step-16-report.md"])
            # Update completed_steps to remove re-run steps
            with open(os.path.join(output_dir, "progress.json")) as f:
                progress = json.load(f)
            completed = progress.get("completed_steps", [])
            step_name_map = {
                "12": "12-feature-extraction",
                "13": "13-model-training",
                "14": "14-model-evaluation",
                "15": "15-model-selection",
                "16": "16-result-presentation",
                "17": "17-critical-self-audit",
            }
            for step_num in steps_to_rerun:
                step_name = step_name_map.get(step_num, "")
                if step_name in completed:
                    completed.remove(step_name)
            progress["completed_steps"] = completed
            with open(os.path.join(output_dir, "progress.json"), "w") as f:
                json.dump(progress, f, indent=2)

            remediation_iter += 1
            update_progress(output_dir, {
                "remediation": {
                    "iteration": remediation_iter,
                    "max_iterations": MAX_REMEDIATION,
                    "actions_applied": actions_applied,
                    "steps_rerun": sorted(list(steps_to_rerun)),
                    "audit_result_before": "fail",
                }
            })

    # ─── Step 18 ─────────────────────────────────────────────────────────────
    rc = run_with_retry("step-18", step_script("step_18_executive_summary.py"))
    if rc != 0:
        print("WARNING: Step 18 failed (non-fatal)")

    # Final status check
    progress = load_progress(output_dir)
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Status: {progress.get('status')}")
    print(f"Final audit: {progress.get('final_audit_result')}")
    print(f"Completed steps: {progress.get('completed_steps', [])}")


if __name__ == "__main__":
    main()
