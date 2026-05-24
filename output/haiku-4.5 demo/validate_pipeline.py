#!/usr/bin/env python3
import json
from pathlib import Path

def check_gate(condition, gate_name):
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"{status:8s} | {gate_name}")
    return condition

output_dir = Path(".")
all_pass = True

print("\n" + "="*80)
print("VALIDATION GATES")
print("="*80 + "\n")

# Step 10 Gates
print("Step 10 (CSV Read & Cleansing):")
with open("step-10-cleanse.json") as f:
    s10 = json.load(f)
all_pass &= check_gate("step" in s10 and s10["step"] == "10-csv-read-cleansing", "Has correct step ID")
all_pass &= check_gate(s10.get("row_count_after", 0) > 0, "Row count > 0")
all_pass &= check_gate("target_column_normalized" in s10, "Target column normalized")
all_pass &= check_gate("null_rate" in s10, "Null rate present")
cleaned_path = s10["artifacts"]["cleaned_parquet"]
if "\\" in cleaned_path:
    cleaned_path = cleaned_path.split("\\")[-1]  # Extract filename
all_pass &= check_gate((output_dir / cleaned_path).exists(), "Cleaned parquet exists")

# Step 11 Gates
print("\nStep 11 (Data Exploration):")
with open("step-11-exploration.json") as f:
    s11 = json.load(f)
all_pass &= check_gate("step" in s11, "Has step ID")
all_pass &= check_gate(len(s11.get("numeric_columns", [])) > 0, "Numeric columns present")
all_pass &= check_gate(len(s11.get("mi_ranking", [])) > 0, "MI ranking present")
all_pass &= check_gate(len(s11.get("recommended_features", [])) > 0, "Recommended features present")
all_pass &= check_gate("noise_mi_baseline" in s11, "Noise baseline present")

# Step 12 Gates
print("\nStep 12 (Feature Extraction):")
with open("step-12-features.json") as f:
    s12 = json.load(f)
all_pass &= check_gate("step" in s12, "Has step ID")
all_pass &= check_gate(len(s12.get("features", [])) > 0, "Features present")
all_pass &= check_gate(len(s12.get("features", [])) >= 2, "At least 2 features")
all_pass &= check_gate(isinstance(s12.get("features_excluded"), dict), "Features_excluded is dict")
features_path = s12["artifacts"]["features_parquet"]
if "\\" in features_path:
    features_path = features_path.split("\\")[-1]  # Extract filename
all_pass &= check_gate((output_dir / features_path).exists(), "Features parquet exists")

# Step 13 Gates
print("\nStep 13 (Model Training):")
with open("step-13-training.json") as f:
    s13 = json.load(f)
all_pass &= check_gate("step" in s13, "Has step ID")
all_pass &= check_gate((output_dir / "model.joblib").exists(), "model.joblib exists")
all_pass &= check_gate((output_dir / "holdout.npz").exists(), "holdout.npz exists")
all_pass &= check_gate(any(c.get("r2") is not None for c in s13.get("candidates", [])), "R² values present")

# Step 14 Gates
print("\nStep 14 (Model Evaluation):")
with open("step-14-evaluation.json") as f:
    s14 = json.load(f)
all_pass &= check_gate("step" in s14, "Has step ID")
all_pass &= check_gate(all(c.get("r2") is not None for c in s14.get("candidates", [])), "All candidates have R²")
all_pass &= check_gate(s14.get("quality_assessment") in ["acceptable", "marginal", "subpar"], "Quality assessment valid")
all_pass &= check_gate("target_stats" in s14, "Target stats present")

# Step 15 Gates
print("\nStep 15 (Model Selection):")
with open("step-15-selection.json") as f:
    s15 = json.load(f)
all_pass &= check_gate("step" in s15, "Has step ID")
all_pass &= check_gate(s15.get("quality_flag") in ["acceptable", "marginal", "subpar", "no_viable_candidate"], "Quality flag valid")
all_pass &= check_gate("selected_model" in s15, "Selected model present")
all_pass &= check_gate("rationale" in s15 and len(s15["rationale"]) > 10, "Rationale present")
all_pass &= check_gate("full_ranking" in s15, "Full ranking present")

# Step 16 Gates
print("\nStep 16 (Result Presentation):")
all_pass &= check_gate((output_dir / "step-16-report.md").exists() and (output_dir / "step-16-report.md").stat().st_size > 500, "Report.md > 500 bytes")

# Step 17 Gates
print("\nStep 17 (Critical Self-Audit):")
with open("step-17-audit.json") as f:
    s17 = json.load(f)
all_pass &= check_gate("step" in s17, "Has step ID")
all_pass &= check_gate(len(s17.get("checks", {})) == 5, "All 5 checks present")
all_pass &= check_gate(s17.get("overall_audit_result") in ["pass", "fail"], "Overall result valid")
all_pass &= check_gate("critical_findings" in s17, "Critical findings present")

print("\n" + "="*80)
if all_pass:
    print("✅ ALL VALIDATION GATES PASSED")
else:
    print("⚠️  SOME GATES FAILED - SEE ABOVE")
print("="*80 + "\n")
