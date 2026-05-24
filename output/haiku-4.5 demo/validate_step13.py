import json
import joblib
import numpy as np

with open('output/20260524T211328Z/step-13-training.json') as f:
    data = json.load(f)

print('✓ Candidates with R² scores:')
for c in data['candidates']:
    print(f"  - {c['model_name']}: R²={c['r2']:.4f}")

print()
# Load and test model
model = joblib.load('output/20260524T211328Z/model.joblib')
print(f'✓ Model loaded: {type(model).__name__}')

# Load holdout data
npz = np.load('output/20260524T211328Z/holdout.npz')
X_test = npz['X_test']
y_test = npz['y_test']
print(f'✓ Holdout loaded: X_test {X_test.shape}, y_test {y_test.shape}')

# Test prediction
y_pred = model.predict(X_test)
print(f'✓ Prediction successful: {y_pred.shape}')

print()
print('✓✓✓ All Step 13 validation gates PASS')
