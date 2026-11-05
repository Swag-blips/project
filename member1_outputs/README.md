# Member 1 handoff: weatherAUS preprocessing

**Target:** `RainTomorrow` (No = 0, Yes = 1). **Source:** `weatherAUS.csv`, 145,460 rows.
**Rebuild everything:** run `Member1_Preprocessing.ipynb` with `weatherAUS.csv` beside it (about 2 minutes, seed 42).

## Which file to use

| Member | Use | Notes |
|---|---|---|
| 2 (EDA) | `weather_clean.csv` | Original units, readable categories. Blanks mean not measured. Not filled or capped, so charts show real values. Has `Month`, `Year` and `split`. |
| 3 (Models) | `X_train.csv.gz`, `y_train.csv`, `X_val.csv.gz`, `y_val.csv`, `X_test.csv.gz`, `y_test.csv` | 133 scaled features, no blanks. `pd.read_csv` opens `.gz` files directly. |
| 3 (optional) | `X_train_smote.csv.gz`, `y_train_smote.csv` | Balanced training set. Train only; see Class imbalance. |
| 4 (Dashboard) | `preprocessor.joblib`, `weather_clean.csv` | Turns raw form inputs into model features. |

| Set | Rows | Rain rate |
|---|---:|---:|
| Train | 99,535 | 22.42% |
| Validation | 21,329 | 22.42% |
| Test | 21,329 | 22.42% |
| Train with SMOTE | 154,442 | 50.00% |

## What was done

- **Validation:** no duplicate rows or duplicate date and location pairs. `MinTemp` never exceeds `MaxTemp`, and `RainToday` always agrees with `Rainfall`.
- **Dropped:** 3,267 rows with no `RainTomorrow`, leaving 142,193.
- **Treated as missing:** 3 cloud readings above 8 oktas and 5 humidity readings of exactly 0.
- **Left unchanged:** 5 rows where wind speed exceeds gust speed, because the correct value is unknown.
- **Date:** replaced by `Month`, one-hot encoded. `Year` is excluded from model features.
- **Missing numbers:** filled with the training median. `Sunshine`, `Evaporation`, `Cloud9am` and `Cloud3pm` also get a was-missing flag, because 12 to 19 of the 49 locations never report them.
- **Missing categories:** wind directions become `Unknown`. `RainToday` takes the most common value.
- **Outliers:** IQR capping (1.5 x IQR) on numeric columns, affecting up to 2.1% of training values per column. `Rainfall` uses log(1 + x) instead, since most days have no rain and capping would erase the difference between light rain and a storm.
- **Encoding and scaling:** one-hot for `Location`, wind directions and `Month`. `RainToday` is 0 or 1. `StandardScaler` on numeric columns.
- **Avoiding leakage:** the split comes first, and all medians, caps, encoders and scaling are learned from the training set only.

## Class imbalance

Logistic regression on the validation set, at a 0.5 threshold:

| Approach | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| No resampling | 0.8693 | 0.7337 | 0.5186 | 0.6077 |
| `class_weight='balanced'` | 0.8700 | 0.5240 | 0.7754 | 0.6254 |
| SMOTENC | 0.8682 | 0.5286 | 0.7687 | 0.6264 |

SMOTE gave no real advantage over class weights here. **Recommendation:** start with `X_train` and class weights. These are baseline results for one model, so Member 3 can retest with others. Keep validation and test unresampled either way.

## Member 3: loading the data

```python
import pandas as pd

X_train = pd.read_csv('member1_outputs/X_train.csv.gz')
y_train = pd.read_csv('member1_outputs/y_train.csv').RainTomorrow
```

Tune on train and validation, and use test only for the final comparison.

## Member 4: scoring form input

Save a model trained on these features, then:

```python
import joblib
import numpy as np
import pandas as pd

preprocessor = joblib.load('member1_outputs/preprocessor.joblib')

row = pd.DataFrame([{
    'Rainfall': 0.0, 'MinTemp': 13.4, 'MaxTemp': 22.9, 'Evaporation': np.nan, 'Sunshine': np.nan,
    'WindGustSpeed': 44.0, 'WindSpeed9am': 20.0, 'WindSpeed3pm': 24.0,
    'Humidity9am': 71.0, 'Humidity3pm': 22.0, 'Pressure9am': 1007.7, 'Pressure3pm': 1007.1,
    'Cloud9am': 8.0, 'Cloud3pm': np.nan, 'Temp9am': 16.9, 'Temp3pm': 21.8,
    'Location': 'Albury', 'WindGustDir': 'W', 'WindDir9am': 'W', 'WindDir3pm': 'WNW',
    'RainToday': 'No', 'Month': 12,
}])
row[['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']] = (
    row[['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']].astype(object)
)

features = preprocessor.transform(row)
probability = model.predict_proba(features)[0, 1]
```

The form must supply all 22 columns above, with `Month` (1 to 12) instead of a date. Numeric fields may be `np.nan`.

## Known limitations

- **Random split:** neighbouring days from one station can land in different sets, so scores may run slightly high compared with forecasting future dates.
- **Library versions:** `preprocessor.joblib` was saved with scikit-learn 1.9.1 on Python 3.14. If it fails to load or warns about versions, rerun the notebook in your environment. Exact versions are in `preprocessing_summary.json`.
- **Brief vs. dataset:** the project brief describes bank churn (`Exited`, `Geography`, `Balance`). Its charts and form fields need weather equivalents.
