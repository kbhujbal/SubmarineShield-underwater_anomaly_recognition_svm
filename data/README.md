# Data Directory

## Structure

```
data/
└── raw/
    └── sonar.all-data    # UCI Sonar dataset (you must download this manually)
```

## Manual Dataset Download Instructions

1. **Download the dataset** from the UCI Machine Learning Repository:
   - URL: https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data

2. **Save the file** as `sonar.all-data` in the `data/raw/` directory

3. **Verify the file**:
   - File name: `sonar.all-data`
   - Location: `data/raw/sonar.all-data`
   - Size: ~20 KB
   - Format: CSV with 208 rows, 61 columns (60 features + 1 label)

## Dataset Information

- **Source**: UCI Machine Learning Repository - Connectionist Bench (Sonar, Mines vs. Rocks)
- **Samples**: 208
- **Features**: 60 (sonar frequency band energy readings)
- **Labels**: R (Rock) or M (Mine)
- **Format**: Comma-separated values, no header row
