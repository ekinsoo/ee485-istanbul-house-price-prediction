# Istanbul House Price Prediction (EE485)

This is a course project I worked on with Arda Bayam for EE485, Statistical Learning and Data Analytics. We used housing listings from Istanbul to compare three regression models: ridge regression, a neural network, and a decision tree. The models were implemented with NumPy; pandas was used to load and prepare the data.

I implemented the ridge regression and neural network parts and worked on the comparison plots and report. Arda worked on data preparation and the decision tree. We tuned and debugged the project together.

## Results

The test-set R² scores reported in the [final report](EE485ProjectFinalReport_Group18.pdf) were:

| Model | R² |
| --- | ---: |
| Neural network | 0.7228 |
| Ridge regression | 0.6385 |
| Decision tree | 0.6291 |

This is a small class project, so I would treat these as results for this dataset and split rather than as a general house-price estimator.

## Run it

1. Install the packages in `requirements.txt`.
2. Download the [House Price Dataset from Kaggle](https://www.kaggle.com/datasets/aselasel/house-price-dataset) and place its `HouseData.csv` file in the repository root.
3. Run `python main.py` from the repository root.

The script writes a processed-data cache to `processed_data.npz` and plots to `report/`. The dataset and generated cache are not included here. The PDF is the final course report, with student ID numbers removed from the public copy.
