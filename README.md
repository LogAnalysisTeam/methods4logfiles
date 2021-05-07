# Anomaly Detection Methods for Log Files
This is Martin's repo of his diploma thesis. 

# Project Organization
```
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── embeddings     <- Log message embedding models, e.g., fastText.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting.
    │   └── results        <- All kinds of results in machine-friendly formats (human-readable reports can be generated out of these).
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`.
    │
    ├── scripts            <- Scripts running the code to reproduce published results. The scripts are meant for a batching system (METACENTRUM, RCI).
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── data           <- Scripts to download or generate data.
        │
        ├── features       <- Scripts to turn raw data into features for modeling.
        │
        ├── models         <- Scripts to train models with their definitions.
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations

```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.

# Getting Started

## HDFS

After downloading HDFS1 data set, save the data to this path `methods4logfiles/data/raw/HDFS1`. After that, you can execute `prepare_hdfs.py` located in `methods4logfiles/src/data`.

```
python prepare_hdfs.py <n_folds>
```

where `<n_folds>` is an integer which specifies the number of folds in CV. If it is equal to 1 then it splits the data (10% testing set, 90% training set). The output can be found in `methods4logfiles/data/interim/HDFS1`. One can access other options using `-h` or `--help`.

Next, train fastText model on training data set. The script is located in `methods4logfiles/scripts`.

```
sbatch train_fasttext_HDFS1.sh <n_folds>
```

This command executes fastText training using `SLURM` in the cloud. One can re-use a part of the script to run the training locally.

Go to `methods4logfiles/src/features` and run `build_features_hdfs.py` as follows.

```
python build_features_hdfs.py <model> --per-block
```

where model is a path to a trained fastText model. The current implementation allows you to use either `--per-block` or `--per-log`. This controls the method of creating the embeddings and labels and saving them as NumPy arrays. One can access other options using `-h` or `--help`.

Once the data are prepared and saved in corresponding folders, one can experiment with the implemented models. The implementation follows scikit-learn recommendation, see an example below.

```python
model = AETCN()
model.set_params(**config['hyperparameters'])

model.fit(x_train[y_train == 0])  # train on normal data examples
y_pred = model.predict(x_val)  # return MSE per example

theta, f1_score = find_optimal_threshold(y_val, y_pred)
y_pred = classify(y_pred, theta)
metrics_report(y_val, y_pred)
confusion_matrix(y_val, y_pred)
```

The hyperparameter tuning of the provided models is located in files starting with `train_`. The final evaluation of all models is provided in `evaluate models.py`.
