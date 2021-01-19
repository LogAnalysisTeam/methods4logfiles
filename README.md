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
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── embeddings     <- Log message embedding models, e.g., fastText.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting.
    │   └── results        <- All kinds of results in machine-friendly formats (human-readable reports can be generated out of these).
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`.
    │
    ├── scripts            <- Scripts running the code to reproduce published results The scripts are meant for a batching system (METACENTRUM, RCI).
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module.
        │
        ├── data           <- Scripts to download or generate data.
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling.
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions.
        │   ├── predict_model.py **FIX**
        │   └── train_model.py   **FIX**
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py     **FIX** 

```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.
