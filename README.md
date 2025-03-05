usage: benchmark.py [-h] [--classification] [--regression] [--xColumns COLUMN/S [COLUMN/S ...]]
                    --yColumn COLUMN [COLUMN ...]
                    file

A script to benchmark the performance of different machine learning models.

positional arguments:
  file                  Path to the dataset CSV file.

options:
  -h, --help            show this help message and exit
  --classification      Specifies a classification machine learning model.
  --regression          Specifies a regression machine learning model.
  --xColumns COLUMN/S [COLUMN/S ...]
                        Specifies the features to use to predict.
  --yColumn COLUMN [COLUMN ...]
                        Specifies the label/value to predict.
