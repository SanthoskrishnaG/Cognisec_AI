def test_imports():
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import xgboost as xgb
    import torch
    import matplotlib
    import seaborn
    import yaml
    import flask
    import joblib

    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert xgb.__version__ is not None
    assert torch.__version__ is not None
    assert yaml.__version__ is not None
    assert flask is not None

