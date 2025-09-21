import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from pred_catboost import load_data  

def test_load_data_shape():
    train, test = load_data()
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert "Risk" in train.columns
    assert "Id" in test.columns

def test_numeric_columns_exist():
    train, _ = load_data()
    for col in ["LoanDuration", "LoanAmount", "Age"]:
        assert col in train.columns
