import pandas as pd
import pytest

from src.validation import DataValidator


@pytest.fixture
def valid_reviews():
    return pd.DataFrame({
        'user_id': ['u1', 'u2'],
        'parent_asin': ['i1', 'i2'],
        'rating': [5, 4],
        'timestamp': [12345, 12346]
    })

@pytest.fixture
def valid_meta():
    return pd.DataFrame({
        'parent_asin': ['i1', 'i2'],
        'title': ['Item 1', 'Item 2']
    })

def test_validation_success(valid_reviews, valid_meta):
    validator = DataValidator()
    assert validator.validate(valid_reviews, valid_meta) is True

def test_validation_missing_review_col(valid_reviews, valid_meta):
    validator = DataValidator()
    invalid_reviews = valid_reviews.drop(columns=['rating'])
    with pytest.raises(ValueError, match="Missing required review columns"):
        validator.validate(invalid_reviews, valid_meta)

def test_validation_missing_meta_col(valid_reviews, valid_meta):
    validator = DataValidator()
    invalid_meta = valid_meta.drop(columns=['title'])
    with pytest.raises(ValueError, match="Missing required metadata columns"):
        validator.validate(valid_reviews, invalid_meta)
