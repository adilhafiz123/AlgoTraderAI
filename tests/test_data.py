"""
Tests for the data module.
"""

import pytest
import pandas as pd
from src.data import calculate_momentum

def test_calculate_momentum():
    """Test momentum calculation."""
    # Create test data
    data = pd.Series([1, 2, 3, 4, 5])
    window = 2
    
    # Calculate momentum
    momentum = calculate_momentum(data, window)
    
    # Check results
    assert len(momentum) == len(data)
    assert momentum.iloc[0:2].isna().all()  # First two values should be NaN
    assert momentum.iloc[2] == 2  # 3 - 1 = 2
    assert momentum.iloc[3] == 2  # 4 - 2 = 2
    assert momentum.iloc[4] == 2  # 5 - 3 = 2

def test_calculate_momentum_invalid_window():
    """Test momentum calculation with invalid window."""
    data = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        calculate_momentum(data, window=0) 