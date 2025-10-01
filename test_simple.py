# test_simple.py - Minimal test
print("Testing basic imports...")

try:
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA
    print("✅ All imports successful!")
    
    # Test ARIMA model creation
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    data = pd.DataFrame({'price': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}, index=dates)
    
    model = ARIMA(data['price'], order=(1,1,1))
    fitted = model.fit()
    print("✅ ARIMA model works!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")