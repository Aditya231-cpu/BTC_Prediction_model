import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

class BTCAnalyzer:
    def __init__(self, data):
        self.data = data
        self.is_stationary = None
        
    def plot_price_series(self):
        """Plot original BTC price series"""
        plt.figure(figsize=(15, 8))
        plt.plot(self.data.index, self.data['price'], linewidth=1, color='orange')
        plt.title('BTC-USD Price History', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def augmented_dickey_fuller_test(self, series):
        """
        Perform Augmented Dickey-Fuller test for stationarity
        H0: Series is non-stationary
        Ha: Series is stationary
        """
        result = adfuller(series.dropna())
        
        print('=' * 50)
        print(' AUGMENTED DICKEY-FULLER TEST')
        print('=' * 50)
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.3f}')
        
        if result[1] <= 0.05:
            print(" REJECT NULL HYPOTHESIS - Series is STATIONARY")
            self.is_stationary = True
        else:
            print(" FAIL TO REJECT NULL HYPOTHESIS - Series is NON-STATIONARY")
            self.is_stationary = False
        
        return result
    
    def make_stationary(self):
        """Apply differencing to make series stationary"""
        if not self.is_stationary:
            self.data['price_diff'] = self.data['price'].diff()
            return self.data['price_diff']
        return self.data['price']
    
    def plot_acf_pacf(self, series, lags=40):
        """Plot ACF and PACF for ARIMA parameter selection"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        plot_acf(series.dropna(), ax=ax1, lags=lags)
        ax1.set_title('Autocorrelation Function (ACF)', fontweight='bold')
        
        plot_pacf(series.dropna(), ax=ax2, lags=lags)
        ax2.set_title('Partial Autocorrelation Function (PACF)', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# Test function
def test_analysis():
    # Load data
    data = pd.read_csv('data/btc_data.csv', index_col=0, parse_dates=True)
    
    # Initialize analyzer
    analyzer = BTCAnalyzer(data)
    
    # Plot original series
    print(" Plotting BTC price history...")
    analyzer.plot_price_series()
    
    # Perform stationarity test
    print(" Performing stationarity test...")
    adf_result = analyzer.augmented_dickey_fuller_test(data['price'])
    
    # If not stationary, apply differencing and test again
    if not analyzer.is_stationary:
        print("\n Applying first differencing...")
        stationary_series = analyzer.make_stationary()
        adf_result_diff = analyzer.augmented_dickey_fuller_test(stationary_series.dropna())
        
        # Plot ACF/PACF of differenced series
        print("\n Plotting ACF/PACF for differenced series...")
        analyzer.plot_acf_pacf(stationary_series.dropna())

if __name__ == "__main__":
    test_analysis()