import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.best_order = None
        self.predictions = None
        self.fitted_model = None
        
    def calculate_mape(self, actual, predicted):
        """Calculate Mean Absolute Percentage Error"""
        # Filter out zeros in actual values to avoid division by zero
        mask = actual != 0
        actual = actual[mask]
        predicted = predicted[mask]
        return np.mean(np.abs((actual - predicted) / actual)) * 100
    
    def train_arima(self, series, order=(2,1,2)):
        """
        Train ARIMA model with specified order
        """
        print(f"🏋️ Training ARIMA{order} model...")
        
        try:
            # Make sure series is clean
            series_clean = pd.to_numeric(series, errors='coerce').dropna().astype(float)
            self.model = ARIMA(series_clean, order=order)
            self.fitted_model = self.model.fit()
            
            print(" Model training completed!")
            print(self.fitted_model.summary())
            
        except Exception as e:
            print(f" ARIMA training failed: {e}")
            print(" Trying with different parameters...")
            # Fallback to simpler model
            try:
                self.model = ARIMA(series_clean, order=(1,1,1))
                self.fitted_model = self.model.fit()
                print(" Fallback model training completed!")
            except Exception as e2:
                print(f" Fallback also failed: {e2}")
                return None
        
        return self.fitted_model
    
    def make_predictions(self, steps=30):
        """
        Make future predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained yet!")
        
        # Get in-sample predictions
        fitted_values = self.fitted_model.fittedvalues
        
        # Make future predictions
        forecast = self.fitted_model.forecast(steps=steps)
        
        self.predictions = {
            'fitted': fitted_values,
            'forecast': forecast
        }
        
        return self.predictions
    
    
    
    def evaluate_model(self, actual, predicted):
        """
        Comprehensive model evaluation
        """
        try:
            # Align indices - handle potential index mismatches
            common_index = predicted.index
            actual_aligned = actual.loc[common_index]
            
            mape = self.calculate_mape(actual_aligned.values, predicted.values)
            mae = mean_absolute_error(actual_aligned.values, predicted.values)
            rmse = np.sqrt(mean_squared_error(actual_aligned.values, predicted.values))
            
            accuracy = 100 - mape
            
            print('=' * 50)
            print(' MODEL EVALUATION METRICS')
            print('=' * 50)
            print(f" MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
            print(f" Accuracy: {accuracy:.1f}%")
            print(f" MAE (Mean Absolute Error): ${mae:.2f}")
            print(f" RMSE (Root Mean Square Error): ${rmse:.2f}")
            print('=' * 50)
            
            return {
                'mape': mape,
                'accuracy': accuracy,
                'mae': mae,
                'rmse': rmse
            }
        except Exception as e:
            print(f" Error in evaluation: {e}")
            return {
                'mape': 0,
                'accuracy': 0,
                'mae': 0,
                'rmse': 0
            }
    
    def plot_results(self, original_series, forecast_days=30):
        """
        Plot original data, fitted values, and forecasts
        """
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot original series
            plt.subplot(2, 1, 1)
            plt.plot(original_series.index, original_series.values, 
                    label='Actual Price', color='blue', alpha=0.7, linewidth=1)
            
            if self.predictions['fitted'] is not None:
                plt.plot(self.predictions['fitted'].index, 
                        self.predictions['fitted'].values, 
                        label='Fitted Values', color='red', alpha=0.8, linewidth=1)
            
            plt.title('BTC-USD Price: Actual vs Fitted Values', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Plot forecast
            plt.subplot(2, 1, 2)
            last_date = original_series.index[-1]
            forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                         periods=forecast_days, freq='D')
            
            # Plot last 90 days of historical data
            historical_days = min(90, len(original_series))
            plt.plot(original_series.index[-historical_days:], original_series.values[-historical_days:], 
                    label='Historical', color='blue', linewidth=2)
            plt.plot(forecast_index, self.predictions['forecast'], 
                    label='Forecast', color='green', linewidth=2, marker='o', markersize=4)
            
            plt.title(f'BTC-USD Price Forecast (Next {forecast_days} Days)', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f" Error plotting results: {e}")
    
    def save_model(self, filename='models/arima_model.pkl'):
        """Save trained model"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.fitted_model, f)
            print(f" Model saved to {filename}")
        except Exception as e:
            print(f" Error saving model: {e}")
    def train_arima_garch(self, series, arima_order=(1,1,1), garch_order=(1,1)):
        """
        Train ARIMA-GARCH model:
        - ARIMA for mean/trend
        - GARCH for volatility of residuals
        """
        print(f"🏋️ Training ARIMA{arima_order} + GARCH{garch_order} model...")

        try:
            # Ensure numeric clean series
            series_clean = pd.to_numeric(series, errors='coerce').dropna().astype(float)

            # Step 1: Fit ARIMA for mean
            arima_model = ARIMA(series_clean, order=arima_order)
            arima_fitted = arima_model.fit()

            # Step 2: Get residuals
            residuals = arima_fitted.resid

            # Step 3: Fit GARCH on residuals
            garch = arch_model(residuals, vol="GARCH", p=garch_order[0], q=garch_order[1])
            garch_fitted = garch.fit(disp="off")

            print(" ARIMA-GARCH training completed!")
            print(garch_fitted.summary())

            self.fitted_model = arima_fitted
            self.garch_model = garch_fitted
            return {"arima": arima_fitted, "garch": garch_fitted}

        except Exception as e:
            print(f" ARIMA-GARCH training failed: {e}")
            return None


# Test function
def test_model():
    """Test the ARIMA model with sample data"""
    print(" Testing ARIMA Model...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    # Create a simple trend + noise
    prices = 30000 + np.cumsum(np.random.randn(100) * 100)
    sample_data = pd.DataFrame({'price': prices}, index=dates)
    
    # Initialize and test model
    model = ARIMAModel(sample_data)
    fitted_model = model.train_arima(sample_data['price'], order=(1,1,1))
    
    if fitted_model is not None:
        predictions = model.make_predictions(steps=10)
        evaluation = model.evaluate_model(sample_data['price'], predictions['fitted'])
        print(" Model test completed successfully!")
    else:
        print(" Model test failed")

if __name__ == "__main__":
    test_model()
