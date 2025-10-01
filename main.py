from src.data_loader import BTCDataLoader
from src.analysis import BTCAnalyzer
import pandas as pd
import os

def complete_workflow():
    """Complete workflow from data collection to prediction"""
    
    print(" STARTING BTC PRICE PREDICTION PROJECT")
    print("=" * 60)
    
    # Step 1: Data Collection
    print("\n1️ -----DATA COLLECTION-----")
    loader = BTCDataLoader()
    btc_data = loader.fetch_data(period="1y")  # Start with 1 year for testing
    
    if btc_data is None:
        print(" Failed to fetch data. Using saved data if available.")
        try:
            btc_data = pd.read_csv('data/btc_data.csv', index_col=0, parse_dates=True)
            print(" Loaded existing data file")
        except Exception as e:
            print(f" No saved data found: {e}")
            print("Please check your internet connection and try again.")
            return
    
    # Save data
    loader.save_data(btc_data)
    print(f" Data shape: {btc_data.shape}")
    print(f" Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
    
    # Step 2: Data Analysis & Stationarity Test
    print("\n2️---- DATA ANALYSIS & STATIONARITY TEST----")
    analyzer = BTCAnalyzer(btc_data)
    
    # Plot the price series
    analyzer.plot_price_series()
    
    # Perform stationarity test
    adf_result = analyzer.augmented_dickey_fuller_test(btc_data['price'])
    
    # Step 3: Model Training & Prediction
    print("\n3️ --------MODEL TRAINING & PREDICTION-------")
    
    # Import the model class
    try:
        from src.model_trainer import ARIMAModel
        print(" Successfully imported ARIMAModel")
    except ImportError as e:
        print(f" Failed to import ARIMAModel: {e}")
        print("Please check the model_trainer.py file")
        return
    
    # Load the data for modeling
    try:
        data = pd.read_csv('data/btc_data.csv', index_col=0, parse_dates=True)
        series = pd.to_numeric(data['price'], errors='coerce')
        print(f" Loaded {len(series)} data points for modeling")
    except Exception as e:
        print(f" Error loading data: {e}")
        return
    
    # Initialize and train model
    btc_model = ARIMAModel(data)
    
    # Use ARIMA(2,1,2) as specified in project
    best_order = (2, 1, 2)
    print(f" Using ARIMA{best_order} as specified in project")
    
    # Train model
    fitted_model = btc_model.train_arima(series, order=best_order)
    
    if fitted_model is None:
        print(" Model training failed. Please check the errors above.")
        return
    print("\ -------TRAINING ARIMA + GARCH MODEL...------")
    results = btc_model.train_arima_garch(series, arima_order=best_order, garch_order=(1,1))

    if results:
        arima_fitted = results["arima"]
        garch_fitted = results["garch"]
        print(" Combined ARIMA+GARCH model trained successfully!")
        print(f" Best ARIMA order: {best_order}")
        print(" GARCH summary above shows volatility model details")
    
    # Make predictions
    print("\n MAKING PREDICTIONS...")
    predictions = btc_model.make_predictions(steps=30)
    
    # Evaluate model
    print("\n EVALUATING MODEL...")
    evaluation = btc_model.evaluate_model(series, predictions['fitted'])
    
    # Plot results
    print("\n PLOTTING RESULTS...")
    btc_model.plot_results(series)
    
    # Save model
    btc_model.save_model()
    
    # Print forecast
    print("\n PRICE FORECAST (Next 30 days):")
    print("=" * 50)
    for i, (date, price) in enumerate(zip(predictions['forecast'].index, predictions['forecast'].values), 1):
        print(f"Day {i:2d}: {date.strftime('%Y-%m-%d')} - ${price:,.2f}")
    
    print("\n PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f" Final Model Accuracy: {evaluation['accuracy']:.1f}%")
    print(f" MAPE: {evaluation['mape']:.2f}%")
    print(f" Model saved to: models/arima_model.pkl")
    print("=" * 50)

if __name__ == "__main__":
    complete_workflow()