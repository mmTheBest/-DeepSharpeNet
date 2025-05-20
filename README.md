# CNN-based Stock Price Prediction with Instantaneous Sharpe Ratio

This project implements a CNN-LSTM model to predict the risk-adjusted stock return, represented as an instantaneous Sharpe ratio. The prediction target quantifies the expected k-bar return normalized by its inherent volatility, allowing the model to focus on high signal-to-noise opportunities.

## Prediction Target: Instantaneous Sharpe Ratio

The model predicts `s_t^(k)`, defined as:

\[
  s_t^{(k)} = \frac{C_{t+k} - C_t}{C_t} \Bigg/ \sqrt{\frac{1}{k}\sum_{i=1}^{k}r_{t+i}^2}
\]

Where:
- `C_t` is the closing price at time `t`
- `r_{t+i}` are the future returns over the next `k` periods
- The denominator represents the square root of the mean of squared future returns (RMS volatility)

### Interpretation

This ratio represents an **instantaneous Sharpe** for a k-bar trade, where:
- `s = 1` means the expected return equals one standard deviation of future noise
- `s = 2` means two standard deviations, etc.

This standardized measure helps the model identify opportunities with attractive risk-reward profiles rather than just high absolute returns.

## Model Architecture

The model employs a sophisticated CNN-LSTM architecture with attention mechanisms:

1. **Feature Importance Layer**:
   - Learns to weight input features with a trainable parameter vector
   - Uses softmax activation to create a probability distribution over features

2. **Convolutional Blocks**:
   - Gated 1D convolutional layers (similar to GLU) for adaptive feature extraction
   - Batch normalization and ReLU activation
   
3. **Dual Attention Mechanism**:
   - Channel attention: Captures feature-wise importance
   - Spatial attention: Focuses on important time steps
   
4. **Temporal Processing**:
   - Temporal attention to aggregate features across time
   - Bidirectional LSTM with 2 layers to capture sequential patterns
   
5. **Output Layers**:
   - Residual blocks with skip connections
   - Trainable output scaling with tanh activation for bounded predictions

## Implementation Details

### Data Preprocessing

- Raw OHLCV data is transformed into 426+ technical indicators
- For "raw mode," only 9 base features are used: open, high, low, close, volume, ret, ma5, ma10, ma20
- Risk-adjusted returns are calculated using the exact formula shown above
- Data is normalized feature-wise to zero mean and unit variance

### Training Process

- Uses sliding window approach for time series data (default window = 30 bars)
- Loss function combines multiple components:
  - Primary loss: Smooth L1 (Huber) loss
  - Correlation penalty: Encourages alignment between prediction and target trends
  - R²-like term: Normalizes MSE by target variance
  - Sparsity regularization: L1 penalty on feature importance
  - Group LASSO: Encourages feature group selection

- Supports balanced sampling across magnitude bins to handle class imbalance
- Implements early stopping with patience and learning rate scheduling

### Evaluation Metrics

- MSE/RMSE: Basic prediction error
- MAE: Absolute magnitude of errors
- Directional Accuracy: % of correct direction predictions
- R²: Proportion of variance explained

### Backtesting

- Converts model predictions to trading signals (+1/-1/0)
- Computes equity curves, returns, Sharpe ratios, and max drawdowns
- Compares performance against buy-and-hold benchmark

## Usage

### Data Preprocessing

```bash
python data_preprocessing.py data/stock_data.xlsx --out processed_data/output.csv --k-bar 1 --vol-window 20
```

### Training

```bash
python train.py processed_data/output.csv --win 30 --target-shift 1 --epochs 20 --raw-only
```

### Evaluation

```bash
python eval.py processed_data/output.csv model_checkpoint/model.pt --win 30
```

### Backtesting

```bash
python backtest.py eval_out/predictions.csv
```


## Key Findings

1. The risk-adjusted target provides more stable training signals
2. Short-term predictions (target_shift=1) significantly outperform longer horizons
3. Using only raw features often yields better results than all engineered features
4. Balanced sampling improves training when target distribution is highly skewed

## Implementation Notes

- The CNN extracts local patterns from price action
- Attention mechanisms help focus on relevant parts of the time series
- LSTM captures temporal dependencies and sequence information
- Feature importance learning identifies the most predictive indicators
- Output scaling helps maintain prediction stability

## Project Structure

The codebase follows a modular design with four main components:

1. **Data Preprocessing** (`data_preprocessing.py`):
   - Loads raw OHLCV data from Excel files
   - Computes technical indicators (~426 features)
   - Calculates risk-adjusted returns
   - Saves processed data as CSV files

2. **Model Training** (`train.py`):
   - Builds and trains CNN-LSTM models
   - Uses sliding window approach for time series data
   - Implements various loss functions (MSE, correlation, R²-like)
   - Offers class-balanced sampling options

3. **Model Evaluation** (`eval.py`):
   - Evaluates model performance on test data
   - Calculates metrics (MSE, MAE, directional accuracy, R²)
   - Generates prediction files for backtesting

4. **Backtesting** (`backtest.py`):
   - Simulates trading strategies based on model predictions
   - Calculates performance metrics (return, Sharpe ratio, drawdown)
   - Compares to buy-and-hold benchmark

