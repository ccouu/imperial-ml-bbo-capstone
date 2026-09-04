import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import Matern,RBF, ConstantKernel as C
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline         

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap
from xgboost import XGBRegressor

# ==========================================
# 1. The Single Model Architecture
# ==========================================
class SingleSurrogateNetwork(nn.Module):
    """
    A standalone PyTorch Neural Network representing a single member 
    of the Bayesian Optimization ensemble.
    """
    def __init__(self, input_dim, k_1=1, hidden_dim=8, activation="Tanh"):
        super().__init__()
        
        layers = []
        # First hidden layer (Input -> Hidden)
        layers.append(nn.Linear(input_dim, hidden_dim))
        match activation:
            case "Tanh":      
                layers.append(nn.Tanh())
            case "SiLU":
                layers.append(nn.SiLU())
            case "CELU":
                layers.append(nn.CELU())
            case "LeakyReLU":
                layers.append(nn.LeakyReLU())
            case "ELU":
                layers.append(nn.ELU())
            case _:
                layers.append(nn.ReLU())
        # Additional hidden layers based on k_1
        for _ in range(k_1 - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            match activation:
                case "Tanh":      
                    layers.append(nn.Tanh())
                case "SiLU":
                    layers.append(nn.SiLU())
                case "CELU":
                    layers.append(nn.CELU()) 
                case "LeakyReLU":
                    layers.append(nn.LeakyReLU())
                case "ELU":
                    layers.append(nn.ELU())                    
                case _:
                    layers.append(nn.ReLU())    
            
        # Output layer (Hidden -> 1 Continuous Value)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Replicates TensorFlow's TruncatedNormal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

    def fit(self, X_train, Y_train, lr=0.01, max_epochs=2000, patience=200, tolerance=1e-5, debug=False):
        """
        Custom training loop that mimics the Keras model.fit() API, 
        including manual Early Stopping and full-batch gradient descent.
        """
        # Convert NumPy arrays to PyTorch Tensors
        X_t = torch.tensor(X_train, dtype=torch.float32)
        Y_t = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)

        # PyTorch uses weight_decay in Adam for L2 Regularization
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.002)
        #optimizer = torch.optim.SGD(self.parameters(),lr=lr,momentum=0.9,weight_decay=0.001)


        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_weights = None
        epochs_no_improve = 0

        self.train() # Set model to training mode
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            
            # Forward pass (Full Batch)
            predictions = self(X_t)
            loss = criterion(predictions, Y_t)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Early Stopping Logic
            current_loss = loss.item()
            if epoch % 150 == 0 and debug ==True:
                print(f"Epoch {epoch} | Current Loss: {current_loss:.4f}")

            if epoch > 500:
                if current_loss < best_loss - tolerance:
                    best_loss = current_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    print(f"Early stopping at Epoch {epoch}")
                    break # Stop training early

        # Restore best weights before returning
        if best_weights is not None:
            self.load_state_dict(best_weights)


# ==========================================
# 2. The Global Ensemble Logic
# ==========================================
def train_deep_ensemble_pt(X, Y, K=5, k_1=1, hidden_dim=16, lr=0.01, max_epochs=2000, tolerance=1e-5, bootstrap=False, activation="Tanh", patience=200):
    """
    Orchestrates the training of K isolated PyTorch networks using bootstrapped data.
    """
    N, M = X.shape
    trained_networks = []
    network_scores = []
    
    # Pre-convert original absolute dataset for final evaluation
    X_orig_t = torch.tensor(X, dtype=torch.float32)
    Y_orig_t = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
    criterion = nn.MSELoss()

    for i in range(K):
        print(f"--- Training Network {i+1}/{K} ---")

        # 1. Bootstrapping: Random pick with replacement
        bootstrapped_indices = np.random.choice(N, size=N, replace=True)
        if bootstrap == True:
            X = X[bootstrapped_indices]
            Y = Y[bootstrapped_indices]

        # 2. Instantiate a fresh PyTorch Model
        model = SingleSurrogateNetwork(input_dim=M, k_1=k_1, hidden_dim=hidden_dim , activation=activation)

        # 3. Train using the bootstrapped data
        model.fit(X, Y, lr=lr, max_epochs=max_epochs, patience=patience, tolerance=tolerance)

        # 4. Evaluate accuracy score on the ORIGINAL absolute dataset
        model.eval() # Set to evaluation mode
        with torch.no_grad():
            preds = model(X_orig_t)
            mse_score = criterion(preds, Y_orig_t).item()

        trained_networks.append(model)
        network_scores.append(mse_score)
        print(f"Network {i+1} Original Dataset MSE: {mse_score:.6f}\n")

    return trained_networks, network_scores


# ==========================================
# 3. The Prediction Wrapper
# ==========================================
def predict_ensemble_pt(trained_networks, X_candidates):
    """
    Takes candidate coordinates and returns the Mean and Std Dev across the PyTorch ensemble.
    """
    # Convert large candidate grid to a single tensor
    X_candidates_t = torch.tensor(X_candidates, dtype=torch.float32)
    predictions = []

    for model in trained_networks:
        model.eval()
        with torch.no_grad(): # Disable gradient tracking for fast inference
            # Forward pass and convert back to flat NumPy array
            pred = model(X_candidates_t).numpy().flatten()
            predictions.append(pred)

    predictions = np.array(predictions) # Shape: (K, number_of_candidates)

    # Calculate Mean (Exploitation) and Standard Deviation (Exploration)
    ensemble_mean = np.mean(predictions, axis=0)
    ensemble_std = np.std(predictions, axis=0)

    return ensemble_mean, ensemble_std


def train_surrogate(X, Y, model_type="gp", **kwargs):
    """
    Universal training wrapper. Accepts **kwargs to allow dynamic hyperparameter 
    tuning from the cross-validation engine.
    """
    if model_type == "gp":
        gp = GaussianProcessRegressor(
            kernel=kwargs.get('kernel', None), 
            alpha=kwargs.get('alpha', 1e-6), 
            n_restarts_optimizer=kwargs.get('n_restarts', 10)
        )
        gp.fit(X, Y)
        return gp
        
    elif model_type == "ensemble":
        # PyTorch Ensemble Logic
        trained_networks, _ = train_deep_ensemble_pt(
            X, Y, 
            K=kwargs.get("K", 3), 
            k_1=kwargs.get("k_1", 1), 
            hidden_dim=kwargs.get("hidden_dim", 4), 
            lr=kwargs.get("lr", 0.005), 
            max_epochs=kwargs.get("max_epochs", 5000), 
            tolerance=kwargs.get("tolerance", 1e-4),
            activation=kwargs.get("activation", "Tanh")
        )
        return trained_networks
        
    elif model_type == "svr":
        svm = SVR(
            kernel=kwargs.get('kernel', 'rbf'), 
            C=kwargs.get('C', 1.0), 
            epsilon=kwargs.get('epsilon', 0.1)
        )
        svm.fit(X, Y)
        return svm
        
    elif model_type == "tree":
        forest = ExtraTreesRegressor(
            n_estimators=kwargs.get('n_estimators', 100), 
            max_depth=kwargs.get('max_depth', 6),
            random_state=42
        )
        forest.fit(X, Y)
        return forest

    elif model_type == "xgboost":
        xgb = XGBRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 3),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=42
        )
        xgb.fit(X, Y)
        return xgb
    elif model_type == "linear":
            degree = kwargs.get('degree', 1)
            fit_intercept = kwargs.get('fit_intercept', True)
            
            # If degree > 1, pipe the data through a polynomial transformer first
            if degree > 1:
                # include_bias=False because LinearRegression handles the intercept natively
                model = make_pipeline(
                    PolynomialFeatures(degree=degree, include_bias=False),
                    LinearRegression(fit_intercept=fit_intercept)
                )
            else:
                model = LinearRegression(fit_intercept=fit_intercept)
                
            model.fit(X, Y)
            return model 
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def run_cross_validation(x_data, y_data, model_type="gp", num_folds=5, **kwargs):
    """
    Universal CV engine for K-Fold or LOOCV. Dynamically initializes and trains 
    the selected model architecture using kwargs.
    """
    fold_train_losses = []
    fold_val_losses = []
    
    print(f"=== Running {num_folds}-Fold Cross-Validation for {model_type.upper()} ===")
    if kwargs:
        print(f"Model Parameters: {kwargs}\n")
    
    # Instantiate the Scikit-Learn KFold generator
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    for k, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        
        # Train and Validation split extraction
        x_train, y_train = x_data[train_idx], y_data[train_idx]
        x_val = x_data[val_idx].reshape(len(val_idx), -1)
        y_val = y_data[val_idx]
        
        # 1. Initialize AND Train the model using the wrapper
        model = train_surrogate(x_train, y_train, model_type=model_type, **kwargs)
        
        # 2. Unified Prediction Handling
        if model_type == "ensemble":
            with torch.no_grad():
                train_pred, _ = predict_ensemble_pt(model, x_train)
                val_pred, _ = predict_ensemble_pt(model, x_val)
        else:
            train_pred = model.predict(x_train)
            val_pred = model.predict(x_val)
            
        # 3. Calculate Metrics
        fold_train_mse = np.mean((train_pred - y_train) ** 2)
        fold_val_mse = np.mean((val_pred - y_val) ** 2)
        
        fold_train_losses.append(fold_train_mse)
        fold_val_losses.append(fold_val_mse)
        
        if num_folds < 20:  # Avoid terminal flooding during long LOOCV runs
            print(f"Fold {k+1:02d} | Train MSE: {fold_train_mse:.4f} | Val MSE: {fold_val_mse:.4f}")
            
    #print("\n=== SYSTEM PERFORMANCE SUMMARY ===")
    #print(f"CV Train Baseline Error: {np.mean(fold_train_losses):.4f}")
    #print(f"CV Out-of-Sample Generalization Error: {np.mean(fold_val_losses):.4f}\n\n")
    
    return fold_train_losses, fold_val_losses