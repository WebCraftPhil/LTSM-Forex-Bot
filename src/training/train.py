"""
Training framework with Optuna hyperparameter optimization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import optuna
import logging
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..utils.config import get_config, TrainingConfig, ModelConfig
from ..utils.logging import get_logger, get_metrics_logger
from ..utils.seeds import set_seed, SeedManager
from ..models.lstm_fusion import MultiTimeframeLSTM, get_loss_function, save_model, load_model
from .metrics import TradingMetrics

logger = get_logger(__name__)
metrics_logger = get_metrics_logger(__name__)

class EarlyStopping:
    """Early stopping callback for training."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improvement = self.best_score - score > self.min_delta
        else:  # max
            improvement = score - self.best_score > self.min_delta

        if improvement:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

class TradingDataset(torch.utils.data.Dataset):
    """Dataset for trading data with sequences."""

    def __init__(self, X: np.ndarray, y: pd.Series, transform: Optional[Callable] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.values) if y.dtype == 'int64' else torch.FloatTensor(y.values)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'X': self.X[idx], 'y': self.y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

class LSTMTrainer:
    """LSTM model trainer with hyperparameter optimization."""

    def __init__(self, config: Optional[TrainingConfig] = None, model_config: Optional[ModelConfig] = None):
        self.config = config or get_config().training
        self.model_config = model_config or get_config().model

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize metrics
        self.metrics = TradingMetrics()

    def prepare_data(self, X: np.ndarray, y: pd.Series) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training."""

        # Create datasets
        dataset = TradingDataset(X, y)

        # Time series split for walk-forward validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Get train/val/test indices
        indices = list(range(len(dataset)))
        train_indices, val_indices, test_indices = self._get_split_indices(indices, tscv)

        # Create data loaders
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def _get_split_indices(self, indices: List[int], tscv: TimeSeriesSplit) -> Tuple[List[int], List[int], List[int]]:
        """Get train/validation/test indices for time series split."""

        # Use last 3 splits for train/val/test
        splits = list(tscv.split(indices))

        if len(splits) >= 3:
            _, test_indices = splits[-1]
            _, val_indices = splits[-2]
            train_indices = splits[0][0]  # First split for training

            # Adjust based on config ratios
            total_size = len(indices)
            train_size = int(total_size * self.config.train_ratio)
            val_size = int(total_size * self.config.val_ratio)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
        else:
            # Fallback to simple split
            total_size = len(indices)
            train_size = int(total_size * 0.7)
            val_size = int(total_size * 0.15)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

        return train_indices, val_indices, test_indices

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   criterion: nn.Module, optimizer: optim.Optimizer,
                   scheduler: Optional[optim.lr_scheduler._LRScheduler] = None) -> Dict[str, float]:
        """Train for one epoch."""

        model.train()
        total_loss = 0
        num_batches = len(train_loader)

        for batch in train_loader:
            X_batch = batch['X'].to(self.device)
            y_batch = batch['y'].to(self.device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            # Handle different output modes
            if self.model_config.output_mode == "regression":
                loss = criterion(outputs.squeeze(), y_batch)
            else:  # classification
                loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if scheduler:
            scheduler.step()

        avg_loss = total_loss / num_batches

        return {'train_loss': avg_loss}

    def validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                      criterion: nn.Module) -> Dict[str, float]:
        """Validate model performance."""

        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                X_batch = batch['X'].to(self.device)
                y_batch = batch['y'].to(self.device)

                outputs = model(X_batch)

                # Handle different output modes
                if self.model_config.output_mode == "regression":
                    loss = criterion(outputs.squeeze(), y_batch)
                    predictions = outputs.squeeze().cpu().numpy()
                else:  # classification
                    loss = criterion(outputs, y_batch)
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                targets = y_batch.cpu().numpy()
                total_loss += loss.item()

                all_predictions.extend(predictions)
                all_targets.extend(targets)

        avg_loss = total_loss / num_batches

        # Calculate metrics
        metrics = self._calculate_metrics(np.array(all_targets), np.array(all_predictions))

        return {
            'val_loss': avg_loss,
            **metrics
        }

    def _calculate_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate trading-specific metrics."""

        if self.model_config.output_mode == "regression":
            # Regression metrics
            mse = np.mean((targets - predictions) ** 2)
            mae = np.mean(np.abs(targets - predictions))

            # Trading metrics
            returns = targets  # Assuming targets are returns
            sharpe = self.metrics.calculate_sharpe_ratio(returns, predictions)
            max_dd = self.metrics.calculate_max_drawdown(returns)

            return {
                'mse': mse,
                'mae': mae,
                'sharpe': sharpe,
                'max_drawdown': max_dd
            }

        else:  # classification
            # Classification metrics
            accuracy = accuracy_score(targets, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

    def train_model(self, X: np.ndarray, y: pd.Series, model: Optional[nn.Module] = None,
                   save_path: Optional[str] = None) -> nn.Module:
        """Train LSTM model."""

        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(X, y)

        # Create model if not provided
        if model is None:
            model = MultiTimeframeLSTM(self.model_config).to(self.device)

        # Setup loss function
        criterion = get_loss_function(
            self.config.loss_function,
            label_smoothing=self.config.label_smoothing if self.config.loss_function == "cross_entropy" else 0
        )

        # Setup optimizer
        optimizer = self._get_optimizer(model.parameters())

        # Setup scheduler
        scheduler = self._get_scheduler(optimizer)

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode='min' if self.model_config.output_mode == "regression" else 'max'
        )

        # Training loop
        best_model_state = None
        best_score = float('inf') if self.model_config.output_mode == "regression" else 0

        for epoch in range(self.config.num_epochs):
            # Train
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, scheduler)

            # Validate
            val_metrics = self.validate_epoch(model, val_loader, criterion)

            # Combined metrics
            epoch_metrics = {**train_metrics, **val_metrics}

            # Log metrics
            metrics_logger.log_model_metrics(
                epoch=epoch,
                train_loss=train_metrics['train_loss'],
                val_loss=val_metrics['val_loss'],
                metrics=val_metrics
            )

            # Check early stopping
            current_score = val_metrics['val_loss']
            if early_stopping(current_score):
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Save best model
            if (self.model_config.output_mode == "regression" and current_score < best_score) or \
               (self.model_config.output_mode == "classification" and current_score > best_score):

                best_score = current_score
                best_model_state = model.state_dict().copy()

                if save_path:
                    save_model(model, f"{save_path}_best.pth")

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Final evaluation on test set
        test_metrics = self.validate_epoch(model, test_loader, criterion)
        logger.info(f"Final test metrics: {test_metrics}")

        # Save final model
        if save_path:
            save_model(model, save_path)

        return model

    def _get_optimizer(self, parameters) -> optim.Optimizer:
        """Get optimizer based on configuration."""

        if self.config.optimizer == "adam":
            return optim.Adam(parameters, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "adamw":
            return optim.AdamW(parameters, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "sgd":
            return optim.SGD(parameters, lr=self.config.learning_rate, weight_decay=self.config.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _get_scheduler(self, optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler."""

        if self.config.scheduler == "cosine_annealing":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif self.config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        else:
            return None

    def objective(self, trial: optuna.Trial, X: np.ndarray, y: pd.Series) -> float:
        """Optuna objective function for hyperparameter optimization."""

        # Suggest hyperparameters
        hidden_size = trial.suggest_int('hidden_size', 32, 256)
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

        # Create model config for this trial
        trial_model_config = ModelConfig(
            input_size=self.model_config.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=self.model_config.sequence_length,
            output_mode=self.model_config.output_mode,
            num_classes=self.model_config.num_classes,
            architecture=self.model_config.architecture,
            fusion_strategy=self.model_config.fusion_strategy
        )

        trial_training_config = TrainingConfig(
            batch_size=self.config.batch_size,
            num_epochs=self.config.num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_function=self.config.loss_function,
            label_smoothing=self.config.label_smoothing,
            optimizer=self.config.optimizer,
            scheduler=self.config.scheduler,
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )

        # Create trainer with trial config
        trainer = LSTMTrainer(trial_training_config, trial_model_config)

        # Prepare data
        train_loader, val_loader, _ = trainer.prepare_data(X, y)

        # Create model
        model = MultiTimeframeLSTM(trial_model_config).to(self.device)

        # Setup training components
        criterion = get_loss_function(trial_training_config.loss_function)
        optimizer = trainer._get_optimizer(model.parameters())
        scheduler = trainer._get_scheduler(optimizer)

        # Simple training loop for optimization
        model.train()
        for epoch in range(min(20, self.config.num_epochs)):  # Shorter for optimization
            trainer.train_epoch(model, train_loader, criterion, optimizer, scheduler)

        # Evaluate on validation set
        val_metrics = trainer.validate_epoch(model, val_loader, criterion)

        # Return validation loss (minimize for regression, maximize for classification)
        if self.model_config.output_mode == "regression":
            return val_metrics['val_loss']
        else:
            return -val_metrics['val_loss']  # Negative for maximization

    def optimize_hyperparameters(self, X: np.ndarray, y: pd.Series,
                               n_trials: int = 50, timeout: int = 3600) -> Dict:
        """Run Optuna hyperparameter optimization."""

        # Set seed for reproducibility
        set_seed(42)

        # Create study
        study = optuna.create_study(
            direction='minimize' if self.model_config.output_mode == "regression" else 'maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Run optimization
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=n_trials,
            timeout=timeout
        )

        # Log best parameters
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best parameters: {study.best_params}")

        # Save study
        study_path = Path("models/optuna_study.pkl")
        study_path.parent.mkdir(parents=True, exist_ok=True)

        import joblib
        joblib.dump(study, study_path)

        return study.best_params

def train_model(X: np.ndarray, y: pd.Series, config_path: Optional[str] = None,
               optuna_trials: int = 0, save_path: Optional[str] = None) -> nn.Module:
    """Train LSTM model with optional hyperparameter optimization."""

    # Load configuration
    if config_path:
        from ..utils.config import load_config
        config = load_config(config_path)
        training_config = config.training
        model_config = config.model
    else:
        training_config = get_config().training
        model_config = get_config().model

    # Create trainer
    trainer = LSTMTrainer(training_config, model_config)

    if optuna_trials > 0:
        logger.info(f"Running hyperparameter optimization with {optuna_trials} trials")

        # Run optimization
        best_params = trainer.optimize_hyperparameters(X, y, optuna_trials)

        # Update model config with best parameters
        model_config = ModelConfig(
            input_size=model_config.input_size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout'],
            sequence_length=model_config.sequence_length,
            output_mode=model_config.output_mode,
            num_classes=model_config.num_classes,
            architecture=model_config.architecture,
            fusion_strategy=model_config.fusion_strategy
        )

        training_config = TrainingConfig(
            batch_size=training_config.batch_size,
            num_epochs=training_config.num_epochs,
            learning_rate=best_params['learning_rate'],
            weight_decay=best_params['weight_decay'],
            loss_function=training_config.loss_function,
            label_smoothing=training_config.label_smoothing,
            optimizer=training_config.optimizer,
            scheduler=training_config.scheduler,
            patience=training_config.patience,
            min_delta=training_config.min_delta
        )

        # Create new trainer with optimized config
        trainer = LSTMTrainer(training_config, model_config)

    # Train model
    logger.info("Training final model")
    model = trainer.train_model(X, y, save_path=save_path)

    return model
