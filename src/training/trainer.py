"""
Model Trainer

PyTorch 모델 학습을 위한 Trainer 클래스
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from src.config import DEVICE, MODELS_DIR, LOGS_DIR, training_config


class EarlyStopping:
    """
    Early Stopping 콜백

    검증 손실이 개선되지 않으면 학습을 조기 종료합니다.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        """
        Args:
            patience: 개선 없이 대기할 에폭 수
            min_delta: 개선으로 인정할 최소 변화량
            mode: 'min' (손실 최소화) 또는 'max' (메트릭 최대화)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class ModelCheckpoint:
    """
    모델 체크포인트 저장

    최상의 모델을 저장하고 관리합니다.
    """

    def __init__(
        self,
        save_dir: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
    ):
        """
        Args:
            save_dir: 저장 디렉토리
            monitor: 모니터링할 메트릭
            mode: 'min' 또는 'max'
            save_top_k: 저장할 최대 체크포인트 수
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_scores: List[Tuple[float, Path]] = []

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metrics: Dict[str, float],
    ) -> Optional[Path]:
        """
        체크포인트 저장

        Returns:
            저장된 경로 (저장되지 않으면 None)
        """
        score = metrics.get(self.monitor, metrics.get("val_loss", 0))

        # 저장 여부 결정
        should_save = False
        if len(self.best_scores) < self.save_top_k:
            should_save = True
        else:
            worst_idx = self._get_worst_idx()
            if self.mode == "min":
                should_save = score < self.best_scores[worst_idx][0]
            else:
                should_save = score > self.best_scores[worst_idx][0]

        if not should_save:
            return None

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_epoch{epoch:03d}_{self.monitor}{score:.4f}_{timestamp}.pt"
        filepath = self.save_dir / filename

        # 체크포인트 저장
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, filepath)

        # 최악의 체크포인트 삭제
        if len(self.best_scores) >= self.save_top_k:
            worst_idx = self._get_worst_idx()
            old_path = self.best_scores[worst_idx][1]
            if old_path.exists():
                old_path.unlink()
            self.best_scores.pop(worst_idx)

        self.best_scores.append((score, filepath))

        return filepath

    def _get_worst_idx(self) -> int:
        """최악의 점수 인덱스 반환"""
        if self.mode == "min":
            return max(range(len(self.best_scores)), key=lambda i: self.best_scores[i][0])
        else:
            return min(range(len(self.best_scores)), key=lambda i: self.best_scores[i][0])

    def get_best_checkpoint(self) -> Optional[Path]:
        """최상의 체크포인트 경로 반환"""
        if not self.best_scores:
            return None
        if self.mode == "min":
            return min(self.best_scores, key=lambda x: x[0])[1]
        else:
            return max(self.best_scores, key=lambda x: x[0])[1]


class Trainer:
    """
    모델 학습 Trainer

    PyTorch 모델의 학습, 검증, 평가를 관리합니다.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: Optional[_LRScheduler] = None,
        device: torch.device = DEVICE,
        model_name: str = "model",
        gradient_clip_val: float = 1.0,
    ):
        """
        Args:
            model: PyTorch 모델
            optimizer: 옵티마이저
            criterion: 손실 함수
            scheduler: 학습률 스케줄러 (선택적)
            device: 연산 디바이스
            model_name: 모델 이름 (저장용)
            gradient_clip_val: 그래디언트 클리핑 값
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name
        self.gradient_clip_val = gradient_clip_val

        # 로그 저장
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        # 체크포인트 디렉토리
        self.checkpoint_dir = MODELS_DIR / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """단일 에폭 학습"""
        self.model.train()
        total_loss = 0.0

        for batch_X, batch_y in tqdm(dataloader, desc="Training", leave=False):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)

            # Loss 계산
            loss = self.criterion(outputs, batch_y)

            # Backward
            loss.backward()

            # Gradient Clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val
                )

            self.optimizer.step()

            total_loss += loss.item() * batch_X.size(0)

        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_X, batch_y in tqdm(dataloader, desc="Validation", leave=False):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            total_loss += loss.item() * batch_X.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)

        # 추가 메트릭 계산
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        metrics = self._calculate_metrics(preds, targets)
        metrics["val_loss"] = avg_loss

        return avg_loss, metrics

    def _calculate_metrics(
        self, preds: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """회귀 메트릭 계산"""
        # Flatten if needed
        preds = preds.flatten()
        targets = targets.flatten()

        # MSE, RMSE
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)

        # MAE
        mae = np.mean(np.abs(preds - targets))

        # MAPE (0 division 방지)
        mask = targets != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100
        else:
            mape = 0.0

        # R²
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 1e-4,
        save_top_k: int = 3,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        모델 학습

        Args:
            train_loader: 학습 DataLoader
            val_loader: 검증 DataLoader
            epochs: 최대 에폭 수
            early_stopping_patience: Early Stopping patience
            early_stopping_min_delta: 최소 개선량
            save_top_k: 저장할 최상위 체크포인트 수
            verbose: 출력 여부

        Returns:
            학습 히스토리
        """
        # 콜백 초기화
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        )
        checkpoint = ModelCheckpoint(
            save_dir=self.checkpoint_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=save_top_k,
        )

        best_val_loss = float("inf")
        start_time = time.time()

        print(f"\n{'=' * 60}")
        print(f"Training {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # 학습
            train_loss = self.train_epoch(train_loader)

            # 검증
            val_loss, metrics = self.validate(val_loader)

            # 스케줄러 업데이트
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 히스토리 기록
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            # 메트릭 기록
            for key, value in metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)

            # 체크포인트 저장
            saved_path = checkpoint.save(self.model, self.optimizer, epoch, metrics)

            # 최상의 모델 업데이트
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improvement = "*"
            else:
                improvement = ""

            # 출력
            epoch_time = time.time() - epoch_start
            if verbose:
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"RMSE: {metrics['rmse']:.4f} | "
                    f"MAPE: {metrics['mape']:.2f}% | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s {improvement}"
                )

            # Early Stopping 체크
            if early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # 학습 완료
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Training completed in {total_time / 60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best checkpoint: {checkpoint.get_best_checkpoint()}")
        print(f"{'=' * 60}")

        # 히스토리 저장
        self._save_history()

        return self.history

    def _save_history(self):
        """학습 히스토리 저장"""
        history_path = self.checkpoint_dir / "training_history.json"

        # numpy 타입을 Python 기본 타입으로 변환
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [
                float(v) if hasattr(v, 'item') else float(v)
                for v in values
            ]

        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2)

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """예측"""
        self.model.eval()
        all_preds = []
        all_targets = []

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            outputs = self.model(batch_X)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())

        return np.concatenate(all_preds), np.concatenate(all_targets)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """테스트 데이터 평가"""
        preds, targets = self.predict(dataloader)
        metrics = self._calculate_metrics(preds, targets)

        print("\n" + "=" * 40)
        print("Evaluation Results")
        print("=" * 40)
        for key, value in metrics.items():
            if key == "mape":
                print(f"  {key.upper()}: {value:.2f}%")
            else:
                print(f"  {key.upper()}: {value:.4f}")
        print("=" * 40)

        return metrics


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    optimizer_type: str = "adam",
) -> Optimizer:
    """옵티마이저 생성"""
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 100,
    **kwargs,
) -> _LRScheduler:
    """학습률 스케줄러 생성"""
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get("step_size", 30), gamma=0.1
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=kwargs.get("patience", 10),
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


if __name__ == "__main__":
    # 테스트
    from src.models.solar_lstm import create_solar_model
    from src.training.dataset import TimeSeriesDataset, create_dataloaders

    # 더미 데이터
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    data = np.random.randn(n_samples, n_features).astype(np.float32)

    train_data = data[:700]
    val_data = data[700:850]
    test_data = data[850:]

    # DataLoader
    dataloaders = create_dataloaders(
        train_data, val_data, test_data,
        seq_length=24, batch_size=32
    )

    # 모델
    model = create_solar_model(input_size=n_features)

    # Trainer
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, epochs=10)
    criterion = nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        model_name="test_solar",
    )

    # 학습 (짧게)
    history = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        epochs=3,
        verbose=True,
    )

    # 평가
    trainer.evaluate(dataloaders["test"])
