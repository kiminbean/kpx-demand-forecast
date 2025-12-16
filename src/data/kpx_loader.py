"""
KPX Data Loader

한국전력거래소(KPX) 데이터 로딩 및 전처리
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, data_config


class KPXDataLoader:
    """
    KPX 발전량 데이터 로더

    전력거래소에서 제공하는 시간대별 발전량 데이터를 로드하고 전처리합니다.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR

    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """
        원시 데이터 로드

        Args:
            filename: 데이터 파일명

        Returns:
            로드된 DataFrame
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {filepath}")

        # 파일 확장자에 따라 로드 방식 분기
        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath, encoding="utf-8-sig")
        elif filepath.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {filepath.suffix}")

        return df

    def preprocess(
        self,
        df: pd.DataFrame,
        datetime_col: str = "datetime",
        target_col: str = "generation_mw",
    ) -> pd.DataFrame:
        """
        데이터 전처리

        Args:
            df: 원시 DataFrame
            datetime_col: 날짜시간 컬럼명
            target_col: 타겟 컬럼명

        Returns:
            전처리된 DataFrame
        """
        df = df.copy()

        # 날짜시간 파싱
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col).sort_index()

        # 결측치 처리
        missing_ratio = df[target_col].isna().mean()
        if missing_ratio > data_config.max_missing_ratio:
            print(f"Warning: 결측치 비율이 {missing_ratio:.2%}로 높습니다.")

        # 선형 보간으로 결측치 대체
        df[target_col] = df[target_col].interpolate(method="linear")

        # 이상치 처리 (IQR 방식)
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[target_col] = df[target_col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간 관련 특성 추가

        Args:
            df: 입력 DataFrame (DatetimeIndex 필요)

        Returns:
            시간 특성이 추가된 DataFrame
        """
        df = df.copy()

        # 시간 특성 (cyclical encoding)
        df["hour"] = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # 요일 특성
        df["dayofweek"] = df.index.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        # 월 특성 (cyclical encoding)
        df["month"] = df.index.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def split_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        학습/검증/테스트 데이터 분할

        시계열 데이터의 특성을 고려하여 시간순으로 분할합니다.

        Args:
            df: 입력 DataFrame

        Returns:
            (train_df, val_df, test_df) 튜플
        """
        n = len(df)
        train_end = int(n * data_config.train_ratio)
        val_end = int(n * (data_config.train_ratio + data_config.val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        print(f"Train: {len(train_df)} ({len(train_df)/n:.1%})")
        print(f"Val: {len(val_df)} ({len(val_df)/n:.1%})")
        print(f"Test: {len(test_df)} ({len(test_df)/n:.1%})")

        return train_df, val_df, test_df

    def save_processed(self, df: pd.DataFrame, filename: str) -> Path:
        """
        전처리된 데이터 저장

        Args:
            df: 저장할 DataFrame
            filename: 저장 파일명

        Returns:
            저장된 파일 경로
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.processed_dir / filename

        df.to_parquet(filepath)
        print(f"데이터 저장 완료: {filepath}")

        return filepath


if __name__ == "__main__":
    # 사용 예시
    loader = KPXDataLoader()
    print(f"Data directory: {loader.data_dir}")
