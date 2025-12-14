from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    enabled: bool = False
    out_dir: str = "data"
    run_name: str = "run"
    save_every_n_steps: int = 2
    image_quality: int = 90  # 1..100


@dataclass(frozen=True)
class MotionConfig:
    # velocidades “no referencial do robô” (m/s e rad/s) — base.move faz a cinemática
    cruise_vx: float = 0.12
    max_vx: float = 0.20
    max_vy: float = 0.12
    max_omega: float = 0.9


@dataclass(frozen=True)
class LidarConfig:
    # distâncias em metros
    clip_min: float = 0.05
    clip_max: float = 3.5
    # setores para “reduzir” o scan (frações do vetor)
    front_frac: float = 0.20
    side_frac: float = 0.25


@dataclass(frozen=True)
class VisionConfig:
    # para a câmera 128x128, amostrar 1/2 já é bem rápido
    step: int = 2
    min_blob_pixels: int = 35
    max_cube_pixels: int = 450        # blobs muito grandes tendem a ser as caixas (bins)
    min_bin_pixels: int = 900         # blobs grandes o suficiente para considerar “caixa”
    # thresholds RGB (0..255) para cores “puras” dos cubos
    r_min: int = 170
    g_min: int = 170
    b_min: int = 170
    other_max: int = 90


@dataclass(frozen=True)
class AiConfig:
    dataset: DatasetConfig = DatasetConfig()
    motion: MotionConfig = MotionConfig()
    lidar: LidarConfig = LidarConfig()
    vision: VisionConfig = VisionConfig()


