import csv
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional


class DatasetLogger:
    """
    Logger leve (sem numpy) para coletar dataset no Webots:
    - salva frames da câmera via camera.saveImage(...)
    - salva ranges do lidar em CSV
    - salva metadados/ações em JSONL
    """

    def __init__(
        self,
        base_dir: str,
        run_name: str,
        config_obj: Optional[Any] = None,
    ):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{run_name}-{ts}"
        self.root = os.path.join(base_dir, self.run_id)
        self.img_dir = os.path.join(self.root, "images")
        self.lidar_dir = os.path.join(self.root, "lidar")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lidar_dir, exist_ok=True)

        self.meta_path = os.path.join(self.root, "samples.jsonl")
        self._meta_fp = open(self.meta_path, "a", encoding="utf-8")

        self.config_obj = config_obj
        if config_obj is not None:
            with open(os.path.join(self.root, "config.json"), "w", encoding="utf-8") as fp:
                json.dump(asdict(config_obj), fp, ensure_ascii=False, indent=2)

        self.sample_idx = 0

    def close(self):
        try:
            self._meta_fp.close()
        except Exception:
            pass

    def log_sample(
        self,
        camera,
        lidar_ranges: List[float],
        vx: float,
        vy: float,
        omega: float,
        timestep_ms: int,
        image_quality: int = 90,
        extra: Optional[Dict[str, Any]] = None,
    ):
        idx = self.sample_idx
        self.sample_idx += 1

        img_name = f"frame_{idx:06d}.png"
        img_path = os.path.join(self.img_dir, img_name)
        # Webots Camera: saveImage(filename, quality)
        camera.saveImage(img_path, image_quality)

        lidar_name = f"lidar_{idx:06d}.csv"
        lidar_path = os.path.join(self.lidar_dir, lidar_name)
        with open(lidar_path, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(lidar_ranges)

        rec: Dict[str, Any] = {
            "i": idx,
            "timestep_ms": timestep_ms,
            "image": os.path.join("images", img_name),
            "lidar": os.path.join("lidar", lidar_name),
            "action": {"vx": vx, "vy": vy, "omega": omega},
        }
        if extra:
            rec["extra"] = extra

        self._meta_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._meta_fp.flush()


