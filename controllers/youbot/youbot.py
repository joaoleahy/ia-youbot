from controller import Robot

import os
import sys
import json
from datetime import datetime
from pathlib import Path

from base import Base
from arm import Arm
from gripper import Gripper

from ai.config import AiConfig, DatasetConfig
from ai.features import downsample_lidar, rgb_histogram
from ai.fuzzy import FuzzyAvoider
from ai.lidar_processing import sanitize_ranges, sector_mins, sector_mins_5
from ai.logger import DatasetLogger
from ai.vision import detect_cube_rgb, detect_bin_rgb

class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.time_step)

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.time_step)
        
        
    def run(self):
        raise NotImplementedError("This method should be implemented")


class YouBotAI(YouBotController):
    """
    Controlador inicial:
    - modo collect: varre a arena e salva dataset (câmera + lidar + ações)
    - modo run (default): navegação reativa (fuzzy) + detecção simples de cubo por cor
    """

    def __init__(self, cfg: AiConfig):
        super().__init__()
        # Se o usuário passar controllerArgs ["collect"], liga dataset automaticamente.
        self.cfg = cfg

        self.step_count = 0
        self.mode = self._parse_mode()

        # ===== MCP Bridge (telemetria/controle via webots-youbot-mcp) =====
        # Path do repo: controllers/youbot/youbot.py -> ../../
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self._mcp_root = os.path.join(repo_root, "webots-youbot-mcp")
        self.mcp = None
        self._mcp_data_dir = None
        self._mcp_status_path = None
        self._mcp_log_path = None
        try:
            if os.path.isdir(self._mcp_root):
                sys.path.insert(0, self._mcp_root)
                from mcp_bridge import MCPBridge  # type: ignore

                self._mcp_data_dir = Path(self._mcp_root) / "data"
                self._mcp_status_path = self._mcp_data_dir / "status.json"
                self._mcp_log_path = self._mcp_data_dir / "logs" / "controller.log"
                self.mcp = MCPBridge(self.robot, data_dir=self._mcp_data_dir)
                # Força escrita a cada chamada (útil p/ debug)
                try:
                    self.mcp._update_interval = 1  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            self.mcp = None

        # “Assinatura”/heartbeat: se o bridge não subir, ainda assim escrevemos status/log
        self._debug_source = "ia-youbot"
        # checagem de devices principais
        if self.camera is None:
            self._write_debug_log("ERROR: camera device not found (name='camera').")
        if self.lidar is None:
            self._write_debug_log("ERROR: lidar device not found (name='lidar').")
        self._write_debug_log(f"BOOT mode={self.mode} mcp_root={self._mcp_root} mcp={'ok' if self.mcp else 'none'}")
        self._write_debug_status({"event": "boot"})

        self.avoider = FuzzyAvoider(
            cruise_vx=cfg.motion.cruise_vx,
            max_vx=cfg.motion.max_vx,
            max_omega=cfg.motion.max_omega,
        )

        self.logger = None
        dataset_enabled = cfg.dataset.enabled or (self.mode == "collect")
        if dataset_enabled:
            # Se o MCP estiver presente, grava dataset dentro do data/ do MCP para ficar fácil de achar
            out_dir = cfg.dataset.out_dir
            if self._mcp_data_dir:
                out_dir = str(self._mcp_data_dir / "datasets")
            self.logger = DatasetLogger(
                base_dir=out_dir,
                run_name=cfg.dataset.run_name,
                config_obj=cfg,
            )
            self._write_debug_log(f"dataset enabled: out_dir={out_dir} run={cfg.dataset.run_name}")

        # Estado mínimo
        self.target_color = None
        self.last_seen_steps = 999999

        # Autoteste de movimento nos primeiros segundos
        self._kick_steps = int(2.0 * 1000 / max(1, self.time_step))  # ~2s
        self._recovery_steps = 0
        self._turn_steps = 0
        self._pinch_steps = 0
        self._escape_steps = 0
        self._stuck_counter = 0

        # Detecção de "stuck in circle" (mesmo omega por muito tempo)
        self._same_omega_direction_steps = 0
        self._last_omega_sign = 0

        # Direção de giro FIXADA durante recovery (evita oscilação)
        self._recovery_turn_left = True
        # Direção FIXADA durante escape
        self._escape_prefer_left = True
        # Contador de recoveries consecutivos (força escape após muitos)
        self._consecutive_recoveries = 0

        # prepara braço/garra para não bater
        try:
            self.arm.reset()
            self.gripper.release()
        except Exception:
            pass

    def _parse_mode(self) -> str:
        """
        Webots permite controller arguments (controllerArgs no .wbt).
        Aqui aceitamos:
        - "collect" => coleta dataset
        - "run" (default) => execução
        """
        try:
            args = self.robot.getControllerArguments()
        except Exception:
            args = []
        if args and len(args) >= 1 and args[0].strip():
            return args[0].strip().lower()
        return "run"

    def _write_debug_log(self, msg: str):
        # tenta via MCPBridge (se existir) e sempre tenta arquivo direto
        try:
            if self.mcp:
                self.mcp.log(f"[{self._debug_source}] {msg}")
        except Exception:
            pass
        try:
            if self._mcp_log_path:
                self._mcp_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._mcp_log_path, "a", encoding="utf-8") as fp:
                    fp.write(f"[{datetime.now().strftime('%H:%M:%S')}] [{self._debug_source}] {msg}\n")
        except Exception:
            pass

    def _write_debug_status(self, extra_fields: dict):
        try:
            if self._mcp_status_path:
                self._mcp_status_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "timestamp": datetime.now().isoformat(),
                    "source": self._debug_source,
                    **extra_fields,
                }
                with open(self._mcp_status_path, "w", encoding="utf-8") as fp:
                    json.dump(payload, fp, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _collect_policy(self) -> tuple[float, float, float]:
        """
        Trajetória de coleta simples: alterna avanço e giro para cobrir o mapa.
        (Depois a gente pode trocar por varredura em grade / wall-following.)
        """
        # varredura simples em “zig-zag”:
        # - anda reto por alguns segundos
        # - gira ~90°
        # - anda reto e repete (cobre bem o mapa fixo com pouco código)
        straight_s = 5.0
        turn_s = 1.25
        steps_straight = int(straight_s * 1000 / max(1, self.time_step))
        steps_turn = int(turn_s * 1000 / max(1, self.time_step))
        block = steps_straight + steps_turn
        phase = self.step_count % block
        if phase < steps_straight:
            return (self.cfg.motion.cruise_vx, 0.0, 0.0)
        # gira alternando direção para não “derivar” sempre pro mesmo lado
        sign = -1.0 if ((self.step_count // block) % 2 == 0) else 1.0
        return (0.0, 0.0, sign * 0.75 * self.cfg.motion.max_omega)

    def _run_policy(self, lidar_ranges) -> tuple[float, float, float, dict]:
        """
        Política online inicial:
        - se um cubo é visto: tenta centralizar (giro) e avançar
        - se não: navega por evitamento fuzzy (lidar)
        """
        bin_det = detect_bin_rgb(
            self.camera,
            step=self.cfg.vision.step,
            min_bin_pixels=self.cfg.vision.min_bin_pixels,
            r_min=self.cfg.vision.r_min,
            g_min=self.cfg.vision.g_min,
            b_min=self.cfg.vision.b_min,
            other_max=self.cfg.vision.other_max,
        )
        det = detect_cube_rgb(
            self.camera,
            step=self.cfg.vision.step,
            min_blob_pixels=self.cfg.vision.min_blob_pixels,
            max_cube_pixels=self.cfg.vision.max_cube_pixels,
            r_min=self.cfg.vision.r_min,
            g_min=self.cfg.vision.g_min,
            b_min=self.cfg.vision.b_min,
            other_max=self.cfg.vision.other_max,
        )

        extra = {"cube": {"found": det.found, "color": det.color, "cx": det.cx, "area": det.area}}
        extra["bin"] = (
            {"found": True, "color": bin_det.color, "cx": bin_det.cx, "area": bin_det.area}
            if bin_det
            else {"found": False}
        )

        # sempre calcula evitamento (segurança)
        clean = sanitize_ranges(lidar_ranges, self.cfg.lidar.clip_min, self.cfg.lidar.clip_max)
        secs = sector_mins(clean, front_frac=self.cfg.lidar.front_frac, side_frac=self.cfg.lidar.side_frac)
        secs5 = sector_mins_5(clean)
        avoid = self.avoider(secs.left_min, secs.front_min, secs.right_min)
        extra["lidar"] = {
            "left_min": secs.left_min,
            "front_min": secs.front_min,
            "right_min": secs.right_min,
            "front_left_min": secs5.front_left_min,
            "front_right_min": secs5.front_right_min,
        }

        # ===== Controle "car-like": SEM strafe (vy = 0) =====
        # Recovery: se encostou/travou muito perto, recua e gira para o lado mais livre
        min_side = min(secs.left_min, secs.right_min)
        min_front_flanks = min(secs5.front_left_min, secs5.front_right_min)

        # ===== Anti-loop: se ficar preso em canto/corredor repetidamente, faz um escape mais agressivo =====
        # Heurística: "travado" quando está muito perto de paredes/obstáculos e não abre espaço nos flancos.
        very_close = (min_side < 0.22 and (secs.front_min < 0.45 or min_front_flanks < 0.35))
        cornerish = (secs.front_min < 0.40 and secs5.front_left_min < 0.40 and secs5.front_right_min < 0.40)
        if very_close or cornerish:
            self._stuck_counter += 1
        else:
            self._stuck_counter = max(0, self._stuck_counter - 2)

        if self._escape_steps > 0:
            self._escape_steps -= 1
            # Usa direção FIXADA no início do escape (evita oscilação)
            prefer_left = self._escape_prefer_left
            
            # Fases do escape (3s total):
            total = int(3.0 * 1000 / max(1, self.time_step))
            phase = self._escape_steps
            third = total // 3
            
            if phase > 2 * third:
                # Fase 1: recua rápido e reto
                vx = -0.15
                vy = 0.0
                omega = 0.0
            elif phase > third:
                # Fase 2: STRAFE forte para o lado livre + giro
                vx = -0.04
                vy = 0.15 if prefer_left else -0.15
                omega = (0.6 * self.cfg.motion.max_omega) * (1.0 if prefer_left else -1.0)
            else:
                # Fase 3: continua strafe + gira forte e avança
                vx = 0.06
                vy = 0.08 if prefer_left else -0.08
                omega = (1.0 * self.cfg.motion.max_omega) * (1.0 if prefer_left else -1.0)
            
            return (vx, vy, omega, {**extra, "escape": True})

        # Se ficou "travado" por ~1.5s, dispara escape
        if self._stuck_counter > int(1.5 * 1000 / max(1, self.time_step)):
            self._stuck_counter = 0
            self._escape_steps = int(3.0 * 1000 / max(1, self.time_step))
            # FIXA direção no início do escape
            self._escape_prefer_left = secs.left_min >= secs.right_min
            omega = (1.0 * self.cfg.motion.max_omega) * (1.0 if self._escape_prefer_left else -1.0)
            return (-0.15, 0.0, omega, {**extra, "escape": True})

        # Detecção de "pinch": preso entre dois obstáculos laterais (ou parede+caixote)
        # Sinal típico: left e right baixos e parecidos, enquanto frente não está tão baixa.
        side_pinch = (secs.left_min < 0.28 and secs.right_min < 0.28 and abs(secs.left_min - secs.right_min) < 0.06)
        if self._pinch_steps > 0:
            self._pinch_steps -= 1
            # manobra: recua um pouco e gira para o lado com mais folga nos flancos frontais
            turn_left = secs5.front_left_min >= secs5.front_right_min
            omega = (0.9 * self.cfg.motion.max_omega) * (1.0 if turn_left else -1.0)
            return (-0.09, 0.0, omega, {**extra, "pinch": True})
        if side_pinch and (secs.front_min < 0.75 or min_front_flanks < 0.65):
            # inicia escape por ~1.4s
            self._pinch_steps = int(1.4 * 1000 / max(1, self.time_step))
            turn_left = secs5.front_left_min >= secs5.front_right_min
            omega = (0.9 * self.cfg.motion.max_omega) * (1.0 if turn_left else -1.0)
            return (-0.09, 0.0, omega, {**extra, "pinch": True})

        # Estado de giro "turn in place" (histerese) para sair de quinas sem oscilar
        if self._turn_steps > 0:
            self._turn_steps -= 1
            turn_left = secs.left_min >= secs.right_min
            omega = (0.75 * self.cfg.motion.max_omega) * (1.0 if turn_left else -1.0)
            return (0.0, 0.0, omega, {**extra, "turning": True})

        # Detecta quina: frente e ambos flancos perto -> gira no lugar até abrir
        if secs.front_min < 0.30 and secs5.front_left_min < 0.30 and secs5.front_right_min < 0.30:
            self._turn_steps = int(1.0 * 1000 / max(1, self.time_step))
            turn_left = secs.left_min >= secs.right_min
            omega = (0.85 * self.cfg.motion.max_omega) * (1.0 if turn_left else -1.0)
            return (0.0, 0.0, omega, {**extra, "turning": True})
        if self._recovery_steps > 0:
            self._recovery_steps -= 1
            # Usa direção FIXADA no início do recovery (evita oscilação)
            omega = (0.85 * self.cfg.motion.max_omega) * (1.0 if self._recovery_turn_left else -1.0)
            return (-0.10, 0.0, omega, {**extra, "recovery": True})

        # Threshold mais conservador: min_side < 0.22 (era 0.17)
        if secs.front_min < 0.28 or min_side < 0.22 or min_front_flanks < 0.22:
            # Fixa direção de giro NO INÍCIO do recovery
            self._recovery_turn_left = secs.left_min >= secs.right_min
            # Recovery mais longo: 2.0s (era 1.2s)
            self._recovery_steps = int(2.0 * 1000 / max(1, self.time_step))
            self._consecutive_recoveries += 1
            
            # Se já tentou vários recoveries sem sucesso, força escape mais agressivo
            if self._consecutive_recoveries >= 3:
                self._escape_steps = int(3.0 * 1000 / max(1, self.time_step))
                self._consecutive_recoveries = 0
                return (-0.12, 0.0, (1.0 * self.cfg.motion.max_omega) * (1.0 if self._recovery_turn_left else -1.0), {**extra, "escape": True})
            
            omega = (0.85 * self.cfg.motion.max_omega) * (1.0 if self._recovery_turn_left else -1.0)
            return (-0.10, 0.0, omega, {**extra, "recovery": True})
        else:
            # Resetar contador quando não está em situação de recovery
            self._consecutive_recoveries = max(0, self._consecutive_recoveries - 1)

        # IMPORTANT: se estamos vendo uma caixa grande (bin), NÃO tratar como cubo/alvo de perseguição.
        if det.found:
            self.last_seen_steps = 0
            self.target_color = det.color

            # erro horizontal: cx<0.5 => cubo à esquerda
            err = (det.cx - 0.5)
            omega_track = -err * self.cfg.motion.max_omega * 1.2
            # avança mais se estiver “centralizado” e frente não muito perto
            vx_track = self.cfg.motion.cruise_vx if abs(err) < 0.18 else 0.06

            # steering de parede (mantém distância lateral) + tracking do cubo
            omega_wall = 1.6 * (secs.left_min - secs.right_min)
            omega_wall = max(-self.cfg.motion.max_omega, min(self.cfg.motion.max_omega, omega_wall))

            side_near = min_side < 0.40
            front_near = secs.front_min < 0.55

            # se está muito perto de algo, evita mais; senão, segue cubo mais
            w = 0.80 if (side_near or front_near) else 0.35
            omega = (1.0 - w) * omega_track + w * omega_wall

            # velocidade: base no avoid.vx, mas limita pelo tracking
            vx = min(vx_track, max(0.02, avoid.vx))

            # saturação
            omega = max(-self.cfg.motion.max_omega, min(self.cfg.motion.max_omega, omega))
            vx = max(0.0, min(self.cfg.motion.max_vx, vx))
            return (vx, 0.0, omega, extra)

        # Sem cubo: “explore” reativo
        self.last_seen_steps += 1
        # em exploração: manter distância lateral (como um carro em corredor)
        omega = 1.6 * (secs.left_min - secs.right_min)
        # desvio de frente
        if secs.front_min < 0.60:
            turn_left = secs.left_min >= secs.right_min
            omega += (0.8 * self.cfg.motion.max_omega) * (1.0 if turn_left else -1.0) * (0.60 - secs.front_min) / 0.60
        # ajuda em quina: se flancos frontais ficam perto, gira mais para o lado com mais espaço
        if min_front_flanks < 0.45:
            turn_left = secs5.front_left_min >= secs5.front_right_min
            omega += (0.45 * self.cfg.motion.max_omega) * (1.0 if turn_left else -1.0) * (0.45 - min_front_flanks) / 0.45

        # Repulsão visual das caixas (bins) quando não estamos perseguindo cubo:
        # evita que o robô "cole" em uma caixa grande e acabe na parede.
        if extra.get("bin", {}).get("found"):
            cx = float(extra["bin"].get("cx", 0.5))
            # bin à esquerda -> virar para direita (omega negativo), e vice-versa
            away = (cx - 0.5)
            omega += (+1.0 if away > 0 else -1.0) * 0.45 * self.cfg.motion.max_omega * (min(1.0, abs(away) * 2.0))
            # desacelera perto de bin
            avoid_vx = max(0.02, avoid.vx) * 0.6
        else:
            avoid_vx = max(0.02, avoid.vx)

        omega = max(-self.cfg.motion.max_omega, min(self.cfg.motion.max_omega, omega))

        vx = avoid_vx
        # Suavizado: threshold 0.50->0.40, multiplier 0.6->0.7
        if secs.front_min < 0.40:
            vx *= 0.7
        if secs.front_min < 0.30:
            vx *= 0.4

        # Garante vx mínimo de exploração quando não há obstáculo próximo
        if secs.front_min > 0.55 and min_side > 0.30:
            vx = max(vx, 0.08)

        vx = max(0.0, min(self.cfg.motion.max_vx, vx))

        # ===== Detecção de "stuck in circle" =====
        # Se omega tem mesmo sinal por muito tempo (~3s), inverte para quebrar ciclo
        current_sign = 1 if omega > 0.05 else (-1 if omega < -0.05 else 0)
        if current_sign != 0 and current_sign == self._last_omega_sign:
            self._same_omega_direction_steps += 1
        else:
            self._same_omega_direction_steps = 0
        self._last_omega_sign = current_sign

        if self._same_omega_direction_steps > int(3.0 * 1000 / max(1, self.time_step)):
            omega = -omega * 0.5  # inverte e suaviza
            self._same_omega_direction_steps = 0
            extra["circle_break"] = True

        return (vx, 0.0, omega, extra)

    def run(self):
        try:
            while self.robot.step(self.time_step) != -1:
                self.step_count += 1

                # sensores
                try:
                    lidar_ranges = self.lidar.getRangeImage()
                except Exception as e:
                    self._write_debug_log(f"ERROR lidar.getRangeImage(): {e}")
                    lidar_ranges = []
                # sempre define 'clean' para dataset/diagnóstico (evita NameError)
                clean = sanitize_ranges(lidar_ranges, self.cfg.lidar.clip_min, self.cfg.lidar.clip_max)

                if self.mode == "collect":
                    vx, vy, omega = self._collect_policy()
                    extra = {"mode": "collect"}
                else:
                    # “Kick” inicial para provar que a base responde
                    if self.step_count <= self._kick_steps:
                        vx, vy, omega = (0.20, 0.0, 0.0)
                        extra = {"mode": "run", "kick": True}
                    else:
                        vx, vy, omega, extra = self._run_policy(lidar_ranges)
                    extra["mode"] = "run"

                # comando
                self.base.move(vx, vy, omega)

                # MCP: publish + câmera
                if self.mcp:
                    try:
                        self.mcp.publish(
                            {
                                "mode": self.mode,
                                "step": self.step_count,
                                "action": {"vx": vx, "vy": vy, "omega": omega},
                                "extra": extra,
                                "source": self._debug_source,
                                "wheels": getattr(self.base, "last_wheel_speeds", None),
                            }
                        )
                        # tenta salvar via PIL (se disponível), mas não depende disso
                        self.mcp.save_camera_frame(self.camera)
                        # lê comandos (p/ futuras extensões; sim control só funciona em Supervisor)
                        self.mcp.get_command()
                    except Exception:
                        pass
                else:
                    # Mesmo sem bridge, atualiza status para o MCP server não ficar preso em dados velhos
                    if self.step_count % 5 == 0:
                        self._write_debug_status(
                            {
                                "mode": self.mode,
                                "step": self.step_count,
                                "action": {"vx": vx, "vy": vy, "omega": omega},
                                "extra": extra,
                                "wheels": getattr(self.base, "last_wheel_speeds", None),
                            }
                        )

                # Salva frame SEM Pillow (direto via Webots camera.saveImage)
                # Isso faz o tool webots_get_camera funcionar mesmo se o Python do Webots não tiver PIL.
                if self._mcp_data_dir and self.camera and (self.step_count % 5 == 0):
                    try:
                        cam_dir = self._mcp_data_dir / "camera"
                        cam_dir.mkdir(parents=True, exist_ok=True)
                        frame_num = (self.step_count // 5) % 50
                        out_path = cam_dir / f"frame_{frame_num:04d}.png"
                        self.camera.saveImage(str(out_path), 90)
                    except Exception:
                        pass

                # dataset
                if self.logger and (self.step_count % max(1, self.cfg.dataset.save_every_n_steps) == 0):
                    # features para place recognition (cenário fixo)
                    feats = {
                        "lidar_ds_60": downsample_lidar(clean, 60),
                        "rgb_hist": rgb_histogram(self.camera, step=4, bins=8).as_dict(),
                    }
                    self.logger.log_sample(
                        camera=self.camera,
                        lidar_ranges=list(lidar_ranges),
                        vx=vx,
                        vy=vy,
                        omega=omega,
                        timestep_ms=self.time_step,
                        image_quality=self.cfg.dataset.image_quality,
                        extra={**extra, "features": feats},
                    )

                # log "batimento" a cada ~1s - agora com mais info para debug
                if self.step_count % max(1, int(1000 / max(1, self.time_step))) == 0:
                    flags = []
                    if extra.get("turning"): flags.append("turn")
                    if extra.get("recovery"): flags.append("rec")
                    if extra.get("escape"): flags.append("esc")
                    if extra.get("pinch"): flags.append("pinch")
                    if extra.get("circle_break"): flags.append("circ")
                    flags_str = ",".join(flags) if flags else "-"
                    self._write_debug_log(f"tick step={self.step_count} vx={vx:.3f} omega={omega:.3f} flags={flags_str}")
        finally:
            if self.logger:
                self.logger.close()

if __name__ == "__main__":
    # Config padrão:
    # - para coletar dataset, altere controllerArgs no .wbt para ["collect"] e dataset.enabled=True
    cfg = AiConfig(
        dataset=DatasetConfig(enabled=False, out_dir="data", run_name="dataset", save_every_n_steps=2),
    )
    YouBotAI(cfg).run()