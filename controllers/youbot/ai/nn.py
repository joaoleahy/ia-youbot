import json
import math
from dataclasses import dataclass
from typing import List, Optional


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _sigmoid(x: float) -> float:
    # numérica simples
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class MLPWeights:
    """
    Pesos de uma MLP densa (inferência apenas), sem numpy.
    Formato:
      layers: lista de camadas; cada camada tem W e b
      W: matriz [out][in]
      b: vetor [out]
      activation: "relu" para escondidas, "sigmoid" ou "linear" na saída
    """

    layers: List[dict]

    @staticmethod
    def load_json(path: str) -> "MLPWeights":
        with open(path, "r", encoding="utf-8") as fp:
            obj = json.load(fp)
        if "layers" not in obj or not isinstance(obj["layers"], list):
            raise ValueError("Formato inválido: esperado {layers:[...]}")
        return MLPWeights(layers=obj["layers"])


class SimpleMLP:
    def __init__(self, weights: MLPWeights):
        self.weights = weights

    def forward(self, x: List[float]) -> List[float]:
        a = x
        for i, layer in enumerate(self.weights.layers):
            W = layer["W"]
            b = layer["b"]
            act = layer.get("activation", "relu")

            out: List[float] = []
            for o in range(len(W)):
                s = b[o]
                row = W[o]
                # dot
                for j in range(len(row)):
                    s += row[j] * a[j]
                out.append(s)

            if act == "relu":
                a = [_relu(v) for v in out]
            elif act == "sigmoid":
                a = [_sigmoid(v) for v in out]
            elif act == "linear":
                a = out
            else:
                raise ValueError(f"Activation desconhecida: {act}")
        return a


def dummy_obstacle_mlp(in_dim: int = 60) -> SimpleMLP:
    """
    MLP “dummy” (placeholder) que retorna 3 saídas ~0.5.
    Serve para fechar o pipeline de RNA desde já; depois trocamos pelos pesos treinados.
    """
    # 1 camada linear -> sigmoid com pesos ~0
    W = [[0.0 for _ in range(in_dim)] for _ in range(3)]
    b = [0.0, 0.0, 0.0]
    return SimpleMLP(MLPWeights(layers=[{"W": W, "b": b, "activation": "sigmoid"}]))


