import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import AIEModel
from layers import LayerNormLayer


def build_and_run(seed: int = 0):
    rng = np.random.default_rng(seed)

    batch = 160
    features = 64

    x = rng.integers(-128, 128, size=(batch, features), dtype=np.int8)

    model = AIEModel(m=4, k=8, n=8, iterations=1, dynamic_quant=False)

    layernorm = LayerNormLayer(
        name='layernorm_0',
        gamma=np.ones(features, dtype=np.float32),
        beta=np.zeros(features, dtype=np.float32),
        output_scale=32,
    )
    model.add_layer(layernorm, inputs=[None])

    y = model.forward(x)
    print(f"\nModel completed. Output shape: {y.shape}")
    return y


if __name__ == "__main__":
    build_and_run()
