import math
from typing import List, Optional

import numpy as np

from .base import AIELayer


def _round_div_signed(num: int, den: int) -> int:
    if den <= 0:
        raise ValueError(f"denominator must be positive, got {den}")
    if num >= 0:
        return (num + den // 2) // den
    return -((-num + den // 2) // den)


def _round_shift_signed(num: int, shift: int) -> int:
    if shift <= 0:
        return num
    return _round_div_signed(num, 1 << shift)


class LayerNormLayer(AIELayer):
    """
    Integer row-wise LayerNorm for int8 AIE streams.

    The layer normalizes each row over the feature dimension and emits int8:
        y ~= round(((x - mean) / std) * output_scale)

    Optional affine parameters follow the normalized value:
        y = round(y * gamma / 2**affine_shift) + beta

    Floating-point gamma values are quantized by 2**affine_shift. Floating-point
    beta values are quantized by output_scale. Integer gamma/beta arrays are
    treated as already-quantized values.
    """

    def __init__(
        self,
        name: str,
        gamma: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        output_scale: int = 32,
        eps: int = 1,
        affine_shift: int = 8,
        sqrt_features_shift: int = 12,
        norm_shift: int = 16,
    ):
        super().__init__(name, 'layernorm', params={
            'gamma': gamma,
            'beta': beta,
            'output_scale': int(output_scale),
            'eps': int(eps),
            'affine_shift': int(affine_shift),
            'sqrt_features_shift': int(sqrt_features_shift),
            'norm_shift': int(norm_shift),
        })

        if output_scale <= 0:
            raise ValueError(f"LayerNormLayer {name}: output_scale must be positive")
        if eps < 0:
            raise ValueError(f"LayerNormLayer {name}: eps must be non-negative")
        if affine_shift < 0:
            raise ValueError(f"LayerNormLayer {name}: affine_shift must be non-negative")
        if sqrt_features_shift < 0:
            raise ValueError(f"LayerNormLayer {name}: sqrt_features_shift must be non-negative")
        if norm_shift < 0:
            raise ValueError(f"LayerNormLayer {name}: norm_shift must be non-negative")

        self.output_scale = int(output_scale)
        self.eps = int(eps)
        self.affine_shift = int(affine_shift)
        self.sqrt_features_shift = int(sqrt_features_shift)
        self.norm_shift = int(norm_shift)

        self._gamma_source = None if gamma is None else np.asarray(gamma)
        self._beta_source = None if beta is None else np.asarray(beta)
        self.gamma = None
        self.beta = None

        self.m = None
        self.k = None
        self.n = None

    @property
    def has_affine(self) -> bool:
        return self._gamma_source is not None or self._beta_source is not None

    def _prepare_affine(self, features: int) -> None:
        if not self.has_affine:
            self.gamma = None
            self.beta = None
            return

        if self._gamma_source is None:
            gamma = np.full(features, 1 << self.affine_shift, dtype=np.int16)
        else:
            if self._gamma_source.shape != (features,):
                raise ValueError(
                    f"LayerNormLayer {self.name}: gamma shape must be ({features},), "
                    f"got {self._gamma_source.shape}"
                )
            if np.issubdtype(self._gamma_source.dtype, np.floating):
                gamma = np.rint(self._gamma_source * (1 << self.affine_shift))
            else:
                gamma = self._gamma_source
            gamma = np.clip(gamma, -32768, 32767).astype(np.int16)

        if self._beta_source is None:
            beta = np.zeros(features, dtype=np.int16)
        else:
            if self._beta_source.shape != (features,):
                raise ValueError(
                    f"LayerNormLayer {self.name}: beta shape must be ({features},), "
                    f"got {self._beta_source.shape}"
                )
            if np.issubdtype(self._beta_source.dtype, np.floating):
                beta = np.rint(self._beta_source * self.output_scale)
            else:
                beta = self._beta_source
            beta = np.clip(beta, -32768, 32767).astype(np.int16)

        self.gamma = gamma
        self.beta = beta

    def _compute_golden(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.validate_inputs(inputs, expected_count=1)
        x = inputs[0]

        assert self.m is not None and self.k is not None and self.n is not None, \
            "Tiling parameters not set. Layer must be added to AIEModel first."
        if x.ndim != 2:
            raise ValueError(f"LayerNormLayer {self.name}: expected 2D input, got shape {x.shape}")
        if x.shape[0] % self.m != 0:
            raise ValueError(f"LayerNormLayer {self.name}: rows {x.shape[0]} must be divisible by m={self.m}")
        if x.shape[1] % self.n != 0:
            raise ValueError(f"LayerNormLayer {self.name}: features {x.shape[1]} must be divisible by n={self.n}")

        rows, features = x.shape
        self._prepare_affine(features)

        sqrt_features_scale = int(round(math.sqrt(features) * (1 << self.sqrt_features_shift)))
        sqrt_den = 1 << self.sqrt_features_shift
        a = np.zeros((rows, features), dtype=np.int8)
        x_i32 = x.astype(np.int32)

        for row in range(rows):
            row_vals = x_i32[row]
            mean = _round_div_signed(int(np.sum(row_vals, dtype=np.int64)), features)
            centered = row_vals - mean
            var_sum = int(np.sum(centered.astype(np.int64) * centered.astype(np.int64), dtype=np.int64))
            denom = math.isqrt(var_sum + self.eps)
            if denom <= 0:
                denom = 1

            norm_factor_num = self.output_scale * sqrt_features_scale * (1 << self.norm_shift)
            norm_factor_den = denom * sqrt_den
            norm_factor = _round_div_signed(norm_factor_num, norm_factor_den)

            for col in range(features):
                norm = _round_shift_signed(int(centered[col]) * norm_factor, self.norm_shift)
                if self.has_affine:
                    norm = _round_shift_signed(int(norm) * int(self.gamma[col]), self.affine_shift)
                    norm += int(self.beta[col])
                a[row, col] = np.int8(np.clip(norm, -128, 127))

        self.outputs['x'] = x
        self.outputs['a'] = a
        self.outputs['sqrt_features_scale'] = np.array([sqrt_features_scale], dtype=np.int32)
        self._golden_computed = True
        return a

    def generate_kernel_code(self, f) -> None:
        features = self.outputs['x'].shape[1]
        t_m = self.outputs['x'].shape[0] // self.m
        t_n = features // self.n
        sqrt_features_scale = int(self.outputs['sqrt_features_scale'][0])

        f.write('#include <cstdint>\n')
        if self.has_affine:
            f.write(f'__attribute__((section(".data"))) alignas(32) int16_t gamma_p [{features}] = {{ ')
            f.write(', '.join(str(int(x)) for x in self.gamma.reshape(-1)))
            f.write(' };\n\n')
            f.write(f'__attribute__((section(".data"))) alignas(32) int16_t beta_p [{features}] = {{ ')
            f.write(', '.join(str(int(x)) for x in self.beta.reshape(-1)))
            f.write(' };\n\n')

        f.write('#include "kernels.h"\n\n')

        has_affine_str = 'true' if self.has_affine else 'false'
        f.write(f'void f{self.idx}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ ')
        f.write(
            f'layernorm<{self.m}, {self.n}, {t_m}, {t_n}, {self.output_scale}, '
            f'{self.eps}, {self.affine_shift}, {self.sqrt_features_shift}, '
            f'{sqrt_features_scale}, {self.norm_shift}, {has_affine_str}>'
        )
        if self.has_affine:
            f.write(' (x, a, gamma_p, beta_p);}\n')
        else:
            f.write(' (x, a, nullptr, nullptr);}\n')

        self._generate_include_code()

    def _generate_include_code(self) -> None:
        with open("aie/include.h", "a") as f:
            f.write(f'void f{self.idx}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')

    def generate_graph_code(self, f, input_ports: List[str]) -> None:
        self.validate_inputs(input_ports, expected_count=1)
        in_port = input_ports[0]
        f.write(f"        {self.name}[0] = kernel::create(::f{self.idx});\n")
        f.write(f'        source({self.name}[0]) = "layer_{self.idx}.cc";\n')
        f.write(f'        runtime<ratio>({self.name}[0]) = 1.0;\n')
        f.write(f"        connect<stream>({in_port}.out[0], {self.name}[0].in[0]);\n\n")

    def num_kernels(self) -> int:
        return 1

    def get_output_port(self, port_idx: int = 0) -> str:
        return f"{self.name}[0]"

    def __repr__(self) -> str:
        idx_str = f"idx={self.idx}" if self.idx is not None else "idx=unassigned"
        affine = "affine" if self.has_affine else "no_affine"
        return (
            f"LayerNormLayer({idx_str}, name='{self.name}', output_scale={self.output_scale}, "
            f"eps={self.eps}, norm_shift={self.norm_shift}, {affine})"
        )
