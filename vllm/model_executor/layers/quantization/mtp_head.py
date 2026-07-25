# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time quantization for the integrated MTP drafter head.

The Qwen3.5/3.6 multi-token-prediction (MTP) head is excluded from the target
model's quantization config, so its linear weights load in bf16/fp16 even when
the main model is GPTQ/FP4. This module quantizes just those linears at load
time, per output channel, reusing vLLM's existing fused Cutlass w8a8 kernels
(no python-emulated matmul in the decode hot path).

The drafter only proposes tokens; the target model verifies every one, so
quantizing the head trades a little acceptance rate (perf) for VRAM -- never
correctness. Per-output-channel scales are used to keep that acceptance cost
small. Opt in via ``VLLM_MTP_HEAD_QUANT=fp8|int8`` (default ``off``).
"""

from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_int8_linear_kernel,
)
from vllm.model_executor.kernels.linear.scaled_mm import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)

MTP_HEAD_QUANT_SCHEMES = ("fp8", "int8")


class MtpHeadFp8LinearMethod(Fp8PtpcOnlineLinearMethod):
    """Per-output-channel FP8 (e4m3) weight quant + dynamic per-token FP8
    activation for the MTP head, applied at load time.

    Mirrors :class:`Fp8PtpcOnlineLinearMethod` but does not use the meta-device
    online-reload path: the bf16 weight is loaded normally and quantized in
    ``process_weights_after_loading``. This keeps the change local to the MTP
    linears -- the rest of the model loads unchanged.
    """

    # Load the bf16 weight normally rather than materializing it just-in-time
    # on the meta device (the online-reload framework the parent uses).
    uses_meta_device = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        from vllm.model_executor.kernels.linear import init_fp8_linear_kernel

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )
        # PTPC needs per-token activation FP8; MarlinFP8 is W8A16 weight-only
        # and would silently ignore the per-token activation scale.
        if isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel):
            raise ValueError(
                "VLLM_MTP_HEAD_QUANT=fp8 requires a Cutlass FP8 kernel "
                "(SM89+ / ROCm MI3xx). This GPU only offers the W8A16 Marlin "
                "FP8 path; use VLLM_MTP_HEAD_QUANT=int8 or off instead."
            )


class MtpHeadInt8LinearMethod(UnquantizedLinearMethod):
    """Per-output-channel INT8 weight quant + dynamic per-token INT8 activation
    for the MTP head, applied at load time.

    The bf16 weight is loaded normally (inherited ``create_weights``) and
    quantized in ``process_weights_after_loading`` via the Cutlass int8 w8a8
    kernel. Symmetric dynamic per-token activation keeps accuracy close to the
    fp8 path without needing native fp8 hardware.
    """

    def __init__(self):
        super().__init__()
        self.kernel = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.logical_widths = output_partition_sizes
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        self.kernel = init_int8_linear_kernel(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Quantize bf16 weight [out, in] per output channel (per-row).
        qweight, weight_scale, _ = ops.scaled_int8_quant(
            layer.weight.contiguous(), scale=None, symmetric=True
        )
        replace_parameter(layer, "weight", qweight)
        replace_parameter(layer, "weight_scale", weight_scale)
        # Dynamic per-token activation quant -> no static input params.
        layer.input_scale = None
        layer.input_zero_point = None
        layer.azp_adj = None

        assert self.kernel is not None
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.kernel is not None
        return self.kernel.apply_weights(layer, x, bias)


class MtpHeadQuantConfig(QuantizationConfig):
    """Quantization config attached only to the MTP head's linear layers.

    Instantiated directly (never registered in the global quant registry) and
    passed to the MTP block's linear construction, so ``get_quant_method`` is
    only ever queried for MTP submodules.
    """

    def __init__(self, scheme: str) -> None:
        super().__init__()
        if scheme not in MTP_HEAD_QUANT_SCHEMES:
            raise ValueError(
                f"Unsupported MTP head quant scheme {scheme!r}; "
                f"expected one of {MTP_HEAD_QUANT_SCHEMES}."
            )
        self.scheme = scheme

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        # Not a registry-backed method; name is informational only.
        return "mtp_head"  # type: ignore[return-value]

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Cutlass w8a8 int8 is Turing+ (75); fp8 rejects non-SM89 at load.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MtpHeadQuantConfig":
        raise NotImplementedError(
            "MtpHeadQuantConfig is constructed from VLLM_MTP_HEAD_QUANT, not "
            "from a checkpoint config."
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if self.scheme == "fp8":
                return MtpHeadFp8LinearMethod()
            return MtpHeadInt8LinearMethod()
        return None


def maybe_get_mtp_head_quant_config(
    target_quant_config: QuantizationConfig | None,
) -> MtpHeadQuantConfig | None:
    """Return an :class:`MtpHeadQuantConfig` when the head should be quantized.

    Only fires when ``VLLM_MTP_HEAD_QUANT`` is ``fp8``/``int8`` *and* the MTP
    head would otherwise load unquantized (bf16/fp16). ``mtp_is_unquantized``
    is decided by the caller from the target checkpoint's quant config.
    """
    import vllm.envs as envs

    scheme = envs.VLLM_MTP_HEAD_QUANT
    if scheme not in MTP_HEAD_QUANT_SCHEMES:
        return None
    cfg = MtpHeadQuantConfig(scheme)
    logger.info_once(
        "VLLM_MTP_HEAD_QUANT=%s: quantizing MTP drafter head linears at load "
        "(target model quant: %s).",
        scheme,
        None if target_quant_config is None else target_quant_config.get_name(),
        scope="global",
    )
    return cfg
