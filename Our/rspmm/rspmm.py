from __future__ import annotations

import os
import re
import shutil
import subprocess

import torch
import torch.backends.openmp
from torch import autograd
from torch.utils import cpp_extension


_RSPMM_EXTENSION = None
_LOAD_ERROR = None


def _requested_rspmm_backend() -> str:
    backend = os.environ.get("C3R_GRAPH_RSPMM_BACKEND", "native").strip().lower()
    if backend not in {"native", "extension"}:
        return "native"
    return backend


def _prepare_rspmm_inputs(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source, target = edge_index
    # The C++/CUDA kernel computes output[row] += message(input[col]).
    # We want source -> target propagation, so we feed (target, source).
    prepared_edge_index = torch.stack([target, source], dim=0)
    key = prepared_edge_index[0] * (prepared_edge_index[1].max() + 1) + prepared_edge_index[1]
    order = key.argsort()
    return (
        prepared_edge_index[:, order],
        edge_type[order],
        edge_weight[order],
    )


def _reference_rspmm(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
    relation: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    row, col = edge_index
    messages = relation[edge_type] * input[col]
    if edge_weight is not None:
        messages = messages * edge_weight.unsqueeze(-1)

    output = torch.zeros_like(input)
    output.index_add_(0, row, messages)
    return output


def _native_rspmm_backward(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
    relation: torch.Tensor,
    input: torch.Tensor,
    output_grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row, col = edge_index
    grad_out_per_edge = output_grad[row]
    relation_per_edge = relation[edge_type]
    input_per_edge = input[col]
    weight = edge_weight.unsqueeze(-1)

    raw_weight_grad = grad_out_per_edge * relation_per_edge * input_per_edge
    weight_grad = raw_weight_grad.sum(dim=-1)

    relation_grad = torch.zeros_like(relation)
    relation_grad.index_add_(0, edge_type, weight * grad_out_per_edge * input_per_edge)

    input_grad = torch.zeros_like(input)
    input_grad.index_add_(0, col, weight * grad_out_per_edge * relation_per_edge)
    return weight_grad, relation_grad, input_grad


def _activate_msvc_environment() -> None:
    if os.name != "nt" or shutil.which("cl") is not None:
        return

    vc_env = None
    try:
        from setuptools._distutils._msvccompiler import _get_vc_env  # type: ignore[attr-defined]

        vc_env = _get_vc_env("x64")
    except Exception:
        try:
            from distutils._msvccompiler import _get_vc_env  # type: ignore[attr-defined]

            vc_env = _get_vc_env("x64")
        except Exception:
            vc_env = None

    if not vc_env:
        return

    for key, value in vc_env.items():
        if not value:
            continue
        os.environ[key.upper()] = value


def _build_compiler_flags() -> tuple[list[str], list[str] | None]:
    if os.name == "nt":
        extra_cflags = ["/O2", "/Zc:preprocessor"]
        if torch.backends.openmp.is_available():
            extra_cflags += ["/openmp", "/DAT_PARALLEL_OPENMP"]
        else:
            extra_cflags.append("/DAT_PARALLEL_NATIVE")
    else:
        extra_cflags = ["-Ofast"]
        if torch.backends.openmp.is_available():
            extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]
        else:
            extra_cflags.append("-DAT_PARALLEL_NATIVE")

    extra_cuda_cflags = None
    if torch.cuda.is_available():
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        extra_cuda_cflags = ["-O3"]
        if os.name == "nt":
            extra_cuda_cflags += [
                "-allow-unsupported-compiler",
                "-Xcompiler",
                "/Zc:preprocessor",
                "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING",
            ]
        extra_cflags.append("-DCUDA_OP" if os.name != "nt" else "/DCUDA_OP")

    return extra_cflags, extra_cuda_cflags


def _parse_major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    match = re.search(r"(\d+)\.(\d+)", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _detect_nvcc_cuda_version() -> str | None:
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        try:
            completed = subprocess.run(
                [nvcc, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            text = f"{completed.stdout}\n{completed.stderr}"
            match = re.search(r"release\s+(\d+\.\d+)", text)
            if match:
                return match.group(1)
        except Exception:
            pass

    cuda_home = cpp_extension.CUDA_HOME
    if cuda_home:
        match = re.search(r"CUDA[\\/]+v?(\d+\.\d+)", cuda_home, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _ensure_cuda_toolkit_matches_torch() -> None:
    if not torch.cuda.is_available():
        return

    torch_cuda = torch.version.cuda
    nvcc_cuda = _detect_nvcc_cuda_version()
    torch_version = _parse_major_minor(torch_cuda)
    nvcc_version = _parse_major_minor(nvcc_cuda)
    if torch_version is None or nvcc_version is None:
        return
    if torch_version != nvcc_version:
        raise RuntimeError(
            "Refuse to build rspmm CUDA extension because CUDA Toolkit does not match PyTorch: "
            f"torch.version.cuda={torch_cuda}, nvcc_cuda={nvcc_cuda}. "
            "This mismatch can compile a broken extension or fail at kernel launch. "
            "Install a CUDA Toolkit matching the PyTorch CUDA runtime, or keep rspmm_backend=native."
        )


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_diff = torch.max(torch.abs(actual - expected)).item()
        raise RuntimeError(f"rspmm 扩展自检失败：{name} 不一致，max_diff={max_diff:.3e}。")


def _validate_extension_on_device(module, device: torch.device) -> None:
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]], dtype=torch.long, device=device)
    edge_type = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    edge_weight = torch.tensor([1.0, 0.5, 1.5, 1.0], dtype=torch.float32, device=device, requires_grad=True)
    relation = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device, requires_grad=True)
    input = torch.tensor([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]], dtype=torch.float32, device=device, requires_grad=True)

    prepared_edge_index, prepared_edge_type, prepared_edge_weight = _prepare_rspmm_inputs(
        edge_index,
        edge_type,
        edge_weight,
    )
    expected = _reference_rspmm(
        prepared_edge_index,
        prepared_edge_type,
        prepared_edge_weight,
        relation,
        input,
    )
    if device.type == "cuda":
        actual = module.rspmm_add_mul_forward_cuda(
            prepared_edge_index,
            prepared_edge_type,
            prepared_edge_weight,
            relation,
            input,
        )
    else:
        actual = module.rspmm_add_mul_forward_cpu(
            prepared_edge_index,
            prepared_edge_type,
            prepared_edge_weight,
            relation,
            input,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _assert_close(f"{device.type} forward", actual, expected)

    output_grad = torch.tensor(
        [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]],
        dtype=torch.float32,
        device=device,
    )
    expected_weight_grad, expected_relation_grad, expected_input_grad = _native_rspmm_backward(
        prepared_edge_index,
        prepared_edge_type,
        prepared_edge_weight,
        relation,
        input,
        output_grad,
    )
    if device.type == "cuda":
        actual_weight_grad, actual_relation_grad, actual_input_grad = module.rspmm_add_mul_backward_cuda(
            prepared_edge_index,
            prepared_edge_type,
            prepared_edge_weight,
            relation,
            input,
            actual,
            output_grad,
        )
        torch.cuda.synchronize(device)
    else:
        actual_weight_grad, actual_relation_grad, actual_input_grad = module.rspmm_add_mul_backward_cpu(
            prepared_edge_index,
            prepared_edge_type,
            prepared_edge_weight,
            relation,
            input,
            actual,
            output_grad,
        )
    _assert_close(f"{device.type} weight_grad", actual_weight_grad, expected_weight_grad)
    _assert_close(f"{device.type} relation_grad", actual_relation_grad, expected_relation_grad)
    _assert_close(f"{device.type} input_grad", actual_input_grad, expected_input_grad)


def _validate_extension(module) -> None:
    _validate_extension_on_device(module, torch.device("cpu"))
    if torch.cuda.is_available():
        _validate_extension_on_device(module, torch.device("cuda"))


def _load_extension():
    global _RSPMM_EXTENSION, _LOAD_ERROR
    if _RSPMM_EXTENSION is not None or _LOAD_ERROR is not None:
        return
    if _requested_rspmm_backend() != "extension":
        _LOAD_ERROR = RuntimeError("配置指定 rspmm 使用 native 后端，跳过扩展编译。")
        return

    _activate_msvc_environment()
    if os.name == "nt" and shutil.which("cl") is None:
        _LOAD_ERROR = RuntimeError("未检测到 Visual C++ 编译器，使用 PyTorch 回退实现。")
        return

    path = os.path.join(os.path.dirname(__file__), "source")
    sources = [os.path.join(path, "rspmm.cpp"), os.path.join(path, "rspmm.cu")]
    extra_cflags, extra_cuda_cflags = _build_compiler_flags()

    if torch.cuda.is_available() and cpp_extension.CUDA_HOME is None and shutil.which("nvcc") is None:
        _LOAD_ERROR = RuntimeError(
            "检测到 CUDA 版 PyTorch，但未找到 nvcc / CUDA Toolkit。"
            " conda 的 cudatoolkit 仅提供运行时，无法编译 rspmm.cu。"
        )
        return
    try:
        _ensure_cuda_toolkit_matches_torch()
    except Exception as exc:
        _LOAD_ERROR = exc
        return

    if not torch.cuda.is_available():
        sources = [source for source in sources if not cpp_extension._is_cuda_file(source)]

    try:
        extension = cpp_extension.load(
            name="our_duetgraph_rspmm",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )
        _validate_extension(extension)
        _RSPMM_EXTENSION = extension
    except Exception as exc:  # pragma: no cover
        _LOAD_ERROR = exc


def get_rspmm_backend_info() -> tuple[str, str | None]:
    _load_extension()
    if _RSPMM_EXTENSION is not None:
        return "extension", None
    if _LOAD_ERROR is not None:
        return "native", str(_LOAD_ERROR)
    return "native", "扩展未加载，使用原生 PyTorch 解析实现。"


class RSPMMAddMulFunction(autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        _load_extension()
        if _RSPMM_EXTENSION is None:
            ctx.use_fallback = True
            ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input)
            return _reference_rspmm(edge_index, edge_type, edge_weight, relation, input)

        if input.device.type == "cuda":
            forward = _RSPMM_EXTENSION.rspmm_add_mul_forward_cuda
        else:
            forward = _RSPMM_EXTENSION.rspmm_add_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.use_fallback = False
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if ctx.use_fallback:
            edge_index, edge_type, edge_weight, relation, input = ctx.saved_tensors
            weight_grad, relation_grad, input_grad = _native_rspmm_backward(
                edge_index,
                edge_type,
                edge_weight,
                relation,
                input,
                output_grad,
            )
            return None, None, weight_grad, relation_grad, input_grad

        if output_grad.device.type == "cuda":
            backward = _RSPMM_EXTENSION.rspmm_add_mul_backward_cuda
        else:
            backward = _RSPMM_EXTENSION.rspmm_add_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


def generalized_rspmm(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
    relation: torch.Tensor,
    input: torch.Tensor,
    sum: str = "add",
    mul: str = "mul",
) -> torch.Tensor:
    if sum != "add" or mul != "mul":
        raise ValueError("The refactored project currently supports only add/mul rspmm.")
    edge_index, edge_type, edge_weight = _prepare_rspmm_inputs(edge_index, edge_type, edge_weight)
    return RSPMMAddMulFunction.apply(
        edge_index,
        edge_type,
        edge_weight,
        relation,
        input,
    )
