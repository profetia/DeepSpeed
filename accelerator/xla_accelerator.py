# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import functools
import torch


from .abstract_accelerator import DeepSpeedAccelerator


try:
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    from torch_xla import amp as xla_amp

    _XLA_AVAILABLE = True
except ImportError as e:
    _XLA_AVAILABLE = False


class XLA_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = "xla"
        self._communication_backend_name = "xla"
        self._compile_backend = "inductor"

    def is_synchronized_device(self):
        return True

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return "xla"
        return f"xla:{device_index}"

    def device(self, device_index=None):
        return xla.device(device_index)

    def set_device(self, device_index):
        pass

    def current_device(self):
        rank = os.environ.get("LOCAL_RANK", 0)
        return int(rank)

    def current_device_name(self):
        return "xla"

    def device_count(self):
        # TODO: Use xla.device_count() will attach the runtime to process that does not actually use it.
        # return xla.device_count()

        device_count = os.environ.get("DEEPSPEED_XLA_DEVICE_COUNT", 1)
        return int(device_count)

    def synchronize(self, device_index=None):
        xla.sync()

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        device = self.device(device_index)
        xm.set_rng_state(new_state, device)

    def get_rng_state(self, device_index=None):
        device = self.device(device_index)
        return xm.get_rng_state(device)

    def manual_seed(self, seed):
        xla.manual_seed(seed)

    def manual_seed_all(self, seed):
        for i in range(self.device_count()):
            self.set_rng_state(seed, i)

    def initial_seed(self):
        pass  # TODO

    def default_generator(self, device_index):  # TODO
        device = self.device(device_index)

        class _RngWrapper:
            def __init__(self, device):
                self.device = device

            def set_state(self, new_state):
                xm.set_rng_state(new_state, self.device)

            def get_state(self):
                return xm.get_rng_state(self.device)

        return _RngWrapper(device)

    # Streams/Events
    @property
    def Stream(self):
        pass

    def stream(self, stream):
        from deepspeed.runtime.utils import noop_context

        return noop_context()

    def current_stream(self, device_index=None):
        pass

    def default_stream(self, device_index=None):
        pass

    @property
    def Event(self):
        pass

    # Memory management
    def empty_cache(self):
        pass

    def memory_allocated(self, device_index=None):
        device = self.device(device_index)
        memory_info = xm.get_memory_info(device)
        return memory_info["bytes_used"]

    def max_memory_allocated(self, device_index=None):
        device = self.device(device_index)
        memory_info = xm.get_memory_info(device)
        return memory_info["bytes_limit"]

    def reset_max_memory_allocated(self, device_index=None):
        pass

    def memory_cached(self, device_index=None):
        pass  # TODO

    def max_memory_cached(self, device_index=None):
        pass  # TODO

    def reset_max_memory_cached(self, device_index=None):
        pass  # TODO

    def memory_stats(self, device_index=None):
        device = self.device(device_index)
        memory_info = xm.get_memory_info(device)
        return {
            "allocated_bytes.all.current": memory_info["bytes_used"],
            "allocated_bytes.all.peak": memory_info["peak_bytes_used"],
        }

    def reset_peak_memory_stats(self, device_index=None):
        pass

    def memory_reserved(self, device_index=None):  # TODO
        return self.memory_allocated(device_index)

    def max_memory_reserved(self, device_index=None):  # TODO
        return self.max_memory_allocated(device_index)

    def total_memory(self, device_index=None):
        pass  # TODO

    def available_memory(self, device_index=None):
        pass  # TODO

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return True

    def supported_dtypes(self):
        return [torch.bfloat16, torch.half, torch.float]

    # Misc
    def amp(self):
        return xla_amp.autocast

    def is_available(self):
        return _XLA_AVAILABLE

    def range_push(self, msg):
        pass

    def range_pop(self):
        pass

    def lazy_call(self, callback):
        xm.add_step_closure(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False

    # Graph operations
    def create_graph(self):
        pass

    def capture_to_graph(self, graph, pool=None, stream=None):
        from deepspeed.runtime.utils import noop_context

        return noop_context()

    def replay_graph(self, graph):
        pass

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        return functools.partial(
            torch.tensor, dtype=torch.bfloat16, device=self.device()
        )

    @property
    def ByteTensor(self):
        return functools.partial(torch.tensor, dtype=torch.uint8, device=self.device())

    @property
    def DoubleTensor(self):
        return functools.partial(torch.tensor, dtype=torch.double, device=self.device())

    @property
    def FloatTensor(self):
        return functools.partial(torch.tensor, dtype=torch.float, device=self.device())

    @property
    def HalfTensor(self):
        return functools.partial(torch.tensor, dtype=torch.half, device=self.device())

    @property
    def IntTensor(self):
        return functools.partial(torch.tensor, dtype=torch.int, device=self.device())

    @property
    def LongTensor(self):
        return functools.partial(torch.tensor, dtype=torch.long, device=self.device())

    def pin_memory(self, tensor, align_bytes=1):
        return tensor

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        return device_str.startswith("xla")

    def op_builder_dir(self):
        try:  # TODO
            from op_builder import __deepspeed__  # noqa: F401

            return "op_builder.cpu"
        except ImportError:
            return "deepspeed.ops.op_builder.cpu"

    # create an instance of op builder, specified by class_name
    def create_op_builder(self, class_name):
        builder_class = self.get_op_builder(class_name)
        if builder_class is not None:
            return builder_class()

    # return an op builder class, specified by class_name
    def get_op_builder(self, class_name):
        try:  # TODO
            from op_builder import __deepspeed__  # noqa: F401
            from op_builder.cpu import (
                AsyncIOBuilder,
                CCLCommBuilder,
                ShareMemCommBuilder,
                FusedAdamBuilder,
                CPUAdamBuilder,
                NotImplementedBuilder,
            )
        except ImportError:
            from deepspeed.ops.op_builder.cpu import (
                AsyncIOBuilder,
                CCLCommBuilder,
                ShareMemCommBuilder,
                FusedAdamBuilder,
                CPUAdamBuilder,
                NotImplementedBuilder,
            )

        if class_name == "CCLCommBuilder":
            return CCLCommBuilder
        elif class_name == "ShareMemCommBuilder":
            return ShareMemCommBuilder
        elif class_name == "FusedAdamBuilder":
            return FusedAdamBuilder
        elif class_name == "CPUAdamBuilder":
            return CPUAdamBuilder
        elif class_name == "AsyncIOBuilder":
            return AsyncIOBuilder

        return NotImplementedBuilder

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension

        return BuildExtension

    def export_envs(self):
        return []

    def visible_devices_envs(self):
        return ["NEURON_RT_VISIBLE_CORES"]

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(str(i) for i in local_accelerator_ids)

    def get_compile_backend(self):
        return self._compile_backend

    def set_compile_backend(self, backend):
        supported_backends = torch._dynamo.list_backends(exclude_tags=())
        if backend in supported_backends:
            self._compile_backend = backend
            return

        raise ValueError(
            f"{backend} not supported by {self.device_name()}. Supported Backends are {supported_backends}"
        )
