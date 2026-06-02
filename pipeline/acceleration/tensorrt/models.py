#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/models.py

#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants


# SDXL ControlNet residual structure: 9 down blocks + 1 mid block
# (vs SD 1.5/2.1's 12 down). Channels and spatial divisions are for a
# 512px image (latent 64x64); divisions scale with resolution.
SDXL_CN_DOWN_CHANNELS = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]
SDXL_CN_DOWN_SPATIAL_DIVS = [1, 1, 1, 2, 2, 2, 4, 4, 4]
SDXL_CN_NUM_DOWN = 9
SDXL_CN_MID_CHANNELS = 1280
SDXL_CN_MID_SPATIAL_DIV = 4


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)

        # ByteSize() itself throws on >2 GB protos — the exact case we want
        # to detect — so wrap the gate in try/except.
        import logging
        try:
            proto_bytes = onnx_graph.ByteSize()
            too_large = proto_bytes > 2147483648
            size_gb_str = f"{proto_bytes / 1e9:.2f} GB"
        except Exception:
            too_large = True
            size_gb_str = ">2 GB (proto unserializable — assumed >2 GB)"

        if too_large:
            # Skip standard ONNX shape inference — TRT will infer during build.
            logging.warning(f"Model size ({size_gb_str}) exceeds 2GB - skipping ONNX shape inference")
            logging.info("TensorRT will infer shapes during engine building")
            if return_onnx:
                return onnx_graph
            return
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


class BaseModel:
    def __init__(
        self,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch_size=16,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
    ):
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device
        self.verbose = verbose

        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.min_image_shape = 256
        self.max_image_shape = 1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


class CLIP(BaseModel):
    def __init__(self, device, max_batch_size, embedding_dim, min_batch_size=1):
        super(CLIP, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
        )
        self.name = "CLIP"

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings", "pooler_output"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.info(self.name + ": original")
        opt.select_outputs([0])
        opt.cleanup()
        opt.info(self.name + ": remove output[1]")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        opt.select_outputs([0], names=["text_embeddings"])
        opt.info(self.name + ": remove output[0]")
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return opt_onnx_graph


class UNet(BaseModel):
    """SD 1.5/2.1 UNet spec: sample/ts/EHS + 12 CN down + 1 mid residual."""
    def __init__(
        self,
        fp16=False,
        device="cuda",
        max_batch_size=16,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
    ):
        super(UNet, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "UNet"

    def get_input_names(self):
        return [
            "sample",
            "timestep",
            "encoder_hidden_states",
            "down_block_0",
            "down_block_1",
            "down_block_2",
            "down_block_3",
            "down_block_4",
            "down_block_5",
            "down_block_6",
            "down_block_7",
            "down_block_8",
            "down_block_9",
            "down_block_10",
            "down_block_11",
            "mid_block",
        ]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }
        for i in range(12):
            axes[f"down_block_{i}"] = {0: "2B"}
        axes["mid_block"] = {0: "2B"}
        return axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        profile = {
            "sample": [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, latent_height, latent_width),
                (max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }
        # CN residual ports share the "2B" batch axis with ``sample`` — they
        # MUST appear in the profile with the SAME batch (min/opt/max) or
        # TRT fails with "Error Code 4: 2B ... contradictory constraints"
        # at batch != 1 (multi-step denoising).
        down_block_channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
        down_block_spatial_divs = [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8]
        for i in range(12):
            ch = down_block_channels[i]
            h = latent_height // down_block_spatial_divs[i]
            w = latent_width // down_block_spatial_divs[i]
            profile[f"down_block_{i}"] = [
                (min_batch, ch, h, w), (batch_size, ch, h, w), (max_batch, ch, h, w),
            ]
        mh, mw = latent_height // 8, latent_width // 8
        profile["mid_block"] = [
            (min_batch, 1280, mh, mw), (batch_size, 1280, mh, mw), (max_batch, 1280, mh, mw),
        ]
        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32

        down_block_channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
        down_block_spatial_divs = [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8]

        sample_inputs = [
            torch.randn(
                2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        ]

        for i in range(12):
            h = latent_height // down_block_spatial_divs[i]
            w = latent_width // down_block_spatial_divs[i]
            sample_inputs.append(
                torch.zeros(2 * batch_size, down_block_channels[i], h, w, dtype=dtype, device=self.device)
            )

        sample_inputs.append(
            torch.zeros(2 * batch_size, 1280, latent_height // 8, latent_width // 8, dtype=dtype, device=self.device)
        )

        return tuple(sample_inputs)


class UNetV2V(UNet):
    """SD 1.5 UNet spec + StreamV2V kvo cache as engine I/O (N kvo_in / N kvo_out ports).

    Cache tensor shape per port: ``(3, cache_maxframes, 2B, seq, dim)``. Leading 3
    stacks [key, value, output]. ``seq`` and ``dim`` are layer-specific and baked
    at build time (passed via ``kvo_cache_shapes`` from ``get_kvo_cache_info``).
    """

    def __init__(self, kvo_cache_shapes, max_cache_frames=1, **kwargs):
        super().__init__(**kwargs)
        self.kvo_cache_shapes = list(kvo_cache_shapes)
        self.max_cache_frames = max_cache_frames
        self.n_kvo = len(self.kvo_cache_shapes)
        self.name = "UNetV2V"

    def get_input_names(self):
        return super().get_input_names() + [f"kvo_in_{i}" for i in range(self.n_kvo)]

    def get_output_names(self):
        return super().get_output_names() + [f"kvo_out_{i}" for i in range(self.n_kvo)]

    def get_dynamic_axes(self):
        axes = super().get_dynamic_axes()
        # (3, cache_maxframes, 2B, seq, dim) — axes 0/3/4 fixed.
        for i in range(self.n_kvo):
            axes[f"kvo_in_{i}"] = {1: "C", 2: "2B"}
            axes[f"kvo_out_{i}"] = {1: "C", 2: "2B"}
        return axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        profile = super().get_input_profile(batch_size, image_height, image_width, static_batch, static_shape)
        min_batch, max_batch = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )[:2]
        for i, (seq, dim) in enumerate(self.kvo_cache_shapes):
            profile[f"kvo_in_{i}"] = [
                (3, 1, min_batch, seq, dim),
                (3, self.max_cache_frames, batch_size, seq, dim),
                (3, self.max_cache_frames, max_batch, seq, dim),
            ]
        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        d = super().get_shape_dict(batch_size, image_height, image_width)
        for i, (seq, dim) in enumerate(self.kvo_cache_shapes):
            d[f"kvo_in_{i}"] = (3, self.max_cache_frames, 2 * batch_size, seq, dim)
            d[f"kvo_out_{i}"] = (3, self.max_cache_frames, 2 * batch_size, seq, dim)
        return d

    def get_sample_input(self, batch_size, image_height, image_width):
        base = list(super().get_sample_input(batch_size, image_height, image_width))
        dtype = torch.float16 if self.fp16 else torch.float32
        for seq, dim in self.kvo_cache_shapes:
            base.append(
                torch.zeros(
                    3, self.max_cache_frames, 2 * batch_size, seq, dim,
                    dtype=dtype, device=self.device,
                )
            )
        return tuple(base)


class UNetSimple(BaseModel):
    """UNet spec without ControlNet: sample/timestep/encoder_hidden_states only."""
    def __init__(
        self,
        fp16=False,
        device="cuda",
        max_batch_size=16,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
    ):
        super(UNetSimple, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "UNetSimple"

    def get_input_names(self):
        return [
            "sample",
            "timestep",
            "encoder_hidden_states",
        ]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "sample": [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, latent_height, latent_width),
                (max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(
                2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        )


class UNetXLSimple(UNetSimple):
    """SDXL UNet spec without ControlNet (adds text_embeds + time_ids)."""
    def __init__(
        self,
        fp16=False,
        device="cuda",
        max_batch_size=16,
        min_batch_size=1,
        embedding_dim=2048,  # SDXL concatenated text-encoder hidden size
        text_maxlen=77,
        unet_dim=4,
    ):
        super(UNetXLSimple, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
            unet_dim=unet_dim,
        )
        self.name = "UNetXLSimple"

    def get_input_names(self):
        return [
            "sample",
            "timestep",
            "encoder_hidden_states",
            "text_embeds",
            "time_ids",
        ]

    def get_dynamic_axes(self):
        axes = super().get_dynamic_axes()
        axes["text_embeds"] = {0: "2B"}
        axes["time_ids"] = {0: "2B"}
        return axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        profile = super().get_input_profile(batch_size, image_height, image_width, static_batch, static_shape)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        profile["text_embeds"] = [
            (min_batch, 1280),
            (batch_size, 1280),
            (max_batch, 1280),
        ]
        profile["time_ids"] = [
            (min_batch, 6),
            (batch_size, 6),
            (max_batch, 6),
        ]
        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        shape_dict = super().get_shape_dict(batch_size, image_height, image_width)
        shape_dict["text_embeds"] = (2 * batch_size, 1280)
        shape_dict["time_ids"] = (2 * batch_size, 6)
        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width):
        base_inputs = list(super().get_sample_input(batch_size, image_height, image_width))
        dtype = torch.float16 if self.fp16 else torch.float32

        base_inputs.append(
            torch.randn(2 * batch_size, 1280, dtype=dtype, device=self.device)
        )
        base_inputs.append(
            torch.randn(2 * batch_size, 6, dtype=dtype, device=self.device)
        )

        return tuple(base_inputs)


class UNetXL(BaseModel):
    """SDXL UNet spec with 9 CN down residuals + 1 mid + SDXL conditioning.

    Input order: sample, timestep, encoder_hidden_states,
                 down_block_0..8, mid_block, text_embeds, time_ids.
    """
    def __init__(
        self,
        fp16=False,
        device="cuda",
        max_batch_size=16,
        min_batch_size=1,
        embedding_dim=2048,
        text_maxlen=77,
        unet_dim=4,
    ):
        super(UNetXL, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "UNetXL"

    def get_input_names(self):
        return (
            ["sample", "timestep", "encoder_hidden_states"]
            + [f"down_block_{i}" for i in range(SDXL_CN_NUM_DOWN)]
            + ["mid_block", "text_embeds", "time_ids"]
        )

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
            "text_embeds": {0: "2B"},
            "time_ids": {0: "2B"},
        }
        for i in range(SDXL_CN_NUM_DOWN):
            axes[f"down_block_{i}"] = {0: "2B"}
        axes["mid_block"] = {0: "2B"}
        return axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch, max_batch, _, _, _, _,
            min_latent_height, max_latent_height,
            min_latent_width, max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        profile = {
            "sample": [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, latent_height, latent_width),
                (max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim),
            ],
            "text_embeds": [(min_batch, 1280), (batch_size, 1280), (max_batch, 1280)],
            "time_ids": [(min_batch, 6), (batch_size, 6), (max_batch, 6)],
        }
        # CN residual ports share "2B" with ``sample`` — see UNet for why
        # they must be in the profile with the same batch (min/opt/max).
        for i in range(SDXL_CN_NUM_DOWN):
            ch = SDXL_CN_DOWN_CHANNELS[i]
            h = latent_height // SDXL_CN_DOWN_SPATIAL_DIVS[i]
            w = latent_width // SDXL_CN_DOWN_SPATIAL_DIVS[i]
            profile[f"down_block_{i}"] = [
                (min_batch, ch, h, w), (batch_size, ch, h, w), (max_batch, ch, h, w),
            ]
        mh = latent_height // SDXL_CN_MID_SPATIAL_DIV
        mw = latent_width // SDXL_CN_MID_SPATIAL_DIV
        profile["mid_block"] = [
            (min_batch, SDXL_CN_MID_CHANNELS, mh, mw),
            (batch_size, SDXL_CN_MID_CHANNELS, mh, mw),
            (max_batch, SDXL_CN_MID_CHANNELS, mh, mw),
        ]
        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        shape_dict = {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
            "text_embeds": (2 * batch_size, 1280),
            "time_ids": (2 * batch_size, 6),
        }
        for i in range(SDXL_CN_NUM_DOWN):
            h = latent_height // SDXL_CN_DOWN_SPATIAL_DIVS[i]
            w = latent_width // SDXL_CN_DOWN_SPATIAL_DIVS[i]
            shape_dict[f"down_block_{i}"] = (2 * batch_size, SDXL_CN_DOWN_CHANNELS[i], h, w)
        shape_dict["mid_block"] = (
            2 * batch_size, SDXL_CN_MID_CHANNELS,
            latent_height // SDXL_CN_MID_SPATIAL_DIV,
            latent_width // SDXL_CN_MID_SPATIAL_DIV,
        )
        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32

        sample_inputs = [
            torch.randn(2 * batch_size, self.unet_dim, latent_height, latent_width,
                        dtype=torch.float32, device=self.device),
            torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim,
                        dtype=dtype, device=self.device),
        ]
        for i in range(SDXL_CN_NUM_DOWN):
            h = latent_height // SDXL_CN_DOWN_SPATIAL_DIVS[i]
            w = latent_width // SDXL_CN_DOWN_SPATIAL_DIVS[i]
            sample_inputs.append(
                torch.zeros(2 * batch_size, SDXL_CN_DOWN_CHANNELS[i], h, w, dtype=dtype, device=self.device)
            )
        sample_inputs.append(
            torch.zeros(2 * batch_size, SDXL_CN_MID_CHANNELS,
                        latent_height // SDXL_CN_MID_SPATIAL_DIV,
                        latent_width // SDXL_CN_MID_SPATIAL_DIV,
                        dtype=dtype, device=self.device)
        )
        sample_inputs.append(torch.randn(2 * batch_size, 1280, dtype=dtype, device=self.device))
        sample_inputs.append(torch.randn(2 * batch_size, 6, dtype=dtype, device=self.device))
        return tuple(sample_inputs)


class UNetXLV2V(UNetXL):
    """SDXL UNet spec + StreamV2V kvo cache as engine I/O. See UNetV2V for cache layout."""

    def __init__(self, kvo_cache_shapes, max_cache_frames=1, **kwargs):
        super().__init__(**kwargs)
        self.kvo_cache_shapes = list(kvo_cache_shapes)
        self.max_cache_frames = max_cache_frames
        self.n_kvo = len(self.kvo_cache_shapes)
        self.name = "UNetXLV2V"

    def get_input_names(self):
        return super().get_input_names() + [f"kvo_in_{i}" for i in range(self.n_kvo)]

    def get_output_names(self):
        return super().get_output_names() + [f"kvo_out_{i}" for i in range(self.n_kvo)]

    def get_dynamic_axes(self):
        axes = super().get_dynamic_axes()
        for i in range(self.n_kvo):
            axes[f"kvo_in_{i}"] = {1: "C", 2: "2B"}
            axes[f"kvo_out_{i}"] = {1: "C", 2: "2B"}
        return axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        profile = super().get_input_profile(batch_size, image_height, image_width, static_batch, static_shape)
        min_batch, max_batch = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )[:2]
        for i, (seq, dim) in enumerate(self.kvo_cache_shapes):
            profile[f"kvo_in_{i}"] = [
                (3, 1, min_batch, seq, dim),
                (3, self.max_cache_frames, batch_size, seq, dim),
                (3, self.max_cache_frames, max_batch, seq, dim),
            ]
        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        d = super().get_shape_dict(batch_size, image_height, image_width)
        for i, (seq, dim) in enumerate(self.kvo_cache_shapes):
            d[f"kvo_in_{i}"] = (3, self.max_cache_frames, 2 * batch_size, seq, dim)
            d[f"kvo_out_{i}"] = (3, self.max_cache_frames, 2 * batch_size, seq, dim)
        return d

    def get_sample_input(self, batch_size, image_height, image_width):
        base = list(super().get_sample_input(batch_size, image_height, image_width))
        dtype = torch.float16 if self.fp16 else torch.float32
        for seq, dim in self.kvo_cache_shapes:
            base.append(
                torch.zeros(
                    3, self.max_cache_frames, 2 * batch_size, seq, dim,
                    dtype=dtype, device=self.device,
                )
            )
        return tuple(base)


class ControlNet(BaseModel):
    """ControlNet spec: 4 inputs, 12 down + 1 mid named outputs."""
    def __init__(
        self,
        fp16=False,
        device="cuda",
        max_batch_size=16,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
    ):
        super(ControlNet, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "ControlNet"

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "controlnet_cond"]

    def get_output_names(self):
        return [f"down_block_{i}" for i in range(12)] + ["mid_block"]

    def get_dynamic_axes(self):
        # ``sample`` is latent-space (H=64 for 512px) while ``controlnet_cond``
        # is image-space (H=512). TRT treats axis-name reuse as a constraint
        # — using the same "H"/"W" for both inputs makes the build fail with
        # API Usage Error 4 (contradictory kMIN/kMAX). Use distinct names
        # ("IH"/"IW") for image-space dims.
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "controlnet_cond": {0: "2B", 2: "IH", 3: "IW"},
            **{f"down_block_{i}": {0: "2B"} for i in range(12)},
            "mid_block": {0: "2B"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "sample": [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, latent_height, latent_width),
                (max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim),
            ],
            "controlnet_cond": [
                (min_batch, 3, min_latent_height * 8, min_latent_width * 8),
                (batch_size, 3, latent_height * 8, latent_width * 8),
                (max_batch, 3, max_latent_height * 8, max_latent_width * 8),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "controlnet_cond": (2 * batch_size, 3, latent_height * 8, latent_width * 8),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(
                2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            torch.randn(2 * batch_size, 3, latent_height * 8, latent_width * 8, dtype=dtype, device=self.device),
        )


class DepthAnything(BaseModel):
    """Static-shape ONNX export spec for Depth-Anything V2.

    Single (1, 3, image_size, image_size) input, single (1, image_size, image_size)
    output. ViT patch size = 14 so ``image_size`` must be divisible by 14.
    """
    def __init__(
        self,
        fp16=False,
        device="cuda",
        image_size=378,
    ):
        super(DepthAnything, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=1,
            min_batch_size=1,
            embedding_dim=None,
        )
        if image_size % 14 != 0:
            raise ValueError(
                f"DepthAnything image_size must be divisible by 14 (ViT patch size); "
                f"got {image_size}."
            )
        self.image_size = image_size
        self.name = f"Depth-Anything V2 ({image_size}x{image_size})"
        # Relax BaseModel's diffusion-specific check_dims assertions.
        self.min_latent_shape = 1
        self.max_latent_shape = 4096

    def get_input_names(self):
        return ["pixel_values"]

    def get_output_names(self):
        return ["predicted_depth"]

    def get_dynamic_axes(self):
        return {}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        s = self.image_size
        return {
            "pixel_values": [
                (1, 3, s, s),
                (1, 3, s, s),
                (1, 3, s, s),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        s = self.image_size
        return {
            "pixel_values": (1, 3, s, s),
            "predicted_depth": (1, s, s),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        dtype = torch.float16 if self.fp16 else torch.float32
        return torch.randn(1, 3, self.image_size, self.image_size, dtype=dtype, device=self.device)


class VAE(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAE, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE decoder"

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {
            "latent": {0: "B", 2: "H", 3: "W"},
            "images": {0: "B", 2: "8H", 3: "8W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            4,
            latent_height,
            latent_width,
            dtype=torch.float32,
            device=self.device,
        )


class VAEEncoder(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAEEncoder, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE encoder"

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "images": {0: "B", 2: "8H", 3: "8W"},
            "latent": {0: "B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            3,
            image_height,
            image_width,
            dtype=torch.float32,
            device=self.device,
        )
