"""IP-Adapter FaceID preprocessor for identity-preserving generation.

Supports base FaceID (InsightFace 512-dim identity vector) and
FaceID PlusV2 (identity + CLIP Vision on face crop). For PlusV2, CLIP
embeddings are injected into the UNet's projection layer per inference.
"""
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..base import BasePreprocessor


class IPAdapterFaceIDProcessor(BasePreprocessor):
    """IP-Adapter FaceID preprocessor.

    Extracts face identity embeddings via InsightFace. For PlusV2,
    additionally extracts CLIP Vision embeddings from the face crop.
    Embeddings are NOT spatial — they condition the UNet's cross-attention.
    """

    EMBEDDING_DIM = 512  # InsightFace buffalo_l
    CLIP_INPUT_SIZE = 224
    FACE_CROP_PADDING = 0.2

    def __init__(self, device: torch.device, torch_dtype: torch.dtype, max_buffer_size: int = 1024):
        super().__init__(device, torch_dtype, max_buffer_size)
        self._face_app = None
        self._cached_id_embedding: Optional[torch.Tensor] = None
        self._cached_clip_embedding: Optional[torch.Tensor] = None
        self._zero_id_embedding: Optional[torch.Tensor] = None
        self._zero_clip_embedding: Optional[torch.Tensor] = None
        self._frame_counter: int = 0
        self._skip_frames: int = 10

        self._plus_v2: bool = False
        self._pipe_ref = None
        self._clip_image_size: int = 224

        # EMA-smooth identity embedding across frames.
        self._ema_id_embedding: Optional[torch.Tensor] = None
        self._ema_alpha: float = 0.3

        # OpenAI CLIP normalization constants, cached on device.
        self._clip_mean: Optional[torch.Tensor] = None
        self._clip_std: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return "faceid"

    def attach_pipe(self, pipe, plus_v2: bool = False):
        """Attach diffusers pipeline (required for PlusV2 CLIP flow)."""
        self._pipe_ref = pipe
        self._plus_v2 = plus_v2
        if plus_v2 and pipe is not None and hasattr(pipe, 'image_encoder') and pipe.image_encoder is not None:
            try:
                self._clip_image_size = pipe.image_encoder.config.image_size
            except Exception:
                self._clip_image_size = 224

    def load_model(self, config) -> None:
        """Initialize InsightFace face analysis model."""
        if self._face_app is not None:
            return

        try:
            from insightface.app import FaceAnalysis

            logging.info("[FaceID] Loading InsightFace (buffalo_l) for face embedding extraction...")

            self._face_app = FaceAnalysis(
                name="buffalo_l",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))

            if hasattr(config, 'skip_frames'):
                self._skip_frames = config.skip_frames
                self._plus_v2 = getattr(config, 'plus_v2', False)
            else:
                self._skip_frames = config.get('faceid_skip_frames', 10)
                self._plus_v2 = config.get('faceid_plus_v2', False)

            # Pre-allocate zero embeddings (avoid shape-change recompilation).
            self._zero_id_embedding = torch.zeros(
                1, 1, self.EMBEDDING_DIM,
                dtype=self.torch_dtype, device=self.device
            )
            # CLIP ViT-H/14: 1280-dim, 1 CLS + 16*16 patches = 257 tokens.
            self._zero_clip_embedding = torch.zeros(
                1, 1, 257, 1280,
                dtype=self.torch_dtype, device=self.device
            )

            self._clip_mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073],
                device=self.device, dtype=torch.float32,
            ).view(1, 3, 1, 1)
            self._clip_std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711],
                device=self.device, dtype=torch.float32,
            ).view(1, 3, 1, 1)

            self._loaded = True
            logging.info(f"[FaceID] InsightFace loaded (plus_v2={self._plus_v2})")
        except Exception as e:
            logging.error(f"[FaceID] Failed to load InsightFace: {e}")
            self._face_app = None
            self._loaded = False

    def unload_model(self) -> None:
        if self._face_app is not None:
            del self._face_app
            self._face_app = None
        self._cached_id_embedding = None
        self._cached_clip_embedding = None
        self._ema_id_embedding = None
        self._zero_id_embedding = None
        self._zero_clip_embedding = None
        self._clip_mean = None
        self._clip_std = None
        self._pipe_ref = None
        self._frame_counter = 0
        self._loaded = False
        logging.info("[FaceID] Unloaded")

    def process(self, image_tensor: torch.Tensor, config=None) -> Optional[list]:
        """Extract face embeddings (identity + optional CLIP).

        Returns a one-element list ``[id_embedding]`` suitable for
        ``ip_adapter_image_embeds``. For PlusV2, CLIP embeddings are
        injected separately into the UNet projection layer.
        """
        if self._face_app is None:
            return [self._zero_id_embedding] if self._zero_id_embedding is not None else None

        if config is not None:
            if hasattr(config, 'skip_frames'):
                self._skip_frames = config.skip_frames
            elif isinstance(config, dict):
                self._skip_frames = config.get('faceid_skip_frames', self._skip_frames)

        if not getattr(self, '_input_range_logged', False):
            mn, mx = image_tensor.min().item(), image_tensor.max().item()
            logging.info(f"[FaceID DIAG] Input tensor shape={tuple(image_tensor.shape)}, "
                         f"dtype={image_tensor.dtype}, range=[{mn:.3f}, {mx:.3f}]")
            self._input_range_logged = True

        self._frame_counter += 1
        use_cache = (
            self._cached_id_embedding is not None
            and self._skip_frames > 1
            and self._frame_counter % self._skip_frames != 0
        )

        if use_cache:
            if self._plus_v2 and self._cached_clip_embedding is not None:
                self._inject_clip_embedding(self._cached_clip_embedding)
            return [self._cached_id_embedding]

        id_embed, clip_embed, face_bbox = self._extract_embeddings(image_tensor)

        if id_embed is None:
            if self._cached_id_embedding is not None:
                if self._plus_v2 and self._cached_clip_embedding is not None:
                    self._inject_clip_embedding(self._cached_clip_embedding)
                return [self._cached_id_embedding]
            return [self._zero_id_embedding]

        if self._ema_id_embedding is None:
            self._ema_id_embedding = id_embed
        else:
            self._ema_id_embedding = (
                self._ema_alpha * id_embed + (1.0 - self._ema_alpha) * self._ema_id_embedding
            )

        smoothed_id = self._ema_id_embedding
        self._cached_id_embedding = smoothed_id

        if self._plus_v2 and clip_embed is not None:
            self._cached_clip_embedding = clip_embed
            self._inject_clip_embedding(clip_embed)

        return [smoothed_id]

    def _extract_embeddings(self, image_tensor: torch.Tensor):
        """Returns (id_embedding, clip_embedding, face_bbox) — any may be None."""
        try:
            # CHW [0,1] RGB → HWC uint8 BGR for InsightFace.
            image_np = (image_tensor.permute(1, 2, 0).contiguous().cpu().numpy() * 255).astype(np.uint8)
            image_bgr = image_np[:, :, ::-1].copy()

            faces = self._face_app.get(image_bgr)

            self._detect_count = getattr(self, '_detect_count', 0) + 1
            self._detect_hits = getattr(self, '_detect_hits', 0) + (1 if len(faces) > 0 else 0)
            if self._detect_count % 30 == 0:
                hit_rate = self._detect_hits / self._detect_count * 100
                logging.info(f"[FaceID DIAG] Face detection: {self._detect_hits}/{self._detect_count} "
                             f"({hit_rate:.0f}%) | last detect: {len(faces)} face(s)")

            if len(faces) == 0:
                return None, None, None

            if len(faces) > 1:
                faces = sorted(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True
                )

            face = faces[0]

            id_embedding = torch.from_numpy(face.normed_embedding).unsqueeze(0).unsqueeze(0)
            id_embedding = id_embedding.to(dtype=self.torch_dtype, device=self.device)

            clip_embedding = None
            if self._plus_v2 and self._pipe_ref is not None:
                clip_embedding = self._extract_clip_embedding(image_tensor, face.bbox)

            return id_embedding, clip_embedding, face.bbox

        except Exception as e:
            logging.warning(f"[FaceID] Face extraction failed: {e}")
            return None, None, None

    def _extract_clip_embedding(self, image_tensor: torch.Tensor, bbox):
        """Extract CLIP Vision embedding from the face crop. Returns (1,1,257,1280) or None."""
        try:
            _, h, w = image_tensor.shape
            x1, y1, x2, y2 = bbox.astype(int) if hasattr(bbox, 'astype') else [int(c) for c in bbox]

            bw, bh = x2 - x1, y2 - y1
            px, py = int(bw * self.FACE_CROP_PADDING), int(bh * self.FACE_CROP_PADDING)
            x1, y1 = max(0, x1 - px), max(0, y1 - py)
            x2, y2 = min(w, x2 + px), min(h, y2 + py)

            if x2 <= x1 or y2 <= y1:
                return None

            face_crop = image_tensor[:, y1:y2, x1:x2].unsqueeze(0)

            face_crop = F.interpolate(
                face_crop,
                size=(self._clip_image_size, self._clip_image_size),
                mode='bilinear',
                align_corners=False
            )

            clip_mean = self._clip_mean.to(dtype=face_crop.dtype)
            clip_std = self._clip_std.to(dtype=face_crop.dtype)
            face_crop = (face_crop - clip_mean) / clip_std
            face_crop = face_crop.to(dtype=self.torch_dtype)

            with torch.inference_mode():
                clip_output = self._pipe_ref.image_encoder(face_crop, output_hidden_states=True)
                # Penultimate hidden state — standard for IP-Adapter.
                clip_embedding = clip_output.hidden_states[-2]

            # Diffusers expects 4D: (batch, num_ip_adapters, seq_len, hidden_dim).
            if clip_embedding.dim() == 3:
                clip_embedding = clip_embedding.unsqueeze(1)

            return clip_embedding

        except Exception as e:
            logging.warning(f"[FaceID] CLIP embedding extraction failed: {e}")
            return None

    def _inject_clip_embedding(self, clip_embedding: torch.Tensor) -> None:
        """Inject CLIP embedding into UNet IP-Adapter projection layer (PlusV2)."""
        if self._pipe_ref is None:
            return
        try:
            proj_layer = self._pipe_ref.unet.encoder_hid_proj.image_projection_layers[0]
            proj_layer.clip_embeds = clip_embedding
        except Exception as e:
            if not getattr(self, '_clip_inject_error_logged', False):
                logging.warning(f"[FaceID] Failed to inject CLIP embedding: {e}")
                self._clip_inject_error_logged = True
