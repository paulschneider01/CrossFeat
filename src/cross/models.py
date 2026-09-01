"""VAE crossing model components used by the upload pipeline."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) from Domain-Adversarial Neural Networks.

    Forward pass: identity function
    Backward pass: negates gradients and scales by lambda

    This makes the encoder actively try to fool the discriminator,
    forcing it to learn domain-invariant features.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    """Wrapper module for gradient reversal layer."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


class ModalityDiscriminator(nn.Module):
    """
    Discriminator that tries to predict modality from z_geom.

    Used with gradient reversal to force z_geom to be modality-invariant.
    If the discriminator can't predict modality from z_geom, then z_geom
    doesn't contain modality-specific information.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_modalities: int = 2):
        super().__init__()
        self.grl = GradientReversal(lambda_=1.0)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_modalities),
        )

    def forward(self, z_geom, reverse_grad: bool = True):
        """
        Predict modality from z_geom.

        Args:
            z_geom: Geometry latent vectors, shape (B, d_geom)
            reverse_grad: If True, apply gradient reversal (training mode)

        Returns:
            logits: Modality prediction logits, shape (B, num_modalities)
        """
        if reverse_grad:
            z_geom = self.grl(z_geom)
        return self.classifier(z_geom)

    def set_lambda(self, lambda_: float):
        """Set gradient reversal strength."""
        self.grl.set_lambda(lambda_)



class DisentangledEncoder(nn.Module):
    """
    Encoder that disentangles geometry and appearance from descriptors.

    Maps d -> (z_geom, z_app) where:
    - z_geom captures local structure (edges, corners, orientations) - SHARED across modalities
    - z_app captures intensity/contrast statistics - DIFFERS across modalities

    Two modes:
    - separate=False (default): Shared backbone with separate heads (faster, less disentangled)
    - separate=True: Completely separate encoders for geometry and appearance (better disentanglement)

    Normalization:
    - normalize_geom=True: Forces z_geom to have fixed L2 norm (prevents collapse to zero)
    """

    def __init__(
        self,
        descriptor_dim: int,
        d_geom: int,
        d_app: int,
        hidden_dim: int = 256,
        separate: bool = False,
        normalize_geom: bool = False,
        geom_scale: float = 1.0,
    ):
        super().__init__()
        self.d_geom = d_geom
        self.d_app = d_app
        self.separate = separate
        self.normalize_geom = normalize_geom
        self.geom_scale = geom_scale

        if separate:
            # Completely separate encoders - no information sharing
            self.geom_encoder = nn.Sequential(
                nn.Linear(descriptor_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, d_geom),
            )
            self.app_encoder = nn.Sequential(
                nn.Linear(descriptor_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, d_app),
            )
        else:
            # Shared encoder backbone (original architecture)
            self.encoder = nn.Sequential(
                nn.Linear(descriptor_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            # Separate heads for geometry and appearance
            self.fc_geom = nn.Linear(hidden_dim, d_geom)
            self.fc_app = nn.Linear(hidden_dim, d_app)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode descriptor into geometry and appearance components.

        Args:
            d: Input descriptors, shape (B, D)

        Returns:
            z_geom: Geometry latent, shape (B, d_geom)
            z_app: Appearance latent, shape (B, d_app)
        """
        if self.separate:
            z_geom = self.geom_encoder(d)
            z_app = self.app_encoder(d)
        else:
            h = self.encoder(d)
            z_geom = self.fc_geom(h)
            z_app = self.fc_app(h)

        # Optionally normalize z_geom to prevent collapse to zero
        if self.normalize_geom:
            z_geom = F.normalize(z_geom, p=2, dim=1) * self.geom_scale

        return z_geom, z_app


class ModalitySpecificEncoders(nn.Module):
    """
    Per-modality encoders that each map descriptors to (z_geom, z_app).

    Key insight: T2 and US descriptors have fundamentally different statistical
    properties. A single encoder cannot separate "modality" from "geometry"
    because what looks like an edge in T2 vs US is encoded very differently
    in the SIFT descriptor space.

    Solution: Give each modality its own encoder. The geometry consistency loss
    (MSE between z_geom_t2 and z_geom_us for aligned pairs) forces the encoders
    to produce matching geometry representations despite different input statistics.

    This is similar to how multi-view systems (e.g., stereo) often use separate
    feature extractors per camera, with a loss that enforces consistency.
    """

    def __init__(
        self,
        descriptor_dim: int,
        d_geom: int,
        d_app: int,
        num_modalities: int,
        hidden_dim: int = 256,
        normalize_geom: bool = False,
        geom_scale: float = 1.0,
    ):
        super().__init__()
        self.d_geom = d_geom
        self.d_app = d_app
        self.num_modalities = num_modalities
        self.normalize_geom = normalize_geom
        self.geom_scale = geom_scale

        # Create a separate encoder for each modality
        self.encoders = nn.ModuleList([
            nn.ModuleDict({
                'geom': nn.Sequential(
                    nn.Linear(descriptor_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, d_geom),
                ),
                'app': nn.Sequential(
                    nn.Linear(descriptor_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, d_app),
                ),
            })
            for _ in range(num_modalities)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        d: torch.Tensor,
        modality: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode descriptor into geometry and appearance components.

        Args:
            d: Input descriptors, shape (B, D)
            modality: Modality indices, shape (B,). Required for per-modality encoding.

        Returns:
            z_geom: Geometry latent, shape (B, d_geom)
            z_app: Appearance latent, shape (B, d_app)
        """
        if modality is None:
            raise ValueError("ModalitySpecificEncoders requires modality indices")

        batch_size = d.shape[0]
        device = d.device

        # Initialize output tensors (match input dtype for mixed-precision safety)
        z_geom = torch.zeros(batch_size, self.d_geom, device=device, dtype=d.dtype)
        z_app = torch.zeros(batch_size, self.d_app, device=device, dtype=d.dtype)

        # Process each modality separately
        for mod_idx in range(self.num_modalities):
            mask = (modality == mod_idx)
            if mask.any():
                d_mod = d[mask]
                z_geom[mask] = self.encoders[mod_idx]['geom'](d_mod)
                z_app[mask] = self.encoders[mod_idx]['app'](d_mod)

        # Optionally normalize z_geom to prevent collapse
        if self.normalize_geom:
            z_geom = F.normalize(z_geom, p=2, dim=1) * self.geom_scale

        return z_geom, z_app

    def encode_modality(
        self,
        d: torch.Tensor,
        modality_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all descriptors using a specific modality's encoder.

        Useful for inference when modality is known.

        Args:
            d: Input descriptors, shape (B, D)
            modality_idx: Which modality encoder to use

        Returns:
            z_geom: Geometry latent, shape (B, d_geom)
            z_app: Appearance latent, shape (B, d_app)
        """
        z_geom = self.encoders[modality_idx]['geom'](d)
        z_app = self.encoders[modality_idx]['app'](d)

        if self.normalize_geom:
            z_geom = F.normalize(z_geom, p=2, dim=1) * self.geom_scale

        return z_geom, z_app


class DisentangledDecoder(nn.Module):
    """
    Decoder that reconstructs descriptors from geometry and appearance components.

    Maps (z_geom, z_app) -> d

    Optionally supports FiLM conditioning on target modality. This allows the
    decoder to adapt its reconstruction based on which modality we're decoding
    into, which is important because SIFT gradient histograms are inherently
    modality-dependent (different tissue contrast → different edges).
    """

    def __init__(
        self,
        descriptor_dim: int,
        d_geom: int,
        d_app: int,
        hidden_dim: int = 256,
        condition_decoder: bool = False,
        num_modalities: int = 2,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.condition_decoder = condition_decoder

        if condition_decoder:
            # Split decoder into two stages with FiLM conditioning in between
            self.fc1 = nn.Sequential(
                nn.Linear(d_geom + d_app, hidden_dim),
                nn.ReLU(),
            )
            self.fc2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, descriptor_dim),
            )
            # Modality embedding and FiLM generator
            self.mod_embed = nn.Embedding(num_modalities, embed_dim)
            self.film_net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * hidden_dim),  # gamma and beta
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(d_geom + d_app, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, descriptor_dim),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize FiLM to near-identity: gamma≈0 (so 1+gamma≈1), beta≈0
        if self.condition_decoder:
            film_layers = [m for m in self.film_net.modules() if isinstance(m, nn.Linear)]
            if film_layers:
                last_layer = film_layers[-1]
                nn.init.zeros_(last_layer.weight)
                if last_layer.bias is not None:
                    nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        z_geom: torch.Tensor,
        z_app: torch.Tensor,
        normalize: bool = True,
        mod_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode geometry and appearance back to descriptor space.

        Args:
            z_geom: Geometry latent, shape (B, d_geom)
            z_app: Appearance latent, shape (B, d_app)
            normalize: L2-normalize output
            mod_b: Target modality indices, shape (B,). Required when
                condition_decoder=True, ignored otherwise.

        Returns:
            d: Reconstructed descriptor, shape (B, D)
        """
        z = torch.cat([z_geom, z_app], dim=1)

        if self.condition_decoder:
            if mod_b is None:
                raise ValueError("mod_b is required when condition_decoder=True")
            h = self.fc1(z)
            # FiLM modulation on target modality
            e = self.mod_embed(mod_b)
            film_params = self.film_net(e)
            gamma, beta = film_params.chunk(2, dim=1)
            h = (1.0 + gamma) * h + beta
            d = self.fc2(h)
        else:
            d = self.decoder(z)

        if normalize:
            d = F.normalize(d, p=2, dim=1)

        return d


class AppearanceCrosser(nn.Module):
    """
    Transforms ONLY the appearance component between modalities.

    Geometry is preserved exactly - only appearance statistics change.
    Uses FiLM-style conditioning on modality pair.

    Has two modes:
    - simple=False (default): FiLM + residual MLP (more expressive, risk of overfitting)
    - simple=True: FiLM only (forces learning a general linear transformation)

    """

    def __init__(
        self,
        d_app: int,
        num_modalities: int,
        hidden_dim: int = 128,
        embed_dim: int = 16,
        simple: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_app = d_app
        self.num_modalities = num_modalities
        self.simple = simple

        # Modality embeddings
        self.mod_embed = nn.Embedding(num_modalities, embed_dim)

        # FiLM generator for appearance transformation
        self.film_net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 2 * d_app)  # gamma and beta
        )

        # Residual transformation (only used if not simple)
        if not simple:
            self.transform = nn.Sequential(
                nn.Linear(d_app, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, d_app)
            )
        else:
            self.transform = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z_app: torch.Tensor,
        mod_a: torch.Tensor,
        mod_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform appearance from modality A to modality B.

        Args:
            z_app: Appearance latent from modality A, shape (B, d_app)
            mod_a: Source modality indices, shape (B,)
            mod_b: Target modality indices, shape (B,)

        Returns:
            z_app_crossed: Transformed appearance, shape (B, d_app)
        """
        # Get modality embeddings
        e_a = self.mod_embed(mod_a)
        e_b = self.mod_embed(mod_b)

        # Generate FiLM parameters
        pair_embed = torch.cat([e_a, e_b], dim=1)
        film_params = self.film_net(pair_embed)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = 1.0 + gamma  # Center at 1 for near-identity init
        z_modulated = gamma * z_app + beta

        if self.simple:
            # Simple mode: just FiLM (affine transformation)
            return z_modulated
        else:
            # Full mode: FiLM + residual MLP
            delta = self.transform(z_modulated)
            z_app_crossed = z_app + delta
            return z_app_crossed


class DisentangledVAECrosser(nn.Module):
    """
    Stage 4: Disentangled VAE for Geometry-Preserving Cross-Modal Transformation.

    Key Insight:
    When crossing descriptors between modalities (e.g., T2 MRI → Ultrasound):
    - We want to transform APPEARANCE information (intensity patterns, contrast)
    - We want to PRESERVE GEOMETRY information (edges, corners, local structure)

    This is because both modalities image the SAME ANATOMY - geometry is shared,
    only appearance differs.

    Architecture:
        1. Encoder: d -> (z_geom, z_app)  - Disentangle geometry and appearance
        2. AppearanceCrosser: z_app_a -> z_app_b  - Transform only appearance
        3. Decoder: (z_geom, z_app_crossed) -> d_crossed  - Reconstruct with original geometry

    Training Losses:
        - Reconstruction: Both modalities should reconstruct accurately
        - Geometry Consistency: Same location = same z_geom across modalities
        - Crossing: Crossed descriptors should match target modality
        - Diversity: Prevent appearance collapse

    References:
        Based on the explicit decomposition approach D = A ∘ G from
        docs/preserving_geometric_invariance.md
    """

    def __init__(
        self,
        descriptor_dim: int,
        num_modalities: int,
        d_geom: int = 64,
        d_app: int = 64,
        hidden_dim: int = 256,
        embed_dim: int = 16,
        dropout: float = 0.0,
        simple_crosser: bool = False,
        separate_encoder: bool = False,
        normalize_geom: bool = False,
        geom_scale: float = 1.0,
        use_adversarial: bool = False,
        modality_specific_encoder: bool = False,
        # Conditioned decoder (FiLM on target modality)
        condition_decoder: bool = False,
        # FSQ quantization for z_geom (E2)
        fsq_levels: int = 0,
        # Variational encoding with reparameterization (E3)
        variational: bool = False,
        # F4: Only make z_app variational, keep z_geom deterministic
        variational_app_only: bool = False,
        # G7-2C auxiliary supervision heads (constructed with model for checkpoint compatibility)
        use_grad_head: bool = False,
        use_modality_heads: bool = False,
        use_coord_head: bool = False,
        # Independent hidden dim for AppearanceCrosser (None = hidden_dim // 2)
        crosser_hidden_dim: Optional[int] = None,
        # Ablation: when False, z_geom is also crossed (no geometry preservation)
        disentangle: bool = True,
    ):
        """
        Initialize DisentangledVAECrosser.

        Args:
            descriptor_dim: Dimension of input/output descriptors
            num_modalities: Number of distinct modalities
            d_geom: Dimension of geometry latent space
            d_app: Dimension of appearance latent space
            hidden_dim: Hidden layer dimension for encoder/decoder
            embed_dim: Dimension of modality embeddings
            dropout: Dropout rate (applied in appearance crosser)
            simple_crosser: If True, use FiLM-only crosser (no residual MLP) for better generalization
            separate_encoder: If True, use completely separate geometry/appearance encoders
            normalize_geom: If True, L2-normalize z_geom to prevent collapse to zero
            geom_scale: Scale factor for normalized z_geom (only used if normalize_geom=True)
            use_adversarial: If True, add modality discriminator with gradient reversal
            modality_specific_encoder: If True, use separate encoders per modality
                (most powerful option - each modality gets its own encoder network)
            condition_decoder: If True, use FiLM conditioning on target modality in decoder
            crosser_hidden_dim: Independent hidden dim for AppearanceCrosser.
                If None, defaults to hidden_dim // 2.
        """
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.num_modalities = num_modalities
        self.d_geom = d_geom
        self.d_app = d_app
        self.use_adversarial = use_adversarial
        self.modality_specific_encoder = modality_specific_encoder
        self.condition_decoder = condition_decoder
        if fsq_levels < 0 or fsq_levels == 1:
            raise ValueError(f"fsq_levels must be 0 (disabled) or >= 2, got {fsq_levels}")
        self.fsq_levels = fsq_levels
        self.variational = variational
        self.variational_app_only = variational_app_only
        self.use_grad_head = use_grad_head
        self.use_modality_heads = use_modality_heads
        self.use_coord_head = use_coord_head
        self.disentangle = disentangle
        # Resolve crosser hidden dim (default: half of encoder/decoder hidden_dim)
        resolved_crosser_hd = crosser_hidden_dim if crosser_hidden_dim is not None else hidden_dim // 2
        # Variational distribution state from last encode() call.
        # _last_dist: set by encode(), holds (mu_geom, logvar_geom, mu_app, logvar_app).
        # _last_dist_a/_last_dist_b: set by DisentangledVAELoss.forward() to capture
        #   distributions for modality A and B separately (used for KL computation).
        self._last_dist: Optional[Tuple[torch.Tensor, ...]] = None
        self._last_dist_a: Optional[Tuple[torch.Tensor, ...]] = None
        self._last_dist_b: Optional[Tuple[torch.Tensor, ...]] = None

        # Variational: add logvar heads for reparameterization trick
        if variational:
            if not variational_app_only:
                self.logvar_geom_head = nn.Linear(d_geom, d_geom)
            self.logvar_app_head = nn.Linear(d_app, d_app)

        # Encoder: d -> (z_geom, z_app)
        if modality_specific_encoder:
            # Each modality gets its own encoder
            self.encoder = ModalitySpecificEncoders(
                descriptor_dim=descriptor_dim,
                d_geom=d_geom,
                d_app=d_app,
                num_modalities=num_modalities,
                hidden_dim=hidden_dim,
                normalize_geom=normalize_geom,
                geom_scale=geom_scale,
            )
        else:
            # Shared encoder (original behavior)
            self.encoder = DisentangledEncoder(
                descriptor_dim=descriptor_dim,
                d_geom=d_geom,
                d_app=d_app,
                hidden_dim=hidden_dim,
                separate=separate_encoder,
                normalize_geom=normalize_geom,
                geom_scale=geom_scale,
            )

        # Appearance crosser (only transforms appearance)
        self.app_crosser = AppearanceCrosser(
            d_app=d_app,
            num_modalities=num_modalities,
            hidden_dim=resolved_crosser_hd,
            embed_dim=embed_dim,
            simple=simple_crosser,
            dropout=dropout,
        )

        # When disentangle=False, route ALL information through a single crosser.
        # No geometry/appearance split — one unified latent gets fully crossed.
        if not disentangle:
            self.full_crosser = AppearanceCrosser(
                d_app=d_geom + d_app,  # full latent dimension
                num_modalities=num_modalities,
                hidden_dim=resolved_crosser_hd,
                embed_dim=embed_dim,
                simple=simple_crosser,
                dropout=dropout,
            )
        else:
            self.full_crosser = None

        # Decoder: (z_geom, z_app) -> d
        self.decoder = DisentangledDecoder(
            descriptor_dim=descriptor_dim,
            d_geom=d_geom,
            d_app=d_app,
            hidden_dim=hidden_dim,
            condition_decoder=condition_decoder,
            num_modalities=num_modalities,
            embed_dim=embed_dim,
        )

        # Optional: Modality discriminator for adversarial training
        if use_adversarial:
            self.modality_discriminator = ModalityDiscriminator(
                input_dim=d_geom,
                hidden_dim=hidden_dim // 2,
                num_modalities=num_modalities,
            )
        else:
            self.modality_discriminator = None

        # G7-2C: Auxiliary supervision heads (gated by config flags).
        if use_grad_head:
            self.grad_head = nn.Sequential(
                nn.Linear(d_geom, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        else:
            self.grad_head = None

        if use_modality_heads:
            self.app_mod_head = nn.Linear(d_app, num_modalities)
            self.geom_mod_head = nn.Sequential(
                GradientReversal(lambda_=1.0),
                nn.Linear(d_geom, num_modalities),
            )
        else:
            self.app_mod_head = None
            self.geom_mod_head = None

        if use_coord_head:
            self.coord_head = nn.Sequential(
                nn.Linear(d_geom, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
            )
        else:
            self.coord_head = None

    @staticmethod
    def _fsq_quantize(z: torch.Tensor, levels: int) -> torch.Tensor:
        """Finite Scalar Quantization with straight-through estimator.

        Forward: quantized values. Backward: gradients flow through tanh.
        """
        if levels < 2:
            raise ValueError(f"levels must be >= 2, got {levels}")
        z_bounded = torch.tanh(z)
        # Map [-1, 1] -> [0, levels - 1], round to nearest bin, then map back.
        # This yields exactly `levels` bins for both odd and even levels.
        z_scaled = (z_bounded + 1.0) * (levels - 1) / 2.0
        z_bin = torch.round(z_scaled).clamp(0, levels - 1)
        z_quantized = (z_bin * 2.0 / (levels - 1)) - 1.0
        # STE: forward uses z_quantized, backward flows through z_bounded
        return z_bounded + (z_quantized - z_bounded).detach()

    def encode(
        self,
        d: torch.Tensor,
        modality: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode descriptor into geometry and appearance components.

        Args:
            d: Input descriptors, shape (B, D)
            modality: Modality indices, shape (B,). Required for modality-specific encoders.

        Returns:
            z_geom: Geometry latent, shape (B, d_geom)
            z_app: Appearance latent, shape (B, d_app)
        """
        # Clear stale distribution state from previous encode() call
        self._last_dist = None

        if self.modality_specific_encoder:
            mu_geom, mu_app = self.encoder(d, modality)
        else:
            mu_geom, mu_app = self.encoder(d)

        # Variational: reparameterization trick (E3)
        if self.variational:
            # Clamp logvar to [-10, 10] to prevent exp() overflow during
            # reparameterization (exp(0.5 * 10) ~ 148, exp(0.5 * 20) ~ 22026).
            # The prior logvar in DisentangledVAELoss is already clamped similarly.
            logvar_app = self.logvar_app_head(mu_app).clamp(-10, 10)

            if self.variational_app_only:
                # F4: z_geom is always deterministic
                z_geom = mu_geom
                logvar_geom = None
            else:
                logvar_geom = self.logvar_geom_head(mu_geom).clamp(-10, 10)
                if self.training:
                    z_geom = mu_geom + torch.exp(0.5 * logvar_geom) * torch.randn_like(mu_geom)
                else:
                    z_geom = mu_geom

            if self.training:
                z_app = mu_app + torch.exp(0.5 * logvar_app) * torch.randn_like(mu_app)
            else:
                z_app = mu_app

            self._last_dist = (mu_geom, logvar_geom, mu_app, logvar_app)
        else:
            self._last_dist = None
            z_geom, z_app = mu_geom, mu_app

        # FSQ quantization on z_geom (E2) — applied after reparameterization
        if self.fsq_levels > 0:
            z_geom = self._fsq_quantize(z_geom, self.fsq_levels)

        return z_geom, z_app

    def decode(
        self,
        z_geom: torch.Tensor,
        z_app: torch.Tensor,
        normalize: bool = True,
        mod_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode geometry and appearance back to descriptor space.

        Args:
            z_geom: Geometry latent, shape (B, d_geom)
            z_app: Appearance latent, shape (B, d_app)
            normalize: L2-normalize output
            mod_b: Target modality indices, shape (B,). Used when condition_decoder=True.

        Returns:
            d: Reconstructed descriptor, shape (B, D)
        """
        return self.decoder(z_geom, z_app, normalize, mod_b=mod_b)

    def reconstruct(
        self,
        d: torch.Tensor,
        modality: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode and decode (reconstruction path).

        Args:
            d: Input descriptors, shape (B, D)
            modality: Modality indices, shape (B,). Required for modality-specific encoders.
            normalize: L2-normalize output

        Returns:
            d_recon: Reconstructed descriptors, shape (B, D)
        """
        z_geom, z_app = self.encode(d, modality)
        return self.decode(z_geom, z_app, normalize, mod_b=modality)

    def forward(
        self,
        d_a: torch.Tensor,
        mod_a: torch.Tensor,
        mod_b: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Transform descriptors from modality A to modality B.

        PRESERVES geometry, only transforms appearance.

        Args:
            d_a: Source descriptors, shape (B, D)
            mod_a: Source modality indices, shape (B,)
            mod_b: Target modality indices, shape (B,)
            normalize: L2-normalize output

        Returns:
            d_crossed: Transformed descriptors, shape (B, D)
        """
        # 1. Encode: d_a -> (z_geom, z_app)
        z_geom, z_app = self.encode(d_a, mod_a)

        if self.disentangle:
            # Standard: preserve geometry, only cross appearance
            z_app_crossed = self.app_crosser(z_app, mod_a, mod_b)
            d_crossed = self.decode(z_geom, z_app_crossed, normalize, mod_b=mod_b)
        else:
            # Ablation: no split — concatenate into single latent, cross all
            z_full = torch.cat([z_geom, z_app], dim=1)
            z_full_crossed = self.full_crosser(z_full, mod_a, mod_b)
            z_geom_c = z_full_crossed[:, :self.d_geom]
            z_app_c = z_full_crossed[:, self.d_geom:]
            d_crossed = self.decode(z_geom_c, z_app_c, normalize, mod_b=mod_b)

        return d_crossed

    def forward_with_latents(
        self,
        d_a: torch.Tensor,
        mod_a: torch.Tensor,
        mod_b: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns latent representations.

        Useful for computing disentanglement losses during training.

        Args:
            d_a: Source descriptors, shape (B, D)
            mod_a: Source modality indices, shape (B,)
            mod_b: Target modality indices, shape (B,)
            normalize: L2-normalize output

        Returns:
            d_crossed: Transformed descriptors, shape (B, D)
            z_geom: Geometry latent, shape (B, d_geom)
            z_app: Source appearance latent, shape (B, d_app)
            z_app_crossed: Crossed appearance latent, shape (B, d_app)
        """
        z_geom, z_app = self.encode(d_a, mod_a)
        if self.disentangle:
            z_app_crossed = self.app_crosser(z_app, mod_a, mod_b)
            d_crossed = self.decode(z_geom, z_app_crossed, normalize, mod_b=mod_b)
        else:
            z_full = torch.cat([z_geom, z_app], dim=1)
            z_full_crossed = self.full_crosser(z_full, mod_a, mod_b)
            z_app_crossed = z_full_crossed[:, self.d_geom:]
            z_geom_c = z_full_crossed[:, :self.d_geom]
            d_crossed = self.decode(z_geom_c, z_app_crossed, normalize, mod_b=mod_b)

        return d_crossed, z_geom, z_app, z_app_crossed
