"""Disentangled VAE loss functions used by the upload pipeline."""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cross.geometry_losses import (
    PairwiseDistancePreservationLoss,
    SimilarityMatrixPreservationLoss,
)


class DisentangledVAELoss(nn.Module):
    """
    Combined loss function for training Disentangled VAE Crosser.

    The loss enforces:
    1. Reconstruction: Descriptors should be accurately reconstructed
    2. Geometry Consistency: Same location across modalities → same z_geom
    3. Crossing: Crossed descriptors should match target modality
    4. Diversity: Prevent appearance latent collapse to mean
    5. Output Geometry Preservation: Pairwise distances preserved from input to output

    L = λ_recon * L_recon + λ_geom * L_geom + λ_cross * L_cross + λ_div * L_div + λ_geo * L_geo

    Key insight: Without geometry consistency loss, the model may encode
    modality-specific information in z_geom, defeating the purpose of
    the disentanglement.

    NEW: Output geometry preservation ensures that the crossed descriptors maintain
    the same pairwise distance structure as the input descriptors.
    """

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_geom_consistency: float = 0.5,
        lambda_crossing: float = 1.0,
        lambda_diversity: float = 0.1,
        lambda_contrastive: float = 0.5,
        temperature: float = 0.07,
        min_geom_norm: float = 0.0,
        lambda_adversarial: float = 0.0,
        # NEW: Output geometry preservation parameters
        lambda_geometry: float = 0.0,
        geometry_type: str = "similarity",
        z_app_dropout: float = 0.0,
        # Latent-space contrastive disentanglement (E1)
        lambda_latent_contrastive: float = 0.0,
        lambda_latent_leakage: float = 0.0,
        latent_leakage_type: str = "cross_cov",
        lambda_app_repulsion: float = 0.0,
        # Supervised modality contrastive on z_app (DC-Seg-inspired)
        lambda_app_mod_supcon: float = 0.0,
        app_mod_supcon_temperature: float = 0.07,
        # Functional Test-4 proxy: enforce geom-only > app-only
        lambda_t4_margin: float = 0.0,
        t4_margin: float = 0.10,
        # Test-1 proxy: preserve cross-modality sensitivity ratio (Delta_app / Delta_geom)
        lambda_t1_margin: float = 0.0,
        t1_margin_target: float = 2.0,
        t1_margin_min_delta: float = 0.05,
        # Probe-C surrogate: enforce gradient predictability in z_geom over z_app
        lambda_probe_c_margin: float = 0.0,
        probe_c_target_gap: float = 0.05,
        probe_c_ridge: float = 1e-3,
        # G7-2D: Probe-C inversion — swap z_geom/z_app roles
        probe_c_invert: bool = False,
        # Probe-B/C dual surrogate: keep intensity in z_app while gradient stays in z_geom
        lambda_probe_bc_margin: float = 0.0,
        probe_bc_target_b: float = 0.05,
        probe_bc_target_c: float = 0.05,
        # Variational iVAE with conditional prior (E3)
        lambda_kl_geom: float = 0.0,
        lambda_kl_app: float = 0.0,
        num_modalities: int = 2,
        d_app: int = 64,
        # F5: Free bits — per-dimension KL floor to prevent posterior collapse
        free_bits: float = 0.0,
        # G7-2B: Geom-regularizer loss-weight scaling
        geom_reg_scale: float = 1.0,
        # G7-2C: Auxiliary supervision heads
        lambda_grad_head: float = 0.0,
        lambda_modality_heads: float = 0.0,
        lambda_coord_head: float = 0.0,
        aux_warmup_epochs: int = 10,
        # Hard negative mining for InfoNCE (matching MLP recipe)
        hard_negative_ratio: float = 0.0,
        hard_negative_weight: float = 2.0,
        # Tier 1 improvements
        symmetric_infonce: bool = False,
        stop_grad_z_geom_recon: bool = False,
        lambda_vicreg_cov: float = 0.0,
        descriptor_noise_std: float = 0.0,
    ):
        """
        Initialize DisentangledVAELoss.

        Args:
            lambda_recon: Weight for reconstruction loss
            lambda_geom_consistency: Weight for geometry consistency loss
            lambda_crossing: Weight for crossing loss (cosine similarity)
            lambda_diversity: Weight for appearance diversity loss
            lambda_contrastive: Weight for contrastive loss on crossed descriptors
            temperature: Temperature for InfoNCE contrastive loss
            min_geom_norm: Minimum expected L2 norm for z_geom (penalize collapse if below)
            lambda_adversarial: Weight for adversarial loss (requires model.use_adversarial=True)
            lambda_geometry: Weight for OUTPUT geometry preservation loss (NEW!)
            geometry_type: Type of geometry loss - "similarity", "distance", or "combined"
        """
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_geom_consistency = lambda_geom_consistency
        self.lambda_crossing = lambda_crossing
        self.lambda_diversity = lambda_diversity
        self.lambda_contrastive = lambda_contrastive
        self.temperature = temperature
        self.min_geom_norm = min_geom_norm
        self.lambda_adversarial = lambda_adversarial

        # NEW: Output geometry preservation
        self.lambda_geometry = lambda_geometry
        self.geometry_type = geometry_type
        self.z_app_dropout = z_app_dropout
        self.lambda_latent_contrastive = lambda_latent_contrastive
        self.lambda_latent_leakage = lambda_latent_leakage
        self.latent_leakage_type = latent_leakage_type
        self.lambda_app_repulsion = lambda_app_repulsion
        self.lambda_app_mod_supcon = lambda_app_mod_supcon
        self.app_mod_supcon_temperature = app_mod_supcon_temperature
        self.lambda_t4_margin = lambda_t4_margin
        self.t4_margin = t4_margin
        self.lambda_t1_margin = lambda_t1_margin
        self.t1_margin_target = t1_margin_target
        self.t1_margin_min_delta = t1_margin_min_delta
        self.lambda_probe_c_margin = lambda_probe_c_margin
        self.probe_c_target_gap = probe_c_target_gap
        self.probe_c_ridge = probe_c_ridge
        self.probe_c_invert = probe_c_invert
        self.lambda_probe_bc_margin = lambda_probe_bc_margin
        self.probe_bc_target_b = probe_bc_target_b
        self.probe_bc_target_c = probe_bc_target_c
        self.lambda_kl_geom = lambda_kl_geom
        self.lambda_kl_app = lambda_kl_app
        self.free_bits = free_bits
        self.geom_reg_scale = geom_reg_scale
        self.lambda_grad_head = lambda_grad_head
        self.lambda_modality_heads = lambda_modality_heads
        self.lambda_coord_head = lambda_coord_head
        self.aux_warmup_epochs = aux_warmup_epochs
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.hard_negative_weight = float(hard_negative_weight)
        self.symmetric_infonce = symmetric_infonce
        self.stop_grad_z_geom_recon = stop_grad_z_geom_recon
        self.lambda_vicreg_cov = lambda_vicreg_cov
        self.descriptor_noise_std = descriptor_noise_std
        self.supports_probe_c_labels = True
        self.supports_probe_bc_labels = True

        if probe_c_ridge <= 0:
            raise ValueError(f"probe_c_ridge must be > 0, got {probe_c_ridge}")
        if t1_margin_min_delta < 0:
            raise ValueError(f"t1_margin_min_delta must be >= 0, got {t1_margin_min_delta}")

        # Warn about double-counting: probe_c and probe_bc both penalize gradient gap
        if lambda_probe_c_margin > 0 and lambda_probe_bc_margin > 0:
            warnings.warn(
                "Both probe_c_margin and probe_bc_margin are active. "
                "The gradient predictability (probe_c) component in probe_bc will be "
                "skipped to avoid double-counting — probe_bc will only enforce the "
                "intensity (probe_b) component.",
                stacklevel=2,
            )

        # Warn about redundant GRL mechanisms on z_geom
        if lambda_adversarial > 0 and lambda_modality_heads > 0:
            warnings.warn(
                "Both adversarial and modality_heads are active. Both apply gradient "
                "reversal to z_geom for modality invariance. Effective strength: "
                f"adversarial={lambda_adversarial} + 0.1*modality_heads="
                f"{0.1 * lambda_modality_heads:.4f} = "
                f"{lambda_adversarial + 0.1 * lambda_modality_heads:.4f}. "
                "Consider using only one mechanism.",
                stacklevel=2,
            )
        if latent_leakage_type not in {"cross_cov", "hsic"}:
            raise ValueError(
                "latent_leakage_type must be one of {'cross_cov', 'hsic'}, "
                f"got {latent_leakage_type!r}"
            )

        # Learnable per-modality prior for z_app (iVAE conditional prior)
        if lambda_kl_app > 0:
            self.app_prior_mu = nn.Parameter(torch.zeros(num_modalities, d_app))
            self.app_prior_logvar = nn.Parameter(torch.zeros(num_modalities, d_app))

        if lambda_geometry > 0:
            if geometry_type == "similarity":
                self.geometry_loss = SimilarityMatrixPreservationLoss()
            elif geometry_type == "distance":
                self.geometry_loss = PairwiseDistancePreservationLoss()
            elif geometry_type == "combined":
                self.geometry_loss_sim = SimilarityMatrixPreservationLoss()
                self.geometry_loss_dist = PairwiseDistancePreservationLoss()

    def forward(
        self,
        model: nn.Module,
        d_a: torch.Tensor,
        d_b: torch.Tensor,
        mod_a: torch.Tensor,
        mod_b: torch.Tensor,
        return_components: bool = False,
        intensities_a: Optional[torch.Tensor] = None,
        intensities_b: Optional[torch.Tensor] = None,
        grad_magnitudes_a: Optional[torch.Tensor] = None,
        grad_magnitudes_b: Optional[torch.Tensor] = None,
        epoch: int = 1,
        coords_a: Optional[torch.Tensor] = None,
        coords_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined disentangled VAE loss.

        Args:
            model: DisentangledVAECrosser model
            d_a: Descriptors from modality A, shape (B, D)
            d_b: Descriptors from modality B at SAME locations, shape (B, D)
            mod_a: Source modality indices, shape (B,)
            mod_b: Target modality indices, shape (B,)
            return_components: If True, return dict with individual losses
            intensities_a: Optional local intensity labels for modality A (B,)
            intensities_b: Optional local intensity labels for modality B (B,)
            grad_magnitudes_a: Optional local gradient magnitudes for modality A (B,)
            grad_magnitudes_b: Optional local gradient magnitudes for modality B (B,)

        Returns:
            Total loss (and optionally component dict)
        """
        losses = {}
        # requires_grad=True needed for edge case where all lambda weights are zero
        total = torch.tensor(0.0, device=d_a.device, requires_grad=True)

        # Gaussian noise injection on input descriptors (training only)
        if self.descriptor_noise_std > 0 and model.training:
            d_a = d_a + torch.randn_like(d_a) * self.descriptor_noise_std
            d_b = d_b + torch.randn_like(d_b) * self.descriptor_noise_std

        # Encode both modalities (pass modality for modality-specific encoders)
        # Capture variational distributions after each encode call (for iVAE KL losses)
        # Clear stale distribution state before encoding to prevent previous-batch leakage
        model._last_dist_a = None
        model._last_dist_b = None
        z_geom_a, z_app_a = model.encode(d_a, mod_a)
        dist_a = getattr(model, '_last_dist', None)
        if dist_a is not None:
            model._last_dist_a = dist_a
        z_geom_b, z_app_b = model.encode(d_b, mod_b)
        dist_b = getattr(model, '_last_dist', None)
        if dist_b is not None:
            model._last_dist_b = dist_b

        # Save pre-dropout z_app for probe-C/BC computations.  Probes measure
        # the *true* predictability gap between z_geom and z_app; using the
        # dropout-modified z_app would artificially inflate the gap, giving a
        # biased training signal.
        z_app_a_predropout = z_app_a
        z_app_b_predropout = z_app_b

        # Apply z_app dropout: replace random samples' z_app with batch mean.
        # Uses model.training (not self.training) because evaluate_cross_patient()
        # in train.py calls model.eval() but NOT loss_fn.eval().
        if self.z_app_dropout > 0 and model.training:
            mask_a = torch.rand(z_app_a.shape[0], 1, device=z_app_a.device) > self.z_app_dropout
            z_app_mean_a = z_app_a.detach().mean(dim=0, keepdim=True)
            z_app_a = torch.where(mask_a, z_app_a, z_app_mean_a.expand_as(z_app_a))

            mask_b = torch.rand(z_app_b.shape[0], 1, device=z_app_b.device) > self.z_app_dropout
            z_app_mean_b = z_app_b.detach().mean(dim=0, keepdim=True)
            z_app_b = torch.where(mask_b, z_app_b, z_app_mean_b.expand_as(z_app_b))

        # Pre-compute crossed appearance and crossed descriptors once.
        # These are reused by crossing loss (step 3), contrastive loss (step 5),
        # and geometry preservation loss (step 7).
        needs_crossed = (
            self.lambda_crossing > 0
            or self.lambda_contrastive > 0
            or self.lambda_geometry > 0
            or self.lambda_t4_margin > 0
        )
        z_app_a_crossed = None
        d_a_crossed = None
        d_b_norm = None
        if needs_crossed:
            if getattr(model, 'disentangle', True):
                z_app_a_crossed = model.app_crosser(z_app_a, mod_a, mod_b)
                z_geom_for_decode = z_geom_a
                z_app_for_decode = z_app_a_crossed
            else:
                z_full = torch.cat([z_geom_a, z_app_a], dim=1)
                z_full_crossed = model.full_crosser(z_full, mod_a, mod_b)
                z_geom_for_decode = z_full_crossed[:, :model.d_geom]
                z_app_for_decode = z_full_crossed[:, model.d_geom:]
            d_a_crossed = model.decode(z_geom_for_decode, z_app_for_decode, normalize=True, mod_b=mod_b)
            d_b_norm = F.normalize(d_b, p=2, dim=1)

        # Stash crossed prediction so the trainer can reuse it for metrics
        # instead of running a redundant forward pass.
        self._last_d_a_crossed = d_a_crossed.detach() if d_a_crossed is not None else None

        # 1. Reconstruction loss - both modalities should reconstruct
        if self.lambda_recon > 0:
            # Stop-gradient on z_geom prevents it from encoding modality info
            # for reconstruction (MMVAE++ insight: z_geom should only receive
            # gradients from the cross-modal crossing path).
            z_geom_a_recon = z_geom_a.detach() if self.stop_grad_z_geom_recon else z_geom_a
            z_geom_b_recon = z_geom_b.detach() if self.stop_grad_z_geom_recon else z_geom_b
            d_a_recon = model.decode(z_geom_a_recon, z_app_a, normalize=True, mod_b=mod_a)
            d_b_recon = model.decode(z_geom_b_recon, z_app_b, normalize=True, mod_b=mod_b)

            # Use cosine similarity for reconstruction (more stable than MSE for normalized vectors)
            recon_loss_a = 1.0 - F.cosine_similarity(d_a_recon, F.normalize(d_a, p=2, dim=1), dim=1).mean()
            recon_loss_b = 1.0 - F.cosine_similarity(d_b_recon, F.normalize(d_b, p=2, dim=1), dim=1).mean()
            recon_loss = recon_loss_a + recon_loss_b

            losses["recon"] = recon_loss
            total = total + self.lambda_recon * recon_loss

        # 2. Geometry consistency loss - same location = same geometry
        #    This is the KEY constraint for disentanglement!
        if self.lambda_geom_consistency > 0 and getattr(model, 'disentangle', True):
            # z_geom should be the same for aligned pairs (same anatomy, different modality)
            geom_consistency = F.mse_loss(z_geom_a, z_geom_b)
            losses["geom_consistency"] = geom_consistency
            total = total + self._scaled_geom_weight(self.lambda_geom_consistency) * geom_consistency

        # 3. Crossing loss - crossed descriptors should match target
        if self.lambda_crossing > 0:
            assert d_a_crossed is not None and d_b_norm is not None
            crossing_loss = 1.0 - F.cosine_similarity(d_a_crossed, d_b_norm, dim=1).mean()
            losses["crossing"] = crossing_loss
            total = total + self.lambda_crossing * crossing_loss

        # 4. Appearance diversity loss - prevent collapse to mean
        if self.lambda_diversity > 0:
            # Variance across batch should be maintained
            app_var_a = z_app_a.var(dim=0).mean()
            app_var_b = z_app_b.var(dim=0).mean()

            # Also ensure geometry has variance (it encodes different locations)
            geom_var_a = z_geom_a.var(dim=0).mean()
            geom_var_b = z_geom_b.var(dim=0).mean()

            # Target variance of ~1.0 (assuming normalized inputs)
            target_var = 1.0
            diversity_loss = (
                F.relu(target_var - app_var_a) +
                F.relu(target_var - app_var_b) +
                F.relu(target_var - geom_var_a) +
                F.relu(target_var - geom_var_b)
            )

            # Add minimum norm constraint for z_geom to prevent collapse to zero
            if self.min_geom_norm > 0:
                geom_norm_a = z_geom_a.norm(dim=1).mean()
                geom_norm_b = z_geom_b.norm(dim=1).mean()
                norm_penalty = (
                    F.relu(self.min_geom_norm - geom_norm_a) +
                    F.relu(self.min_geom_norm - geom_norm_b)
                )
                diversity_loss = diversity_loss + norm_penalty

            losses["diversity"] = diversity_loss
            total = total + self.lambda_diversity * diversity_loss

        # 4b. VICReg covariance regularization on z_geom
        if self.lambda_vicreg_cov > 0:
            z_geom_all = torch.cat([z_geom_a, z_geom_b], dim=0)
            z_centered = z_geom_all - z_geom_all.mean(dim=0, keepdim=True)
            n = z_centered.shape[0]
            cov_matrix = (z_centered.T @ z_centered) / (n - 1)
            d = cov_matrix.shape[0]
            # Penalize off-diagonal elements (dimension correlations)
            off_diag = cov_matrix.pow(2).sum() - cov_matrix.diagonal().pow(2).sum()
            vicreg_cov = off_diag / d
            losses["vicreg_cov"] = vicreg_cov
            total = total + self.lambda_vicreg_cov * vicreg_cov

        # 5. Contrastive loss on crossed descriptors
        if self.lambda_contrastive > 0:
            assert d_a_crossed is not None and d_b_norm is not None
            contrastive = self._infonce_loss(d_a_crossed, d_b_norm)
            losses["contrastive"] = contrastive
            total = total + self.lambda_contrastive * contrastive

        # 6. Adversarial loss - force z_geom to be modality-invariant
        if self.lambda_adversarial > 0 and model.modality_discriminator is not None:
            # Concatenate z_geom from both modalities
            z_geom_all = torch.cat([z_geom_a, z_geom_b], dim=0)
            # Use actual modality indices (supports multi-modality universal training)
            labels_all = torch.cat([mod_a, mod_b])

            # Get discriminator predictions (with gradient reversal)
            logits = model.modality_discriminator(z_geom_all, reverse_grad=True)

            # Cross-entropy loss (discriminator tries to predict modality)
            # Due to gradient reversal, encoder tries to MAXIMIZE this loss
            # (i.e., make z_geom unpredictable for modality)
            adversarial_loss = F.cross_entropy(logits, labels_all)
            losses["adversarial"] = adversarial_loss
            total = total + self.lambda_adversarial * adversarial_loss

        # 7. OUTPUT Geometry preservation loss - preserve pairwise structure from input to output
        #    This is DIFFERENT from geom_consistency which operates on latent z_geom!
        if self.lambda_geometry > 0:
            assert d_a_crossed is not None
            d_a_norm = F.normalize(d_a, p=2, dim=1)

            if self.geometry_type == "combined":
                geo_sim = self.geometry_loss_sim(d_a_norm, d_a_crossed)
                geo_dist = self.geometry_loss_dist(d_a_norm, d_a_crossed)
                geo_loss = 0.5 * geo_sim + 0.5 * geo_dist
                losses["geo_similarity"] = geo_sim
                losses["geo_distance"] = geo_dist
            else:
                geo_loss = self.geometry_loss(d_a_norm, d_a_crossed)

            losses["geometry"] = geo_loss
            total = total + self.lambda_geometry * geo_loss

        # 8. Latent contrastive: InfoNCE on z_geom across modalities
        if self.lambda_latent_contrastive > 0:
            latent_contrastive = self._infonce_loss(z_geom_a, z_geom_b)
            losses["latent_contrastive"] = latent_contrastive
            total = total + self._scaled_geom_weight(self.lambda_latent_contrastive) * latent_contrastive

        # 9. App repulsion: push z_app apart for same-keypoint cross-modal pairs
        if self.lambda_app_repulsion > 0:
            app_repulsion = self._app_repulsion_loss(z_app_a, z_app_b)
            losses["app_repulsion"] = app_repulsion
            total = total + self.lambda_app_repulsion * app_repulsion

        # 9a. Explicit latent leakage penalty: discourage shared factors between z_geom and z_app.
        if self.lambda_latent_leakage > 0:
            leakage_a = self._latent_leakage_penalty(z_geom_a, z_app_a)
            leakage_b = self._latent_leakage_penalty(z_geom_b, z_app_b)
            latent_leakage = 0.5 * (leakage_a + leakage_b)
            losses["latent_leakage"] = latent_leakage
            total = total + self.lambda_latent_leakage * latent_leakage

        # 9b. Supervised modality contrastive on z_app.
        #     Same modality latents should be closer than different modalities.
        if self.lambda_app_mod_supcon > 0:
            z_app_all = torch.cat([z_app_a, z_app_b], dim=0)
            labels_all = torch.cat([mod_a, mod_b], dim=0)
            app_mod_supcon = self._supervised_contrastive_loss(
                z_app_all, labels_all, temperature=self.app_mod_supcon_temperature
            )
            losses["app_mod_supcon"] = app_mod_supcon
            total = total + self.lambda_app_mod_supcon * app_mod_supcon

        # 9c. Functional T4 proxy: enforce geom-only branch to outperform app-only branch.
        if self.lambda_t4_margin > 0:
            assert z_app_a_crossed is not None and d_b_norm is not None

            n = z_geom_a.shape[0]
            z_app_mean = z_app_a_crossed.mean(dim=0, keepdim=True).expand(n, -1)
            z_geom_mean = z_geom_a.mean(dim=0, keepdim=True).expand(n, -1)

            d_geom_only = model.decode(z_geom_a, z_app_mean, normalize=True, mod_b=mod_b)
            d_app_only = model.decode(z_geom_mean, z_app_a_crossed, normalize=True, mod_b=mod_b)

            s_geom = F.cosine_similarity(d_geom_only, d_b_norm, dim=1).mean()
            s_app = F.cosine_similarity(d_app_only, d_b_norm, dim=1).mean()
            t4_margin_loss = F.relu(self.t4_margin - (s_geom - s_app))

            losses["t4_margin"] = t4_margin_loss
            losses["t4_geom_sim"] = s_geom
            losses["t4_app_sim"] = s_app
            total = total + self.lambda_t4_margin * t4_margin_loss

        # 9d. Test-1 proxy: preserve appearance-over-geometry sensitivity under modality swap.
        if self.lambda_t1_margin > 0:
            t1_ratio, delta_geom_norm, delta_app_norm = self._t1_sensitivity_ratio(
                z_geom_a, z_geom_b, z_app_a, z_app_b
            )
            inconclusive = (
                (delta_geom_norm < self.t1_margin_min_delta)
                & (delta_app_norm < self.t1_margin_min_delta)
            )
            raw_t1_margin = F.relu(self.t1_margin_target - t1_ratio)
            t1_margin_loss = torch.where(
                inconclusive,
                torch.zeros_like(raw_t1_margin),
                raw_t1_margin,
            )

            losses["t1_margin"] = t1_margin_loss
            losses["t1_ratio"] = t1_ratio
            losses["t1_delta_geom_norm"] = delta_geom_norm
            losses["t1_delta_app_norm"] = delta_app_norm
            losses["t1_inconclusive"] = inconclusive.float()
            total = total + self.lambda_t1_margin * t1_margin_loss

        # 9e. Probe-C surrogate: enforce gradient predictability in z_geom over z_app.
        #     G7-2D: When probe_c_invert=True, swap z_geom/z_app roles.
        if (
            self.lambda_probe_c_margin > 0
            and grad_magnitudes_a is not None
            and grad_magnitudes_b is not None
        ):
            gap_a, r2_geom_a, r2_app_a = self._probe_c_predictability_gap(
                z_geom_a, z_app_a_predropout, grad_magnitudes_a
            )
            gap_b, r2_geom_b, r2_app_b = self._probe_c_predictability_gap(
                z_geom_b, z_app_b_predropout, grad_magnitudes_b
            )
            gap = 0.5 * (gap_a + gap_b)
            probe_c_margin_loss = F.relu(self.probe_c_target_gap - gap)

            losses["probe_c_margin"] = probe_c_margin_loss
            losses["probe_c_gap"] = gap
            if self.probe_c_invert:
                # Inverted: r2 roles are swapped, rename keys to avoid confusion
                losses["probe_c_inverted"] = torch.tensor(1.0, device=d_a.device)
                losses["probe_c_r2_target_app"] = 0.5 * (r2_geom_a + r2_geom_b)
                losses["probe_c_r2_target_geom"] = 0.5 * (r2_app_a + r2_app_b)
            else:
                losses["probe_c_r2_geom"] = 0.5 * (r2_geom_a + r2_geom_b)
                losses["probe_c_r2_app"] = 0.5 * (r2_app_a + r2_app_b)
            total = total + self._scaled_geom_weight(self.lambda_probe_c_margin) * probe_c_margin_loss

        # 9f. Probe-B/C dual surrogate:
        #     keep intensity predictability in z_app (Probe-B) and
        #     gradient predictability in z_geom (Probe-C).
        #
        #     BUG FIX: When probe_c_margin is already active (step 9e), the gradient
        #     gap (probe_c component) would be penalized twice — once via probe_c_margin
        #     and again via probe_bc_margin. To avoid double-counting, skip the
        #     probe_c component here when probe_c_margin is already handling it.
        if (
            self.lambda_probe_bc_margin > 0
            and intensities_a is not None
            and intensities_b is not None
            and grad_magnitudes_a is not None
            and grad_magnitudes_b is not None
        ):
            skip_gradient = self.lambda_probe_c_margin > 0
            (
                gap_b_a,
                gap_c_a,
                r2_geom_int_a,
                r2_app_int_a,
                r2_geom_grad_a,
                r2_app_grad_a,
            ) = self._probe_bc_predictability_gaps(
                z_geom_a, z_app_a_predropout, intensities_a, grad_magnitudes_a
            )
            (
                gap_b_b,
                gap_c_b,
                r2_geom_int_b,
                r2_app_int_b,
                r2_geom_grad_b,
                r2_app_grad_b,
            ) = self._probe_bc_predictability_gaps(
                z_geom_b, z_app_b_predropout, intensities_b, grad_magnitudes_b
            )

            probe_b_loss = 0.5 * (
                F.relu(self.probe_bc_target_b - gap_b_a)
                + F.relu(self.probe_bc_target_b - gap_b_b)
            )
            if skip_gradient:
                # Gradient gap already penalized by probe_c_margin (step 9e)
                probe_c_loss = torch.zeros_like(probe_b_loss)
            else:
                probe_c_loss = 0.5 * (
                    F.relu(self.probe_bc_target_c - gap_c_a)
                    + F.relu(self.probe_bc_target_c - gap_c_b)
                )
            probe_bc_margin_loss = probe_b_loss + probe_c_loss

            losses["probe_bc_margin"] = probe_bc_margin_loss
            losses["probe_bc_loss_b"] = probe_b_loss
            losses["probe_bc_loss_c"] = probe_c_loss
            losses["probe_bc_gap_b"] = 0.5 * (gap_b_a + gap_b_b)
            losses["probe_bc_gap_c"] = 0.5 * (gap_c_a + gap_c_b)
            losses["probe_bc_r2_geom_int"] = 0.5 * (r2_geom_int_a + r2_geom_int_b)
            losses["probe_bc_r2_app_int"] = 0.5 * (r2_app_int_a + r2_app_int_b)
            losses["probe_bc_r2_geom_grad"] = 0.5 * (r2_geom_grad_a + r2_geom_grad_b)
            losses["probe_bc_r2_app_grad"] = 0.5 * (r2_app_grad_a + r2_app_grad_b)
            total = total + self.lambda_probe_bc_margin * probe_bc_margin_loss

        # 10a. KL for z_geom toward N(0,I)
        if self.lambda_kl_geom > 0:
            dist_a_saved = getattr(model, '_last_dist_a', None)
            dist_b_saved = getattr(model, '_last_dist_b', None)
            if dist_a_saved is not None and dist_b_saved is not None:
                mu_g_a, lv_g_a, _, _ = dist_a_saved
                mu_g_b, lv_g_b, _, _ = dist_b_saved
                if lv_g_a is not None and lv_g_b is not None:
                    kl_a = -0.5 * (1 + lv_g_a - mu_g_a.pow(2) - lv_g_a.exp()).mean()
                    kl_b = -0.5 * (1 + lv_g_b - mu_g_b.pow(2) - lv_g_b.exp()).mean()
                    kl_geom = 0.5 * (kl_a + kl_b)
                else:
                    # variational_app_only: z_geom is deterministic, L2 fallback
                    kl_geom = 0.5 * (mu_g_a.pow(2).mean() + mu_g_b.pow(2).mean())
            else:
                # Deterministic encoder — L2 fallback
                kl_geom = 0.5 * (z_geom_a.pow(2).mean() + z_geom_b.pow(2).mean())
            losses["kl_geom"] = kl_geom
            total = total + self._scaled_geom_weight(self.lambda_kl_geom) * kl_geom

        # 10b. KL for z_app toward learned per-modality prior
        if self.lambda_kl_app > 0:
            prior_mu_a = self.app_prior_mu[mod_a]
            prior_lv_a = self.app_prior_logvar[mod_a].clamp(min=-10.0, max=10.0)
            prior_mu_b = self.app_prior_mu[mod_b]
            prior_lv_b = self.app_prior_logvar[mod_b].clamp(min=-10.0, max=10.0)

            dist_a_saved = getattr(model, '_last_dist_a', None)
            dist_b_saved = getattr(model, '_last_dist_b', None)
            if dist_a_saved is not None and dist_b_saved is not None:
                _, _, mu_a_a, lv_a_a = dist_a_saved
                _, _, mu_a_b, lv_a_b = dist_b_saved
                if lv_a_a is not None and lv_a_b is not None:
                    # Variational — full Gaussian KL: KL(q(z|x) || p(z|u))
                    # Compute per-dim KL for free bits support
                    kl_per_dim_a = -0.5 * (1 + lv_a_a - prior_lv_a
                                - ((mu_a_a - prior_mu_a).pow(2) + lv_a_a.exp()) / prior_lv_a.exp())
                    kl_per_dim_b = -0.5 * (1 + lv_a_b - prior_lv_b
                                - ((mu_a_b - prior_mu_b).pow(2) + lv_a_b.exp()) / prior_lv_b.exp())
                    # Average over batch, then apply free bits, then average over dims
                    kl_app_a = torch.clamp(kl_per_dim_a.mean(dim=0) - self.free_bits, min=0.0).mean()
                    kl_app_b = torch.clamp(kl_per_dim_b.mean(dim=0) - self.free_bits, min=0.0).mean()
                else:
                    # variational_app_only but lv_a should always exist; defensive
                    kl_app_a = F.mse_loss(z_app_a, prior_mu_a)
                    kl_app_b = F.mse_loss(z_app_b, prior_mu_b)
            else:
                # Deterministic — MSE fallback toward prior mean
                kl_app_a = F.mse_loss(z_app_a, prior_mu_a)
                kl_app_b = F.mse_loss(z_app_b, prior_mu_b)
            kl_app = 0.5 * (kl_app_a + kl_app_b)
            losses["kl_app"] = kl_app
            total = total + self.lambda_kl_app * kl_app

        # 11. G7-2C: Auxiliary supervision heads with warmup
        warmup = self._aux_warmup_factor(epoch)

        if (
            self.lambda_grad_head > 0
            and grad_magnitudes_a is not None
            and grad_magnitudes_b is not None
            and getattr(model, "grad_head", None) is not None
        ):
            pred_a = model.grad_head(z_geom_a).squeeze(-1)
            pred_b = model.grad_head(z_geom_b).squeeze(-1)
            grad_head_loss = 0.5 * (
                F.mse_loss(pred_a, grad_magnitudes_a)
                + F.mse_loss(pred_b, grad_magnitudes_b)
            )
            losses["grad_head"] = grad_head_loss
            total = total + self.lambda_grad_head * warmup * grad_head_loss

        if self.lambda_modality_heads > 0 and getattr(model, "app_mod_head", None) is not None:
            # z_app predicts modality (standard CE)
            app_logits_a = model.app_mod_head(z_app_a)
            app_logits_b = model.app_mod_head(z_app_b)
            app_mod_loss = 0.5 * (
                F.cross_entropy(app_logits_a, mod_a)
                + F.cross_entropy(app_logits_b, mod_b)
            )
            losses["app_mod_head"] = app_mod_loss
            total = total + self.lambda_modality_heads * warmup * app_mod_loss

            # z_geom confused via GRL (separate head)
            if getattr(model, "geom_mod_head", None) is not None:
                geom_logits_a = model.geom_mod_head(z_geom_a)
                geom_logits_b = model.geom_mod_head(z_geom_b)
                geom_mod_loss = 0.5 * (
                    F.cross_entropy(geom_logits_a, mod_a)
                    + F.cross_entropy(geom_logits_b, mod_b)
                )
                losses["geom_mod_head"] = geom_mod_loss
                total = total + 0.1 * self.lambda_modality_heads * warmup * geom_mod_loss

        if (
            self.lambda_coord_head > 0
            and coords_a is not None
            and coords_b is not None
            and getattr(model, "coord_head", None) is not None
        ):
            pred_coords_a = model.coord_head(z_geom_a)
            pred_coords_b = model.coord_head(z_geom_b)
            coord_head_loss = 0.5 * (
                F.mse_loss(pred_coords_a, coords_a)
                + F.mse_loss(pred_coords_b, coords_b)
            )
            losses["coord_head"] = coord_head_loss
            total = total + self.lambda_coord_head * warmup * coord_head_loss

        losses["total"] = total

        if return_components:
            return total, losses
        return total

    def _scaled_geom_weight(self, base_weight: float) -> float:
        """Apply G7-2B geom-regularizer scaling to a loss weight."""
        return base_weight * self.geom_reg_scale

    def _aux_warmup_factor(self, epoch: int) -> float:
        """Linear warmup factor for G7-2C auxiliary heads."""
        if self.aux_warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.aux_warmup_epochs)

    def _infonce_loss(
        self,
        d_pred: torch.Tensor,
        d_target: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE contrastive loss with optional hard negative mining."""
        batch_size = len(d_pred)

        d_pred = F.normalize(d_pred, p=2, dim=1)
        d_target = F.normalize(d_target, p=2, dim=1)

        logits = d_pred @ d_target.T / self.temperature

        # Apply hard negative mining (upweight hardest negatives)
        if (
            self.hard_negative_ratio > 0.0
            and batch_size > 1
            and self.hard_negative_weight != 1.0
        ):
            k = max(1, int((batch_size - 1) * self.hard_negative_ratio))
            k = min(k, batch_size - 1)

            with torch.no_grad():
                neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=logits.device)
                neg_logits = logits.masked_fill(~neg_mask, float("-inf"))
                hard_indices = neg_logits.topk(k, dim=1).indices

            log_w = float(math.log(self.hard_negative_weight))
            log_weights = logits.new_zeros(logits.shape)
            row_idx = torch.arange(batch_size, device=logits.device).unsqueeze(1)
            log_weights[row_idx, hard_indices] = log_w
            logits = logits + log_weights

        labels = torch.arange(batch_size, device=d_pred.device)
        loss_row = F.cross_entropy(logits, labels)
        if self.symmetric_infonce:
            loss_col = F.cross_entropy(logits.T, labels)
            return 0.5 * loss_row + 0.5 * loss_col
        return loss_row

    def _latent_leakage_penalty(
        self, z_geom: torch.Tensor, z_app: torch.Tensor
    ) -> torch.Tensor:
        """Penalize statistical dependence between z_geom and z_app.

        cross_cov: mean squared cross-covariance.
        hsic: normalized linear-HSIC proxy (scale-invariant cross-correlation energy).
        """
        if z_geom.numel() == 0 or z_app.numel() == 0:
            return z_geom.new_zeros(())

        z_geom_c = z_geom - z_geom.mean(dim=0, keepdim=True)
        z_app_c = z_app - z_app.mean(dim=0, keepdim=True)
        denom = float(max(z_geom_c.shape[0] - 1, 1))

        if self.latent_leakage_type == "cross_cov":
            cross_cov = (z_geom_c.T @ z_app_c) / denom
            return cross_cov.pow(2).mean()

        # Normalized linear-HSIC proxy: compute squared cross-correlation.
        z_geom_std = z_geom_c.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        z_app_std = z_app_c.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        z_geom_n = z_geom_c / z_geom_std
        z_app_n = z_app_c / z_app_std
        cross_corr = (z_geom_n.T @ z_app_n) / denom
        return cross_corr.pow(2).mean()

    def _app_repulsion_loss(
        self, z_app_a: torch.Tensor, z_app_b: torch.Tensor
    ) -> torch.Tensor:
        """Push z_app apart for matching cross-modal pairs via cyclic-shifted InfoNCE."""
        batch_size = len(z_app_a)
        z_app_a = F.normalize(z_app_a, p=2, dim=1)
        z_app_b = F.normalize(z_app_b, p=2, dim=1)
        sim = z_app_a @ z_app_b.T / self.temperature
        # Cyclic shift: target is next sample, not self — pushes diagonal apart
        labels = (torch.arange(batch_size, device=z_app_a.device) + 1) % batch_size
        return F.cross_entropy(sim, labels)

    def _supervised_contrastive_loss(
        self, z: torch.Tensor, labels: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """Supervised contrastive loss (Khosla et al.) with label-defined positives."""
        z = F.normalize(z, p=2, dim=1)
        n = z.shape[0]
        device = z.device

        logits = z @ z.T / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        eye = torch.eye(n, dtype=torch.bool, device=device)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & (~eye)

        masked_logits = logits.masked_fill(eye, float('-inf'))
        log_prob = logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)

        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if not valid.any():
            return z.new_zeros(())

        mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_counts.clamp_min(1).float()
        return -mean_log_prob_pos[valid].mean()

    # -- Ridge R² with Gram caching --
    # _ridge_prepare computes the O(n*d^2) Gram matrix and O(d^3) regularized
    # factorization once per latent z.  _ridge_r2_from_prep reuses the cached
    # result for each target y, requiring only O(n*d + d^3) per call instead of
    # the full O(n*d^2 + d^3).  When both probe_c and probe_bc are active, this
    # avoids recomputing the same Gram matrix up to 3x per latent per modality.

    def _ridge_prepare(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float] | None:
        """Preprocess z for ridge R²: center, standardize, build regularized Gram.

        Returns (zc, gram_reg, scale) or None if z is empty.
        """
        if z.numel() == 0:
            return None

        zc = z - z.mean(dim=0, keepdim=True)
        z_std = zc.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        zc = zc / z_std

        n = zc.shape[0]
        d = zc.shape[1]
        scale = float(max(n, 1))
        gram = (zc.T @ zc) / scale
        eye = torch.eye(d, device=zc.device, dtype=zc.dtype)
        gram_reg = gram + self.probe_c_ridge * eye

        return zc, gram_reg, scale

    def _ridge_r2_from_prep(
        self,
        zc: torch.Tensor,
        gram_reg: torch.Tensor,
        scale: float,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute R² using precomputed centered z and regularized Gram matrix."""
        y = y.view(-1).to(dtype=zc.dtype)
        yc = y - y.mean()
        sst = (yc * yc).sum().clamp_min(1e-6)

        rhs = (zc.T @ yc) / scale
        w = torch.linalg.solve(gram_reg, rhs.unsqueeze(1)).squeeze(1)
        y_hat = zc @ w

        sse = ((yc - y_hat) ** 2).sum()
        r2 = 1.0 - sse / sst
        return torch.clamp(r2, min=-1.0, max=1.0)

    def _ridge_r2(
        self, z: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable ridge-regression R^2 from latent z to scalar target y."""
        prep = self._ridge_prepare(z)
        if prep is None:
            return z.new_zeros(())
        return self._ridge_r2_from_prep(*prep, y)

    def _probe_c_predictability_gap(
        self,
        z_geom: torch.Tensor,
        z_app: torch.Tensor,
        gradients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Probe-C-like predictability gap: R^2(z_geom)-R^2(z_app).

        When probe_c_invert=True (G7-2D), swaps z_geom/z_app roles so that
        the loss encourages z_app to predict gradients better than z_geom.
        """
        if self.probe_c_invert:
            z_geom, z_app = z_app, z_geom
        prep_geom = self._ridge_prepare(z_geom)
        prep_app = self._ridge_prepare(z_app)
        if prep_geom is None or prep_app is None:
            zero = (z_geom if z_geom.numel() > 0 else z_app).new_zeros(())
            return zero, zero, zero
        r2_geom = self._ridge_r2_from_prep(*prep_geom, gradients)
        r2_app = self._ridge_r2_from_prep(*prep_app, gradients)
        return r2_geom - r2_app, r2_geom, r2_app

    def _t1_sensitivity_ratio(
        self,
        z_geom_a: torch.Tensor,
        z_geom_b: torch.Tensor,
        z_app_a: torch.Tensor,
        z_app_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute probe-aligned T1 ratio Delta_app_norm / Delta_geom_norm."""
        delta_geom = (z_geom_a - z_geom_b).norm(dim=1).mean()
        delta_app = (z_app_a - z_app_b).norm(dim=1).mean()

        z_geom_pool = torch.cat([z_geom_a, z_geom_b], dim=0)
        z_app_pool = torch.cat([z_app_a, z_app_b], dim=0)
        std_geom = z_geom_pool.std(unbiased=False).clamp_min(1e-8)
        std_app = z_app_pool.std(unbiased=False).clamp_min(1e-8)

        delta_geom_norm = delta_geom / std_geom
        delta_app_norm = delta_app / std_app
        ratio = delta_app_norm / (delta_geom_norm + 1e-8)
        return ratio, delta_geom_norm, delta_app_norm

    def _probe_bc_predictability_gaps(
        self,
        z_geom: torch.Tensor,
        z_app: torch.Tensor,
        intensities: torch.Tensor,
        gradients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Probe-B and Probe-C gaps in a differentiable form.

        Caches Gram matrices for z_geom/z_app and reuses across intensity and
        gradient targets (up to 4 _ridge_r2 calls share 2 Gram preparations).

        Returns:
            gap_b: R^2(z_app -> intensity) - R^2(z_geom -> intensity)
            gap_c: R^2(z_geom -> gradient) - R^2(z_app -> gradient)
            r2_geom_int, r2_app_int, r2_geom_grad, r2_app_grad
        """
        prep_geom = self._ridge_prepare(z_geom)
        prep_app = self._ridge_prepare(z_app)
        if prep_geom is None or prep_app is None:
            zero = (z_geom if z_geom.numel() > 0 else z_app).new_zeros(())
            return zero, zero, zero, zero, zero, zero

        r2_geom_int = self._ridge_r2_from_prep(*prep_geom, intensities)
        r2_app_int = self._ridge_r2_from_prep(*prep_app, intensities)
        gap_b = r2_app_int - r2_geom_int

        r2_geom_grad = self._ridge_r2_from_prep(*prep_geom, gradients)
        r2_app_grad = self._ridge_r2_from_prep(*prep_app, gradients)
        gap_c = r2_geom_grad - r2_app_grad

        return gap_b, gap_c, r2_geom_int, r2_app_int, r2_geom_grad, r2_app_grad
