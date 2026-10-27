
import numpy as np
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree, to_undirected
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import NMF
np.set_printoptions(threshold=np.inf)
from training.tools import *
import math

import torch
import torch.nn as nn
import math



class MarketDrivenSpectralAttention(nn.Module):
    """
    Market-Driven Spectral Attention Module

    This module transforms time-series signals into the frequency domain using FFT,
    applies a complex-valued attention mechanism to normalize stock spectra with
    market trends, and generates a market-aware spectral representation.
    """
    def __init__(self, T: int, N: int, C: int, embed_dim: int, bias=True):
        super().__init__()
        self.T = int(T)  # Time steps
        self.N = int(N)  # Number of stocks
        self.C = int(C)  # Number of feature channels
        self.L = self.T // 2 + 1  # Effective frequency length
        self.embed_dim = embed_dim  # Embedding dimension for intermediate projection

        # Complex-valued MLPs for query, key, and value projections
        self.phi_q = ComplexLinear(self.C * self.L, self.L, bias=bias)  # Output dim matches L
        self.phi_k = ComplexLinear(self.C * self.L, self.L, bias=bias)  # Output dim matches L
        self.phi_v = ComplexLinear(self.C * self.L, self.L, bias=bias)  # Output dim matches L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, N, C] real-valued time series of stocks.

        Returns:
            h_n: [N, L] calibrated complex-valued spectral representation.
        """
        if x.dim() != 3:
            raise ValueError(f"x must be [T, N, C], got {tuple(x.shape)}")
        T_in, N_in, C_in = x.shape
        if T_in != self.T or N_in != self.N or C_in != self.C:
            raise ValueError(f"Input shape {tuple(x.shape)} != [{self.T}, {self.N}, {self.C}]")

        # Align dtype/device
        dtype = self.phi_q.Linear_real.weight.dtype
        device = self.phi_q.Linear_real.weight.device
        x = x.to(device=device, dtype=dtype)

        # Step 1: Aggregate market trend signal
        x_market = x.mean(dim=1, keepdim=True)  # [T, 1, C]

        # Step 2: Transform to frequency domain via FFT
        # [T, N, C] -> [N, C, T] for FFT along time dimension
        x_nct = x.permute(1, 2, 0).contiguous()  # [N, C, T]
        h_nc = torch.fft.fft(x_nct, dim=-1)  # [N, C, T] -> [N, C, T] complex
        h_nc = h_nc[:, :, :self.L]  # Take non-redundant portion [N, C, L]

        x_market_nct = x_market.permute(1, 2, 0).contiguous()  # [1, C, T]
        h_market_nc = torch.fft.fft(x_market_nct, dim=-1)  # [1, C, T] -> [1, C, T] complex
        h_market_nc = h_market_nc[:, :, :self.L]  # [1, C, L]

        # Step 3: Flatten and concatenate channel spectra
        h_n = h_nc.reshape(N_in, -1)  # [N, C*L]
        h_market = h_market_nc.reshape(1, -1)  # [1, C*L]

        # Step 4: Generate query, key, and value vectors
        q_n = self.phi_q(h_n)  # [N, L]
        k_market = self.phi_k(h_market).expand(N_in, -1)  # [N, L]
        v_market = self.phi_v(h_market).expand(N_in, -1)  # [N, L]

        # Step 5: Compute complex-valued similarity score
        s_n = torch.sum(q_n * torch.conj(k_market), dim=-1)  # [N]
        alpha_n = torch.sigmoid(torch.real(s_n))  # [N], attention gate in (0, 1)

        # Step 6: Reconstruct calibrated spectral representation
        h_n = (1 - alpha_n.unsqueeze(-1)) * q_n + alpha_n.unsqueeze(-1) * v_market  # [N, L]

        return h_n

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.Linear_real = torch.nn.Linear(in_features, out_features, bias=bias)
        self.Linear_img = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        real_real = self.Linear_real(input.real)
        img_real = self.Linear_img(input.real)
        real_img = self.Linear_real(input.imag)
        img_img = self.Linear_img(input.imag)
        return real_real - img_img + 1j * (real_img + img_real)




class AdaptiveFrequencyRelationInference(nn.Module):
    """
    Adaptive Frequency-domain Relation Inference Module

    This module decouples aligned spectra into multiple frequency bands,
    infers inter-stock relationships using a tensor-based global attention mechanism,
    and generates sparsified relation matrices and time-domain sequences.
    """
    def __init__(self, T: int, N: int, C: int, K: int, embed_dim: int, lambda_sparsity: float = 0.1, bias=True):
        super().__init__()
        self.T = int(T)  # Time steps
        self.N = int(N)  # Number of stocks
        self.C = int(C)  # Number of feature channels
        self.L = self.T // 2 + 1  # Effective frequency length
        self.K = int(K)  # Number of frequency bands
        self.embed_dim = int(embed_dim)  # Embedding dimension, used as F
        self.lambda_sparsity = lambda_sparsity

        # Learnable parameter tensor W ∈ ℝ^(K × L × embed_dim), where L=31, embed_dim is feature dim
        self.W = nn.Parameter(torch.randn(self.K, self.L, self.embed_dim))
        self.alpha = nn.Parameter(torch.randn(self.embed_dim))  # Learnable vector, shape [embed_dim]

        # Binary masks for frequency bands (first K-1 bands of length L//K, last band for remainder)
        band_length = self.L // self.K
        remainder = self.L - (self.K - 1) * band_length
        self.register_buffer('masks', torch.zeros(self.K, self.L))  # [K, L]
        for k in range(self.K - 1):
            self.masks[k, k * band_length:(k + 1) * band_length] = 1
        self.masks[self.K - 1, (self.K - 1) * band_length:] = 1

    def forward(self, h_tilde_n: torch.Tensor) -> tuple:
        """
        Args:
            h_tilde_n: [N, L] calibrated complex-valued spectral representation.

        Returns:
            R_tilde: [K, N, N] sparsified relation tensor.
            x_n_k: [K, N, T] time-domain sequences for each frequency band.
        """
        if h_tilde_n.dim() != 2 or h_tilde_n.shape[0] != self.N or h_tilde_n.shape[1] != self.L:
            raise ValueError(f"h_tilde_n must be [N, L], got {tuple(h_tilde_n.shape)}")

        # Move masks to the same device as input
        masks = self.masks.to(h_tilde_n.device)

        # Step 1: Decouple into frequency bands
        h_n_k = h_tilde_n.unsqueeze(1).expand(-1, self.K, -1) * masks  # [N, K, L]

        # Step 2: Reorganize to [K, N, L] for further processing
        h_n_k = h_n_k.transpose(0, 1)  # [K, N, L]

        # Step 2: Decompose into magnitude and phase
        A_n_k = torch.abs(h_n_k)  # [K, N, L]
        phi_n_k = torch.angle(h_n_k)  # [K, N, L]

        # Step 3: Compute relation scores
        R_n_m_k = torch.zeros(self.K, self.N, self.N, device=h_tilde_n.device)
        for k in range(self.K):
            A_n = A_n_k[k]  # [N, L]
            phi_n = phi_n_k[k]  # [N, L]
            A_n_m = A_n.unsqueeze(1).expand(-1, self.N, -1)  # [N, N, L]
            A_m_n = A_n.unsqueeze(0).expand(self.N, -1, -1)  # [N, N, L]
            phi_n_m = phi_n.unsqueeze(1).expand(-1, self.N, -1)  # [N, N, L]
            phi_m_n = phi_n.unsqueeze(0).expand(self.N, -1, -1)  # [N, N, L]
            resonance = A_n_m * A_m_n  # [N, N, L]
            phase_diff = torch.cos(phi_n_m - phi_m_n)  # [N, N, L]
            relation = resonance * phase_diff  # [N, N, L]

            # Apply W[k] to transform [N, N, L] to [N, N, embed_dim]
            W_k = self.W[k].to(h_tilde_n.device)  # [L, embed_dim]
            intermediate = torch.einsum('ijk,kl->ijl', relation, W_k)  # [N, N, L] x [L, embed_dim] -> [N, N, embed_dim]

            # Apply alpha to reduce [N, N, embed_dim] to [N, N]
            alpha_k = self.alpha.to(h_tilde_n.device)  # [embed_dim]
            score = torch.einsum('ijl,l->ij', intermediate, alpha_k)  # [N, N, embed_dim] x [embed_dim] -> [N, N]

            R_n_m_k[k] = torch.nn.functional.leaky_relu(score)  # [N, N]

        # Step 4: Normalize relation weights
        R_n_m_k = torch.softmax(R_n_m_k, dim=-1)  # [K, N, N]

        # Step 5: Dynamic sparsification
        tau_k = self.lambda_sparsity * torch.mean(torch.sum(R_n_m_k, dim=(-2, -1)))  # Scalar
        R_tilde = torch.where(torch.abs(R_n_m_k) > tau_k, R_n_m_k, torch.zeros_like(R_n_m_k))  # [K, N, N]

        # Step 6: Inverse FFT to time domain
        h_n_k_padded = torch.zeros(self.K, self.N, self.T, dtype=torch.cfloat, device=h_tilde_n.device)
        h_tilde_n_expanded = h_tilde_n.unsqueeze(0).expand(self.K, -1, -1)  # [K, N, L]
        h_n_k_padded[:, :, :self.L] = h_n_k  # Ensure correct assignment
        x_n_k = torch.fft.ifft(h_n_k_padded, dim=-1).real  # [K, N, T]

        return R_tilde, x_n_k




# Inner Module: Intra-Frequency Sub-band Risk Embedding
class InnerModule(nn.Module):
    def __init__(self, T: int, N: int, F: int, S: int, kernel_sizes: list):
        """
        Args:
            T: Time steps
            N: Number of stocks
            F: Embedding dimension
            S: Number of convolutional scales
            kernel_sizes: List of kernel lengths for each scale
        """
        super().__init__()
        self.N = N
        self.F = F
        self.S = S
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=ks, padding=ks//2)
            for ks in kernel_sizes
        ])
        # W_fuse to match concatenated output dimension
        output_dim = T * S  # Each conv output matches T due to padding
        self.W_fuse = nn.Parameter(torch.randn(F, output_dim))
        self.W_agg = nn.Parameter(torch.randn(F, F))

    def forward(self, x_n_k: torch.Tensor, R_k: torch.Tensor):
        """
        Args:
            x_n_k: [N, T] Time-domain sequence for current frequency band
            R_k: [N, N] Adjacency matrix for current frequency band

        Returns:
            x_n_prime_k: [N, F] Initial representation
            x_tilde_n_k: [N, F] Risk spillover representation
        """
        N, T = x_n_k.shape
        # Multi-scale 1D convolution with padding to preserve length
        conv_outputs = []
        x_n_k = x_n_k.unsqueeze(1)  # [N, 1, T]
        for conv in self.convs:
            x_conv = conv(x_n_k)  # [N, 1, T] -> [N, 1, T] due to padding
            conv_outputs.append(x_conv.squeeze(1))  # [N, T]

        # Concatenate along feature dimension
        x_concat = torch.cat(conv_outputs, dim=-1)  # [N, T * S]
        # Project to embedding space
        x_n_prime_k = torch.matmul(x_concat, self.W_fuse.T)  # [N, T * S] x [T * S, F] -> [N, F]

        # Graph attention using matrix multiplication
        # Transform features with W_agg
        transformed_features = torch.matmul(x_n_prime_k, self.W_agg)  # [N, F]
        # Aggregate using adjacency matrix
        agg = torch.matmul(R_k, transformed_features)  # [N, N] x [N, F] -> [N, F]
        x_tilde_n_k = F.leaky_relu(agg)  # [N, F]

        return x_n_prime_k, x_tilde_n_k

# Outer Module: Inter-Frequency Sub-band State Transition
class OuterModule(nn.Module):
    def __init__(self, F: int):
        """
        Args:
            F: Embedding dimension
        """
        super().__init__()
        self.F = F
        self.W_r = nn.Parameter(torch.randn(2*F, F))
        self.U_r = nn.Parameter(torch.randn(F, F))
        self.b_r = nn.Parameter(torch.zeros(F))
        self.W_z = nn.Parameter(torch.randn(2*F, F))
        self.U_z = nn.Parameter(torch.randn(F, F))
        self.b_z = nn.Parameter(torch.zeros(F))
        self.W_u = nn.Parameter(torch.randn(2*F, F))
        self.U_u = nn.Parameter(torch.randn(F, F))
        self.b_u = nn.Parameter(torch.zeros(F))

    def forward(self, x_n_prime_k: torch.Tensor, x_tilde_n_k: torch.Tensor, z_n_prev: torch.Tensor):
        """
        Args:
            x_n_prime_k: [N, F] Temporal features
            x_tilde_n_k: [N, F] Spatial risk features
            z_n_prev: [N, F] Previous state

        Returns:
            z_n_k: [N, F] Updated state
        """
        N = x_n_prime_k.shape[0]
        # Concatenate temporal and spatial features
        combined = torch.cat([x_n_prime_k, x_tilde_n_k], dim=-1)  # [N, 2F]

        # Reset gate
        r_n_k = torch.sigmoid(torch.matmul(combined, self.W_r) + torch.matmul(z_n_prev, self.U_r) + self.b_r)  # [N, F]

        # Candidate state
        z_tilde_n_k = torch.tanh(torch.matmul(combined, self.W_z) + torch.matmul(r_n_k * z_n_prev, self.U_z) + self.b_z)  # [N, F]

        # Update gate
        u_n_k = torch.sigmoid(torch.matmul(combined, self.W_u) + torch.matmul(z_n_prev, self.U_u) + self.b_u)  # [N, F]

        # Final state
        z_n_k = (1 - u_n_k) * z_n_prev + u_n_k * z_tilde_n_k  # [N, F]
        return z_n_k

# Main Module: Progressive Risk Propagation
class ProgressiveRiskPropagation(nn.Module):
    def __init__(self, T: int, N: int, K: int, F: int, S: int, kernel_sizes: list):
        """
        Args:
            T: Time steps
            N: Number of stocks
            K: Number of frequency bands
            F: Embedding dimension
            S: Number of convolutional scales
            kernel_sizes: List of kernel lengths for each scale
        """
        super().__init__()
        self.K = K
        self.inner = InnerModule(T, N, F, S, kernel_sizes)
        self.outer = OuterModule(F)
        self.initial_state = torch.zeros(N, F)  # Initial state for k=0

    def forward(self, x_n_k: torch.Tensor, R_tilde: torch.Tensor):
        """
        Args:
            x_n_k: [K, N, T] Time-domain sequences for all frequency bands
            R_tilde: [K, N, N] Sparsified relation tensor

        Returns:
            z_n_K: [N, F] Final state after progressive propagation
        """
        N, T = x_n_k.shape[1], x_n_k.shape[2]
        z_n_prev = self.initial_state.to(x_n_k.device)

        # Progressive propagation across frequency bands
        for k in range(self.K):
            # Extract current band data
            x_n_k_curr = x_n_k[k]  # [N, T]
            R_k = R_tilde[k]  # [N, N]

            # Intra-frequency embedding
            x_n_prime_k, x_tilde_n_k = self.inner(x_n_k_curr, R_k)

            # Inter-frequency state transition
            z_n_k = self.outer(x_n_prime_k, x_tilde_n_k, z_n_prev)
            z_n_prev = z_n_k

        return z_n_k  # Final state after K iterations