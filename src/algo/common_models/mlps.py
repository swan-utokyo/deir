import torch as th

from torch import Tensor, nn
from torch.nn import functional as F
from typing import Type, Tuple, Optional

from src.utils.enum_types import NormType


class PolicyValueOutputHeads(nn.Module):
    def __init__(
        self,
        inputs_dim: int,
        latents_dim: int = 128,
        activation_fn: Type[nn.Module] = nn.ReLU,
        mlp_norm: NormType = NormType.NoNorm,
        mlp_layers: int = 1,
    ):
        super(PolicyValueOutputHeads, self).__init__()
        self.input_dim = inputs_dim
        self.mlp_layers = mlp_layers

        self.latent_dim_pi = latents_dim
        self.latent_dim_vf = latents_dim

        p_modules = [
            nn.Linear(self.input_dim, self.latent_dim_pi),
            NormType.get_norm_layer_1d(mlp_norm, self.latent_dim_pi),
            activation_fn(),
        ]
        for _ in range(1, mlp_layers):
            p_modules += [
                nn.Linear(self.latent_dim_pi, self.latent_dim_pi),
                NormType.get_norm_layer_1d(mlp_norm, self.latent_dim_pi),
                activation_fn(),
            ]
        self.policy_latent_nn = nn.Sequential(*p_modules)

        v_modules = [
            nn.Linear(self.input_dim, self.latent_dim_vf),
            NormType.get_norm_layer_1d(mlp_norm, self.latent_dim_vf),
            activation_fn(),
        ]
        for _ in range(1, mlp_layers):
            v_modules += [
                nn.Linear(self.latent_dim_vf, self.latent_dim_vf),
                NormType.get_norm_layer_1d(mlp_norm, self.latent_dim_vf),
                activation_fn(),
            ]
        self.value_latent_nn = nn.Sequential(*v_modules)

    def forward(self, features: Tensor):
        return self.policy_latent_nn(features), self.value_latent_nn(features)


class ForwardModelOutputHeads(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 128,
        activation_fn: Type[nn.Module] = nn.ReLU,
        action_num: int = 0,
        mlp_norm: NormType = NormType.NoNorm,
        mlp_layers: int = 1,
    ):
        super(ForwardModelOutputHeads, self).__init__()
        self.action_num = action_num

        modules = [
            nn.Linear(feature_dim + action_num, latent_dim),
            NormType.get_norm_layer_1d(mlp_norm, latent_dim),
            activation_fn(),
        ]
        for _ in range(1, mlp_layers):
            modules += [
                nn.Linear(latent_dim, latent_dim),
                NormType.get_norm_layer_1d(mlp_norm, latent_dim),
                activation_fn(),
            ]
        modules.append(nn.Linear(latent_dim, feature_dim))
        self.nn = nn.Sequential(*modules)

    def forward(self, curr_emb: Tensor, curr_act: Tensor) -> Tensor:
        one_hot_actions = F.one_hot(curr_act, num_classes=self.action_num)
        inputs = th.cat([curr_emb, one_hot_actions], dim=1)
        return self.nn(inputs)


class InverseModelOutputHeads(nn.Module):
    def __init__(
        self,
        features_dim: int,
        latents_dim: int = 128,
        activation_fn: Type[nn.Module] = nn.ReLU,
        action_num: int = 0,
        mlp_norm: NormType = NormType.NoNorm,
        mlp_layers: int = 1,
    ):
        super(InverseModelOutputHeads, self).__init__()

        modules = [
            nn.Linear(features_dim * 2, latents_dim),
            NormType.get_norm_layer_1d(mlp_norm, latents_dim),
            activation_fn(),
        ]
        for _ in range(1, mlp_layers):
            modules += [
                nn.Linear(latents_dim, latents_dim),
                NormType.get_norm_layer_1d(mlp_norm, latents_dim),
                activation_fn(),
            ]
        modules.append(nn.Linear(latents_dim, action_num))
        self.nn = nn.Sequential(*modules)

    def forward(self, curr_emb: Tensor, next_emb: Tensor) -> Tensor:
        inputs = th.cat([curr_emb, next_emb], dim=1)
        return self.nn(inputs)


class ICMOutputHeads(nn.Module):
    def __init__(
        self,
        features_dim: int,
        latents_dim: int = 128,
        activation_fn: Type[nn.Module] = nn.ReLU,
        action_num: int = 0,
        mlp_norm: NormType = NormType.NoNorm,
        mlp_layers: int = 1,
    ):
        super(ICMOutputHeads, self).__init__()
        self.icm_forward_model = ForwardModelOutputHeads(
            features_dim, latents_dim, activation_fn, action_num,
            mlp_norm, mlp_layers
        )
        self.icm_inverse_model = InverseModelOutputHeads(
            features_dim, latents_dim, activation_fn, action_num,
            mlp_norm, mlp_layers
        )

    def forward(self, curr_emb: Tensor, next_emb: Tensor, curr_act: Tensor) -> Tuple[Tensor, Tensor]:
        return self.icm_forward_model(curr_emb, curr_act), \
               self.icm_inverse_model(curr_emb, next_emb)


class RNDOutputHeads(nn.Module):
    def __init__(self,
                 features_dim: int,
                 latents_dim: int = 128,
                 outputs_dim: int = 128,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 mlp_norm: NormType = NormType.NoNorm,
                 mlp_layers: int = 1,
    ):
        super().__init__()

        self.target = nn.Sequential(
            nn.Linear(features_dim, latents_dim),
            NormType.get_norm_layer_1d(mlp_norm, latents_dim),
            activation_fn(),

            nn.Linear(latents_dim, latents_dim),
            NormType.get_norm_layer_1d(mlp_norm, latents_dim),
            activation_fn(),

            nn.Linear(latents_dim, outputs_dim),
            NormType.get_norm_layer_1d(mlp_norm, outputs_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(features_dim, latents_dim),
            NormType.get_norm_layer_1d(mlp_norm, latents_dim),
            activation_fn(),

            nn.Linear(latents_dim, outputs_dim),
            NormType.get_norm_layer_1d(mlp_norm, outputs_dim),
        )

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, emb: Tensor) -> Tuple[Tensor, Tensor]:
        with th.no_grad():
            target_outputs = self.target(emb)
        predicted_outputs = self.predictor(emb)
        return target_outputs, predicted_outputs


class NGUOutputHeads(nn.Module):
    def __init__(
        self,
        features_dim: int,
        latents_dim: int = 128,
        activation_fn: Type[nn.Module] = nn.ReLU,
        action_num: int = 0,
        mlp_norm: NormType = NormType.NoNorm,
        mlp_layers: int = 1,
        use_rnd: int = 0,
    ):
        super(NGUOutputHeads, self).__init__()
        self.use_rnd = use_rnd
        if use_rnd:
            self.ngu_rnd_model = RNDOutputHeads(
                features_dim, latents_dim, latents_dim, activation_fn,
                mlp_norm, mlp_layers
            )
        self.ngu_inverse_model = InverseModelOutputHeads(
            features_dim, latents_dim, activation_fn, action_num,
            mlp_norm, mlp_layers
        )

    def inverse_forward(self, curr_emb: Tensor, next_emb: Tensor) -> Tensor:
        return self.ngu_inverse_model(curr_emb, next_emb)

    def rnd_forward(self, curr_emb: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if self.use_rnd:
            return self.ngu_rnd_model(curr_emb)
        return None, None


class NovelDOutputHeads(nn.Module):
    def __init__(
            self,
            features_dim: int,
            latents_dim: int = 128,
            activation_fn: Type[nn.Module] = nn.ReLU,
            mlp_norm: NormType = NormType.NoNorm,
            mlp_layers: int = 1,
    ):
        super(NovelDOutputHeads, self).__init__()
        self.noveld_rnd_model = RNDOutputHeads(
            features_dim, latents_dim, latents_dim, activation_fn,
            mlp_norm, mlp_layers
        )

    def forward(self, curr_emb: Tensor, next_emb: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        curr_tgt, curr_prd = self.noveld_rnd_model(curr_emb)
        next_tgt, next_prd = self.noveld_rnd_model(next_emb)
        return curr_tgt, curr_prd, next_tgt, next_prd


class DiscriminatorOutputHeads(nn.Module):
    def __init__(
        self,
        inputs_dim: int,
        latents_dim: int = 128,
        activation_fn: Type[nn.Module] = nn.ReLU,
        action_num: int = 0,
        mlp_norm : NormType = NormType.NoNorm,
        mlp_layers : int = 1,
    ):
        super(DiscriminatorOutputHeads, self).__init__()
        self.action_num = action_num

        modules = [
            nn.Linear(inputs_dim * 2 + action_num, latents_dim),
            NormType.get_norm_layer_1d(mlp_norm, latents_dim),
            activation_fn(),
        ]
        for _ in range(1, mlp_layers):
            modules += [
                nn.Linear(latents_dim, latents_dim),
                NormType.get_norm_layer_1d(mlp_norm, latents_dim),
                activation_fn(),
            ]
        modules.append(nn.Linear(latents_dim, 1))
        self.nn = nn.Sequential(*modules)

    def forward(self, curr_emb: Tensor, next_emb: Tensor, curr_act: Tensor) -> Tensor:
        one_hot_act = F.one_hot(curr_act, num_classes=self.action_num)
        inputs = th.cat([curr_emb, next_emb, one_hot_act], dim=1)
        return self.nn(inputs)
