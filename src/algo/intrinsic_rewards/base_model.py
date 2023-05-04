import gym
import numpy as np

from torch.nn import GRUCell
from typing import Dict, Any, List

from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor

from src.algo.common_models.gru_cell import CustomGRUCell
from src.algo.common_models.mlps import *
from src.utils.enum_types import NormType
from src.utils.common_func import init_module_with_name


class IntrinsicRewardBaseModel(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 0.5,
        model_learning_rate: float = 3e-4,
        model_cnn_features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_cnn_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        model_features_dim: int = 256,
        model_latents_dim: int = 256,
        model_mlp_norm: NormType = NormType.BatchNorm,
        model_cnn_norm: NormType = NormType.BatchNorm,
        model_gru_norm: NormType = NormType.NoNorm,
        use_model_rnn: int = 0,
        model_mlp_layers: int = 1,
        gru_layers: int = 1,
        use_status_predictor: int = 0,
    ):
        super().__init__()
        if isinstance(observation_space, gym.spaces.Dict):
            observation_space = observation_space["rgb"]
        self.observation_space = observation_space
        self.normalize_images = normalize_images
        self.action_space = action_space
        self.action_num = action_space.n
        self.max_grad_norm = max_grad_norm

        self.model_features_dim = model_features_dim
        self.model_latents_dim = model_latents_dim
        self.model_learning_rate = model_learning_rate
        self.model_mlp_norm = model_mlp_norm
        self.model_cnn_norm = model_cnn_norm
        self.model_gru_norm = model_gru_norm
        self.model_mlp_layers = model_mlp_layers
        self.gru_layers = gru_layers
        self.use_status_predictor = use_status_predictor
        self.model_gru_cell = GRUCell if self.model_gru_norm == NormType.NoNorm else CustomGRUCell
        self.use_model_rnn = use_model_rnn
        self.model_cnn_features_extractor_class = model_cnn_features_extractor_class
        self.model_cnn_features_extractor_kwargs = model_cnn_features_extractor_kwargs
        self.activation_fn = activation_fn
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.model_rnn_kwargs = dict(
            input_size=self.model_features_dim,
            hidden_size=self.model_features_dim,
        )
        if self.model_gru_norm != NormType.NoNorm:
            self.model_rnn_kwargs.update(dict(
                norm_type=self.model_gru_norm,
            ))

        self.constant_zero = th.zeros(1, dtype=th.float)
        self.constant_one = th.ones(1, dtype=th.float)


    def _build(self) -> None:
        self.model_cnn_features_extractor_kwargs.update(dict(
            features_dim=self.model_features_dim,
        ))
        self.model_cnn_extractor = \
            self.model_cnn_features_extractor_class(
                self.observation_space,
                **self.model_cnn_features_extractor_kwargs
            )

        # Build RNNs
        self.model_rnns = []
        if self.use_model_rnn:
            for l in range(self.gru_layers):
                name = f'model_rnn_layer_{l}'
                setattr(self, name, self.model_gru_cell(**self.model_rnn_kwargs))
                self.model_rnns.append(getattr(self, name))

        # Build Key/Door status predictors
        if self.use_status_predictor:
            self.model_key_door_status_predictor = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.model_features_dim, self.model_features_dim),
                NormType.get_norm_layer_1d(self.model_mlp_norm, self.model_features_dim),
                self.activation_fn(),
                nn.Dropout(p=0.1),
                nn.Linear(self.model_features_dim, 2),
            )
            self.model_agent_position_predictor = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.model_features_dim, self.model_features_dim),
                NormType.get_norm_layer_1d(self.model_mlp_norm, self.model_features_dim),
                self.activation_fn(),
                nn.Dropout(p=0.1),
                nn.Linear(self.model_features_dim, 3),
            )
        else:
            self.model_key_door_status_predictor = nn.Identity()
            self.model_agent_position_predictor = nn.Identity()


    def _init_modules(self) -> None:
        assert hasattr(self, 'model_mlp'), "Be sure to define the model's MLP first"

        module_names = {
            self.model_cnn_extractor: 'model_cnn_extractor',
            self.model_mlp: 'model_mlp',
        }
        if self.use_model_rnn:
            for l in range(self.gru_layers):
                name = f'model_rnn_layer_{l}'
                module = getattr(self, name)
                module_names.update({module: name})
        if self.use_status_predictor:
            module_names.update({
                self.model_key_door_status_predictor: 'model_key_door_status_predictor',
                self.model_agent_position_predictor: 'model_agent_position_predictor',
            })
        for module, name in module_names.items():
            init_module_with_name(name, module)


    def _init_optimizers(self) -> None:
        param_dicts = dict(self.named_parameters(recurse=True)).items()
        self.model_params = [
            param for name, param in param_dicts
            if (name.find('status_predictor') < 0 and name.find('position_predictor') < 0)
        ]
        self.model_optimizer = self.optimizer_class(self.model_params, lr=self.model_learning_rate, **self.optimizer_kwargs)

        if self.use_status_predictor:
            param_dicts = dict(self.named_parameters(recurse=True)).items()
            self.status_predictor_params = [
                param for name, param in param_dicts
                    if name.find('status_predictor') >= 0 or name.find('position_predictor') >= 0
            ]
            self.predictor_optimizer = \
                self.optimizer_class(self.status_predictor_params, lr=self.model_learning_rate, **self.optimizer_kwargs)


    def _get_rnn_embeddings(self, hiddens: Optional[Tensor], inputs: Tensor, modules: List[nn.Module]):
        outputs = []
        for i, module in enumerate(modules):
            hidden_i = th.squeeze(hiddens[:, i, :])
            output_i = module(inputs, hidden_i)
            inputs = output_i
            outputs.append(output_i)
        outputs = th.stack(outputs, dim=1)
        return outputs


    def _get_cnn_embeddings(self, obs, module=None):
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        if module is None:
            return self.model_cnn_extractor(obs)
        return module(obs)


    def _get_rnd_embeddings(self, obs, mems):
        if self.rnd_use_policy_emb:
            with th.no_grad():
                cnn_embs = self._get_cnn_embeddings(obs, module=self.policy_cnn)
                if self.use_model_rnn:
                    gru_mems = self._get_rnn_embeddings(mems, cnn_embs, self.policy_rnns)
                    rnn_embs = th.squeeze(gru_mems[:, -1, :])
        else:
            cnn_embs = self._get_cnn_embeddings(obs)
            if self.use_model_rnn:
                gru_mems = self._get_rnn_embeddings(mems, cnn_embs, self.model_rnns)
                rnn_embs = th.squeeze(gru_mems[:, -1, :])

        if self.use_model_rnn:
            return rnn_embs, gru_mems
        return cnn_embs, None


    # Key, Door, Agent pos predictor
    def _get_status_prediction_losses(self, embs, key_status, door_status, target_dists):
        if self.use_status_predictor:
            if key_status.shape[0] < embs.shape[0]:
                embs = embs[:key_status.shape[0]]
            # Key / Door status prediction
            pred_status = self.model_key_door_status_predictor(embs.detach()).view(-1, 2)
            pred_key_status = pred_status[:, 0]
            pred_door_status = pred_status[:, 1]
            pred_target_dists = self.model_agent_position_predictor(embs.detach()).view(-1, 3)
            key_loss = F.binary_cross_entropy_with_logits(pred_key_status, key_status.float())
            door_loss = F.binary_cross_entropy_with_logits(pred_door_status, door_status.float())
            # Target distances prediction
            pos_losses = F.mse_loss(pred_target_dists, target_dists.float().view(-1, 3), reduction='none')
            key_dist = pos_losses[:, 0].mean()
            door_dist = pos_losses[:, 1].mean()
            goal_dist = pos_losses[:, 2].mean()
            pos_loss = pos_losses.mean()
        else:
            key_loss = self.constant_zero
            door_loss = self.constant_zero
            pos_loss = self.constant_zero
            key_dist = self.constant_zero
            door_dist = self.constant_zero
            goal_dist = self.constant_zero
        return key_loss, door_loss, pos_loss, \
               key_dist, door_dist, goal_dist

    @staticmethod
    @th.jit.script
    def calc_euclidean_dists(x : Tensor, y : Tensor):
        """
        Calculate the Euclidean distances between two batches of embeddings.
        Input shape: [n, d]
        Return: ((x - y) ** 2).sum(dim=-1) ** 0.5
        """
        features_dim = x.shape[-1]
        x = x.view(1, -1, features_dim)
        y = y.view(1, -1, features_dim)
        return th.cdist(x, y, p = 2.0)[0]