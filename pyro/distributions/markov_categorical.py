import torch
from torch._six import nan
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs, lazy_property
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.ops.indexing import Vindex
from pyro.distributions.util import broadcast_shape


class MarkovCategorical(TorchDistribution):
    r"""
    MarkovCategorical
    """
    arg_constraints = {"initial_logits": constraints.real_vector,
                       "transition_logits": constraints.real_vector}
    has_enumerate_support = True

    def __init__(self, initial_logits, transition_logits, validate_args=None):
        if initial_logits.dim() < 1:
            raise ValueError("expected initial_logits to have at least one dim, "
                             "actual shape = {}".format(initial_logits.shape))
        if transition_logits.dim() < 2:
            raise ValueError("expected transition_logits to have at least two dims, "
                             "actual shape = {}".format(transition_logits.shape))
        batch_shape = broadcast_shape(initial_logits.shape[:-1],
                                transition_logits.shape[:-3])
        self._duration = transition_logits.shape[-3] + 1
        event_shape = torch.Size((self._duration,))
        trans_shape = broadcast_shape(initial_logits.shape[-1:],
                                transition_logits.shape[-2:])
        self.initial_logits = initial_logits - initial_logits.logsumexp(-1, True)
        self.transition_logits = transition_logits - transition_logits.logsumexp(-1, True)
        self.logits = torch.zeros(batch_shape + event_shape + trans_shape)
        self.logits[..., 0, 0, :] = self.initial_logits
        self.logits[..., 1:, :, :] = self.transition_logits 

        self._num_events = trans_shape[-1]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)


    @constraints.dependent_property
    def support(self):
        return constraints.integer_interval(0, self._num_events - 1)


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long()
        # normal interpretation
        if value.shape[-1] == self._duration:
            value_prev = torch.cat(
                (torch.zeros(value.shape[:-1] + (1,), dtype=torch.long), value[..., :-1]),
                dim=-1)
            result = Vindex(self.logits)[..., value_prev, value].sum(-1)
        # enumarated interpretation
        elif value.shape[-1] == 1:
            time = torch.arange(self._duration).view((-1,) + (1,) * (len(value)+2))
            result = Vindex(self.logits)[..., time, value.unsqueeze(-1), value]
        return result


    def enumerate_support(self, expand=True):
        num_events = self._num_events
        values = torch.arange(num_events, dtype=torch.long, device=self.logits.device)
        values = values.view((-1,) + (1,) * (1+len(self._batch_shape)))
        values = values.repeat((1,) + (self._duration,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values
