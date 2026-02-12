from typing import Tuple

import torch
from torch import nn, Tensor


class SPCLLoss(nn.Module):

    def __init__(
        self,
        m: float = 0.25,
        gamma: float = 256,
        neutral_weight: float = 0.05,
    ) -> None:
        super(SPCLLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.neutral_weight = neutral_weight
        self.soft_plus = nn.Softplus()

    def forward(
        self,
        sp: Tensor,
        sn: Tensor,
        s_neu: Tensor,
        s_orig: Tensor,
    ) -> Tensor:

        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        circle_loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)
        )

        target_upper = torch.min(s_orig + self.m, sp)
        target_lower = torch.max(s_orig - self.m, sn)


        valid_mask = (target_upper > target_lower).float()


        loss_p_neu = self._circle_loss_pair(target_upper, s_neu)
        loss_neu_n = self._circle_loss_pair(s_neu, target_lower)


        loss_ranking = (
            (loss_p_neu + loss_neu_n) * 0.5 * valid_mask
        ).sum() / (valid_mask.sum() + 1e-8)


        return circle_loss + self.neutral_weight * loss_ranking

    def _circle_loss_pair(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:

        ap = torch.clamp_min(-pos_scores.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(neg_scores.detach() + self.m, min=0.0)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (pos_scores - delta_p) * self.gamma
        logit_n = an * (neg_scores - delta_n) * self.gamma

        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)
        )
        return loss


