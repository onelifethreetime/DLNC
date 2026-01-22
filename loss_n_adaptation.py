import torch
import torch as th
import torch.nn as nn

#import config
import kernel

EPSILON = 1E-9
DEBUG_MODE = False


def triu(X):
    # Sum of strictly upper triangular part
    return th.sum(th.triu(X, diagonal=1))


def _atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return th.where(X < eps, X.new_tensor(eps), X)


def d_cs(A, K,n_clusters):
    """
    Cauchy-Schwarz divergence.

    :param A: Cluster assignment matrix
    :type A:  th.Tensor
    :param K: Kernel matrix
    :type K: th.Tensor
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :return: CS-divergence
    :rtype: th.Tensor
    """
    nom = th.t(A) @ K @ A
    dnom_squared = th.unsqueeze(th.diagonal(nom), -1) @ th.unsqueeze(th.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / th.sqrt(dnom_squared))
    return d


# ======================================================================================================================
# Loss terms
# ======================================================================================================================

class LossTerm:
    # Names of tensors required for the loss computation
    required_tensors = []

    def __init__(self, *args, **kwargs):
        """
        Base class for a term in the loss function.

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        pass

    def __call__(self, net):
        raise NotImplementedError()


class Contrastive(LossTerm):
    large_num = 1e9

    def __init__(self, ):
        """
        Contrastive loss function

        :param cfg: Loss function config
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        # Select which implementation to use

        n_clusters=10
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")
        self.negative_samples_ratio=0.25

        if self.negative_samples_ratio == -1:
            self._loss_func = self._loss_without_negative_sampling
        else:
            self.eye = th.eye(n_clusters, device=self.device)
            self._loss_func = self._loss_with_negative_sampling

        # Set similarity function
        self.contrastive_similarity="cos"
        if self.contrastive_similarity == "cos":
            self.similarity_func = self._cosine_similarity
        elif self.contrastive_similarity == "gauss":
            self.similarity_func = kernel.vector_kernel
        else:
            raise RuntimeError(f"Invalid contrastive similarity: {self.contrastive_similarity}")

    @staticmethod
    def _norm(mat):
        return th.nn.functional.normalize(mat, p=2, dim=1)

    @staticmethod
    def get_weight(net):
        w = th.min(th.nn.functional.softmax(net.fusion.weights.detach(), dim=0))
        return w

    @classmethod
    def _normalized_projections(cls, net):
        n = net.projections.size(0) // 2
        h1, h2 = net.projections[:n], net.projections[n:]
        h2 = cls._norm(h2)
        h1 = cls._norm(h1)
        return n, h1, h2

    @classmethod
    def _cosine_similarity(cls, projections):
        h = cls._norm(projections)
        #h=projections
        return h @ h.t()

    def _draw_negative_samples(self, y, pos_indices):
        """
        Construct set of negative samples.

        :param net: Model
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: Loss config
        :type cfg: config.defaults.Loss
        :param v: Number of views
        :type v: int
        :param pos_indices: Row indices of the positive samples in the concatenated similarity matrix
        :type pos_indices: th.Tensor
        :return: Indices of negative samples
        :rtype: th.Tensor
        """
        #cat = net.output.detach().argmax(dim=1)
        cat=y
        v=2
        cat = th.cat(v * [cat], dim=0)

        weights = (1 - self.eye[cat])[:, cat[[pos_indices]]].T
        n_negative_samples = int(self.negative_samples_ratio * cat.size(0))
        negative_sample_indices = th.multinomial(weights, n_negative_samples, replacement=True)
        if DEBUG_MODE:
            self._check_negative_samples_valid(cat, pos_indices, negative_sample_indices)
        return negative_sample_indices

    @staticmethod
    def _check_negative_samples_valid(cat, pos_indices, neg_indices):
        pos_cats = cat[pos_indices].view(-1, 1)
        neg_cats = cat[neg_indices]
        assert (pos_cats != neg_cats).detach().cpu().numpy().all()

    @staticmethod
    def _get_positive_samples(logits, v, n):
        """
        Get positive samples

        :param logits: Input similarities
        :type logits: th.Tensor
        :param v: Number of views
        :type v: int
        :param n: Number of samples per view (batch size)
        :type n: int
        :return: Similarities of positive pairs, and their indices
        :rtype: Tuple[th.Tensor, th.Tensor]
        """
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = th.diagonal(logits, offset=diagonal_offset)
            _lower = th.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = th.arange(0, diag_length)
            _lower_inds = th.arange(i * n, v * n)
            if DEBUG_MODE:
                assert _upper.size() == _lower.size() == _upper_inds.size() == _lower_inds.size() == (diag_length,)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = th.cat(diagonals, dim=0)
        pos_inds = th.cat(inds, dim=0)
        return pos, pos_inds

    def _loss_with_negative_sampling(self,net, cat_feature,y):
        """
        Contrastive loss implementation with negative sampling.

        :param net: Model
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: Loss config
        :type cfg: config.defaults.Loss
        :param extra:
        :type extra:
        :return: Loss value
        :rtype: th.Tensor
        """
        self.tau=0.5
        n=cat_feature.size(0)//2
        #n = net.projections.size(0)//2
        #v = len(net.backbone_outputs)
        v = 2
        logits = self.similarity_func(cat_feature) / self.tau

        pos, pos_inds = self._get_positive_samples(logits, v, n)
        neg_inds = self._draw_negative_samples( y, pos_inds)
        neg = logits[pos_inds.view(-1, 1), neg_inds]

        inputs = th.cat((pos.view(-1, 1), neg), dim=1)
        labels = th.zeros(v * (v - 1) * n, device=self.device, dtype=th.long)
        loss = th.nn.functional.cross_entropy(inputs, labels)
        self.adaptive_contrastive_weight=False
        if self.adaptive_contrastive_weight:
            loss *= self.get_weight(net)
        self.delta=1
        return self.delta * loss

    def _loss_without_negative_sampling(self, net, extra):
        """
        Contrastive loss implementation without negative sampling.
        Adapted from: https://github.com/google-research/simclr/blob/master/objective.py

        :param net: Model
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: Loss config
        :type cfg: config.defaults.Loss
        :param extra:
        :type extra:
        :return:
        :rtype:
        """
        assert len(net.backbone_outputs) == 2, "Contrastive loss without negative sampling only supports 2 views."
        n, h1, h2 = self._normalized_projections(net)

        labels = th.arange(0, n, device=self.device, dtype=th.long)
        masks = th.eye(n, device=self.device)

        logits_aa = ((h1 @ h1.t()) / self.tau) - masks * self.large_num
        logits_bb = ((h2 @ h2.t()) / self.tau) - masks * self.large_num

        logits_ab = (h1 @ h2.t()) / self.tau
        logits_ba = (h2 @ h1.t()) / self.tau

        loss_a = th.nn.functional.cross_entropy(th.cat((logits_ab, logits_aa), dim=1), labels)
        loss_b = th.nn.functional.cross_entropy(th.cat((logits_ba, logits_bb), dim=1), labels)

        loss = (loss_a + loss_b)

        self.adaptive_contrastive_weight=False
        self.delta=0.1
        if self.adaptive_contrastive_weight:
            loss *= self.get_weight(net)

        return self.delta * loss

    def __call__(self,net, cat_feature,y):
        return self._loss_func(net,cat_feature,y)

#contrastive=Contrastive()
#loss=contrastive(net,y)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        print('....................')
        print(features.shape)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        print('....................')
        print(mask.shape)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
