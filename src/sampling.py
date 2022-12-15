import torch
import plotly.express as px
from torch.utils.data import Dataset, DataLoader


def get_ground_truth_features(n, d):
    rnd = torch.randn(n, d)
    samples = rnd / torch.norm(rnd, dim=-1, keepdim=True)
    uniform = torch.rand(n)
    sample_scaling = uniform**(1/d)
    samples = samples * sample_scaling[:, None]
    return samples


def generate_cov(n, seed=0):
    torch.manual_seed(seed)
    cov = torch.rand(n, n)*2 - 1
    cov = cov @ cov.T
    return cov


def get_feature_sample(ground_truth_features, decay=0.99, cov: torch.Tensor = None, rescale_features: int = None):
    '''
    ground_truth_features_toy = get_ground_truth_features(100, 2)
    cov = generate_cov(100)
    feature_samples, active = get_feature_sample(ground_truth_features_toy, decay = 0.99, cov = cov, rescale_features = 10)
    '''

    n = ground_truth_features.shape[0]

    # get probability a feature is on

    # Feature Correlations
    if cov is not None:
        correlated_sample = torch.distributions.MultivariateNormal(
            torch.zeros(n), cov).sample()
        probability_feature_is_on = torch.distributions.Normal(
            0, 1).cdf(correlated_sample)
    else:
        probability_feature_is_on = torch.rand(n)

    # Decay
    if decay:  # decay the probability of activation
        probability_feature_is_on = probability_feature_is_on ** (decay**torch.arange(n))

    # Feature Rescaling
    if rescale_features is not None:  # rescale the probability of activation
        ratio = rescale_features/n
        probability_feature_is_on = probability_feature_is_on * ratio

    active = torch.bernoulli(probability_feature_is_on)

    ground_truth_features = ground_truth_features * \
        active[:, None]  # * sample_scaling[:, None]

    return ground_truth_features, active


class FeatureSampleDataset(Dataset):
    def __init__(self, G, h, decay=0.99, rescale_features: int = 10, size_of_dataset: int = 1000):
        '''
        G: number of features
        h: dimension of features
        decay: decay of feature activation
        rescale_features: rescale the number of features
        '''
        self.G = G
        self.h = h

        self.reset() # create ground truth features and covariance matrix
        # self.ground_truth_features = ground_truth_features
        # self.cov = cov

        self.decay = decay
        self.rescale_features = rescale_features
        self.size_of_dataset = size_of_dataset

    def __len__(self):
        return self.size_of_dataset

    def __getitem__(self, idx):
        return get_feature_sample(
            self.ground_truth_features, 
            decay=self.decay, 
            cov=self.cov, 
            rescale_features=self.rescale_features)[0].sum(dim=0)

    def get_ground_truth_features(self):
        return self.ground_truth_features

    def get_cov(self):
        return self.cov

    def reset(self):
        self.ground_truth_features = get_ground_truth_features(self.G, self.h)
        self.cov = generate_cov(self.ground_truth_features.shape[0])