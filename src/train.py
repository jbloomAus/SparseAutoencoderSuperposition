import torch
from tqdm.notebook import tqdm

def train(dataloader, model, dictionary_model, G, h, batch_size, decay = 0.99, lr = 1e-3, alpha = 0.1, rescale_factor = 10):
    '''
    feature_space: samples from unit ball
    cov: covariance matrix,
    rescale_factor: rescale factor (10 seems to approximately get 5 features active on average with decay = 0.99, G = 512)
    model: AutoEncoderModel
    dictionary_model: DictionaryModel
    G: True number of features
    h: Actual number of features
    batch_size: number of samples in a batch
    lr: learning rate
    decay: decay parameter
    alpha: regularization weight parameter
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    l_reconstruction_cache = []
    l_regularization_cache = []
    Loss_cache = []
    model_checkpoints = []
    dictionary_checkpoints = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch, samples in pbar:
        c = model.forward(samples)
        Dc = dictionary_model.forward(c)
        l_reconstruction = torch.norm(samples-Dc, dim=1)
        l_regularization = torch.norm(c, dim=1)
        L = l_reconstruction + alpha*l_regularization
        Loss = L.mean()
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        l_reconstruction_cache.append(l_reconstruction.mean().item())
        l_regularization_cache.append(l_regularization.mean().item())
        Loss_cache.append(Loss.item())
        model_checkpoints.append(model.state_dict())
        dictionary_checkpoints.append(dictionary_model.state_dict())

        # if epoch % 10 == 0:
        #     print(f"epoch {epoch} loss {Loss.item()}")
        pbar.set_description(f"Batch {batch} loss {Loss.item()}")

    return model, dictionary_model, l_reconstruction_cache, l_regularization_cache, Loss_cache, model_checkpoints, dictionary_checkpoints

