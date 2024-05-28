import pickle
import numpy as np
from sklearn.decomposition import PCA

from ultra.util import get_root_logger


BASE_PATH = 'descriptions-datasets/'
ADA_EMBEDDING_DIM = 1536

logger = get_root_logger(file=False)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_descriptions_from_pickle():
    p = load_pickle(f'{BASE_PATH}/entity2vec.pickle')
    entity2id = {}
    embeddings = []
    for entity in p:
        entity2id[entity] = len(entity2id)
        embeddings.append(np.array(p[entity]))

    return embeddings, entity2id


def run_pca(np_array, target_dim=64):
    pca = PCA(n_components=target_dim)
    pca.fit(np_array)
    return pca.transform(np_array)
        

def save_vectors(embeddings, entity2id, use_pca):
    logger.info(f'Creating entity2pca vector dict')
    entity2pca = {entity: embeddings[entity2id[entity]] for entity in entity2id}
    
    logger.info(f'Saving vectors to file...')
    with open(f'{BASE_PATH}/entity2{"pca_128_" if use_pca else ""}vec.pickle', 'wb') as f:
        pickle.dump((entity2pca), f)

    logger.info(f'Vectors saved to file.')


if __name__ == '__main__':
    USE_PCA = True
    PCA_DIM = 128
    embeddings, entity2id = load_descriptions_from_pickle()
    embeddings = np.array(embeddings)
    logger.info(f'Loaded embeddings of shape: {embeddings.shape}')

    if USE_PCA:
        logger.info('Running PCA...')
        embeddings = run_pca(embeddings, PCA_DIM)
        logger.info(f'PCA done. Embeddings shape: {embeddings.shape}')

    save_vectors(embeddings, entity2id, USE_PCA)
    logger.info('Done.')

    # cosine similarity test
    # from sklearn.metrics.pairwise import cosine_similarity

