from src.paths import ROOT
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform


def get_distance_vec(embedding_matrix, metric='cosine'):
    """
        Takes as input an N x M embedding matrix and outputs the upper triangle of 
        its distance matrix, where distance is computed as 1 - cosine_similarity.
    """
    dist_mat = squareform(pdist(embedding_matrix, metric=metric))
    return dist_mat[np.triu_indices(n = dist_mat.shape[0], m=dist_mat.shape[0], k=1)]

def get_fmri_rdm_study1_aggregated(roi):
    
    ''' 
        Returns the average of the subject RDM matrices for the specified ROI.
    '''

    if roi.isnumeric():

        base_path = f"data/Study1_neural_vectors_RDMs/DK_roi{roi}_mask/voxel_spearman_mat_CREA300_DK_roi{roi}_{{subject}}.txt"
    if roi=='semantic':
        base_path = f"data/Study1_neural_vectors_RDMs/ALE_semantic_network/voxel_spearman_mat_CREA300_ALE.99_S{{subject}}.txt"
    subjects = [str(i) for i in range(1,9)]
    avg_mat = np.empty((8, 300, 300))

    for sub, i in zip(subjects, range(8)):
        sub_mat = np.loadtxt(ROOT / base_path.format(subject=sub))
        avg_mat[i] = sub_mat
    
    avg_dist_mat = 1 - avg_mat.mean(axis=0)

    return avg_dist_mat[np.triu_indices(n = avg_dist_mat.shape[0], m=avg_dist_mat.shape[0], k=1)]        


def get_fmri_rdm_study1_per_part(roi):
    
    ''' 
        Returns the average of the subject RDM matrices for the specified ROI.
    '''

    if roi.isnumeric():

        base_path = f"data/Study1_neural_vectors_RDMs/DK_roi{roi}_mask/voxel_spearman_mat_CREA300_DK_roi{roi}_{{subject}}.txt"
    if roi=='semantic':
        base_path = f"data/Study1_neural_vectors_RDMs/ALE_semantic_network/voxel_spearman_mat_CREA300_ALE.99_S{{subject}}.txt"
    subjects = [str(i) for i in range(1,9)]
    avg_mat = np.empty((8, 300, 300))

    for sub, i in zip(subjects, range(8)):
        sub_mat = np.loadtxt(ROOT / base_path.format(subject=sub))
        avg_mat[i] = sub_mat
    
    ret_mat = 1 - np.stack([avg_mat[i][np.triu_indices(n = avg_mat.shape[1], m=avg_mat.shape[1], k=1)]  for i in range(avg_mat.shape[0])]) 
    
    return ret_mat       


def get_fmri_rdm_study2_aggregated(roi):
    
    ''' 
        Returns the average of the subject RDM matrices for the specified ROI.
    '''

    if roi.isnumeric():
        base_path = f"data/Study2_neural_RSMs_A/DK_roi{roi}_mask/voxel_spearman_mat_SOE320_DK_roi{roi}_{{subject}}.txt"
    if roi=='semantic':
        base_path = f"data/Study2_neural_RSMs_A/ALE_semantic_network/voxel_spearman_mat_SOE320_ALE.99_{{subject}}.txt"
    subjects = ['117', '110', '126', '137', '109', '121', '118', '142', '111', '106', '134', '140', '114', '104', 
                '121', '115', '120', '104', '128', '142', '128', '139', '124', '117', '141', '129', '131', '123', 
                '123', '125', '133', '126', '105', '130', '108', '131', '130', '110', '116', '133', '139', '112', 
                '120', '103', '108', '109', '115', '101', '112', '118', '114', '136', '140', '137', '138', '136', 
                '129', '134', '116', '111', '122', '122', '106', '105', '113', '138', '141', '103', '101', '113', 
                '124', '125']
    avg_mat = np.empty((len(subjects), 320, 320))

    for sub, i in zip(subjects, range(len(subjects))):
        sub_mat = np.loadtxt(ROOT / base_path.format(subject=sub))
        avg_mat[i] = sub_mat

    avg_dist_mat = 1 - avg_mat.mean(axis=0)

    return avg_dist_mat[np.triu_indices(n = avg_dist_mat.shape[0], m=avg_dist_mat.shape[0], k=1)]  

    

def get_fmri_rdm_study2(roi):
    
    ''' 
        Returns the average of the subject RDM matrices for the specified ROI.
    '''

    if roi.isnumeric():
        base_path = f"data/Study2_neural_RSMs_A/DK_roi{roi}_mask/voxel_spearman_mat_SOE320_DK_roi{roi}_{{subject}}.txt"
    if roi=='semantic':
        base_path = f"data/Study2_neural_RSMs_A/ALE_semantic_network/voxel_spearman_mat_SOE320_ALE.99_{{subject}}.txt"
    subjects = ['117', '110', '126', '137', '109', '121', '118', '142', '111', '106', '134', '140', '114', '104', 
                '121', '115', '120', '104', '128', '142', '128', '139', '124', '117', '141', '129', '131', '123', 
                '123', '125', '133', '126', '105', '130', '108', '131', '130', '110', '116', '133', '139', '112', 
                '120', '103', '108', '109', '115', '101', '112', '118', '114', '136', '140', '137', '138', '136', 
                '129', '134', '116', '111', '122', '122', '106', '105', '113', '138', '141', '103', '101', '113', 
                '124', '125']
    avg_mat = np.empty((len(subjects), 320, 320))

    for sub, i in zip(subjects, range(len(subjects))):
        sub_mat = np.loadtxt(ROOT / base_path.format(subject=sub))
        avg_mat[i] = sub_mat

    ret_mat = 1 - np.stack([avg_mat[i][np.triu_indices(n = avg_mat.shape[1], m=avg_mat.shape[1], k=1)]  for i in range(avg_mat.shape[0])]) 
    
    return ret_mat 

def compute_rsa_corr_per_part(brain_mat, model_mat):

    n_part = brain_mat.shape[0]

    rsa_vec = np.empty(n_part)
    for i in range(n_part):
        rho, _ = spearmanr(brain_mat[i], model_mat)
        rsa_vec[i] = rho

    return round(rsa_vec.mean(), 3)


def compute_poolout_alignment_w_exp48(model_embeddings, exp_model_rdm, word_list):

    mod_embeddings = np.stack([model_embeddings[word] for word in word_list])
    rdm = get_distance_vec(mod_embeddings)
    rho, pval = spearmanr(rdm, exp_model_rdm)

    return round(rho, 3)

def compute_layer_alignment_w_exp48(model_embeddings, exp_model_rdm, word_list):

    rho_list = []
    pval_list = []
    
    for layer in range(13):

        mod_embeddings = np.stack([model_embeddings[word][layer] for word in word_list])
        # print(mod_embeddings.shape)
        rdm = get_distance_vec(mod_embeddings)
        rho, pval = spearmanr(rdm, exp_model_rdm)
        rho_list.append(round(rho, 3))
        pval_list.append(round(pval, 3))

    return rho_list


def compute_poolout_alignment_w_brain_aggregated(model_embeddings, brain_rdm, word_list):
    
    '''
        Computes alignment with brain data aggregated across participants.
    '''
    
    mod_embeddings = np.stack([model_embeddings[word] for word in word_list])
    model_rdm = get_distance_vec(mod_embeddings)
    rho_val, _ = spearmanr(model_rdm, brain_rdm)

    return rho_val

def compute_poolout_alignment_w_brain(model_embeddings, brain_rdm, word_list):

    '''
        Computes alignment with single-participant brain data.
    '''

    mod_embeddings = np.stack([model_embeddings[word] for word in word_list])
    model_rdm = get_distance_vec(mod_embeddings)
    rho_val = compute_rsa_corr_per_part(brain_mat=brain_rdm, model_mat=model_rdm)

    return rho_val

def compute_layer_alignment_w_brain_aggregated(model_embeddings, brain_rdm, word_list):
    
    '''
        Computes alignment with brain data aggregated across participants.
    '''
    
    rho_values = []
    for layer in range(13):

        mod_embeddings = np.stack([model_embeddings[word][layer] for word in word_list])
        rdm = get_distance_vec(mod_embeddings)
   
        rho_val, _ = spearmanr(rdm, brain_rdm)
        rho_values.append(rho_val)

    return rho_values

def compute_layer_alignment_w_brain(model_embeddings, brain_rdm, word_list):
    
    '''
        Computes alignment with single-participant brain data.
    '''

    rho_values = []
    for layer in range(13):

        mod_embeddings = np.stack([model_embeddings[word][layer] for word in word_list])
        rdm = get_distance_vec(mod_embeddings)
   
        rho_val = compute_rsa_corr_per_part(brain_mat=brain_rdm, model_mat=rdm)
        rho_values.append(rho_val)

    return rho_values


def average_representations_across_prompts(emb_dict):
    new_dict = {}
    word_list = list(emb_dict.keys())
    n_layers = emb_dict[word_list[0]]['template_1'].shape[0]
    n_templates = len(emb_dict[word_list[0]])
    for word in emb_dict:
        word_repr = []
        for layer in range(n_layers):
            word_repr.append(np.stack([emb_dict[word][f'template_{template}'][layer, :] for template in range(1,n_templates+1)]).mean(axis=0))
     
        new_dict[word] = np.array(word_repr)
     
    return new_dict

def get_representations_for_one_prompt(emb_dict, templ_ind):
    new_dict = {}
    for word in emb_dict:
        new_dict[word] = emb_dict[word][f'template_{templ_ind}']     
    return new_dict

def get_fmri_rdm_study2_aggregated_subset(roi, indices):
    
    ''' 
        Returns the average of the subject RDM matrices for the specified ROI.
    '''

    if roi.isnumeric():
        base_path = f"data/Study2_neural_RSMs_A/DK_roi{roi}_mask/voxel_spearman_mat_SOE320_DK_roi{roi}_{{subject}}.txt"
    if roi=='semantic':
        base_path = f"data/Study2_neural_RSMs_A/ALE_semantic_network/voxel_spearman_mat_SOE320_ALE.99_{{subject}}.txt"
    subjects = ['117', '110', '126', '137', '109', '121', '118', '142', '111', '106', '134', '140', '114', '104', 
                '121', '115', '120', '104', '128', '142', '128', '139', '124', '117', '141', '129', '131', '123', 
                '123', '125', '133', '126', '105', '130', '108', '131', '130', '110', '116', '133', '139', '112', 
                '120', '103', '108', '109', '115', '101', '112', '118', '114', '136', '140', '137', '138', '136', 
                '129', '134', '116', '111', '122', '122', '106', '105', '113', '138', '141', '103', '101', '113', 
                '124', '125']
    avg_mat = np.empty((len(subjects), len(indices), len(indices)))

    for sub, i in zip(subjects, range(len(subjects))):
        sub_mat = np.loadtxt(ROOT / base_path.format(subject=sub))[indices, :]
        avg_mat[i] = sub_mat[:, indices]

    avg_dist_mat = 1 - avg_mat.mean(axis=0)

    return avg_dist_mat[np.triu_indices(n = avg_dist_mat.shape[0], m=avg_dist_mat.shape[0], k=1)]  

       



    