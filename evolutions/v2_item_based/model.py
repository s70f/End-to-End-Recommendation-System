import numpy as np


def create_matrix(file: str) -> np.ndarray:
    """Returns a item-user matrix from a csv where the columns are users and the rows are items. Null values are represented by 0's.

    Representation Invariant:
    - csv is seperated by commas
    - csv has header
    - Ratings on scale from 1-5
    """

    data = np.genfromtxt(file, delimiter=',',
                         filling_values=0, skip_header=1)

    data = data[:, 1:].T
    return data


def similarity_matrix(item_user_matrix) -> np.ndarray:
    """Calculates cosine similarity between two items and stores them in matrix for easy access.

    Cosine similarity is between 0 and 1. 

    Cosine similarity between item i and item j is given by the value at row i, column j.
    """

    # Create item-user matrix where rows are items

    dot_products = item_user_matrix @ item_user_matrix.T
    norms = np.linalg.norm(item_user_matrix, axis=1)

    denominators = np.outer(norms, norms)

    sim_matrix = dot_products / denominators

    return sim_matrix


def get_weighted_avg(user_id: int, item_id: int, sim_matrix: np.ndarray, itm_matrix: np.ndarray):

    item_sims = sim_matrix[item_id]

    item_ratings = itm_matrix[item_id]

    numerator = np.sum(item_sims * item_ratings)  # Another 1D array output

    # Numpy creates an array with T/F values then compares index to index
    relevant_sims = item_sims[item_ratings > 0]
    denominator = np.sum(relevant_sims)

    if denominator > 0:
        return round(numerator / denominator, 1)
    else:
        return 0


if __name__ == '__main__':

    # Create item-user matrix
    item_user_matrix = create_matrix(
        "/home/suhaybsana/Projects/DataScience/recsys/End-to-End-Recommendation-System/evolutions/v2_item_based/ratings.csv")

    # Create similarity scores matrix
    item_sims = similarity_matrix(item_user_matrix)

    # Get estimed ratings
    e_rating = get_weighted_avg(
        1, 3, sim_matrix=item_sims, itm_matrix=item_user_matrix)

    print(e_rating)
