import numpy as np


def create_matrix(file: str) -> np.ndarray:
    """Returns a user-item matrix from a csv where the columns are items and the rows are users. Null values are represented by 0's.

    Representation Invariant:
    - csv is seperated by commas
    - csv has header
    - Ratings on scale from 1-5
    """

    data = np.genfromtxt(file, delimiter=',', filling_values=0, skip_header=1)
    return data


def calculate_dot_product(usr_matrix: np.ndarray):
    """Calculate dot product of all user vectors using matrix multiplication. Note that a user vector is one row of ratings.

    Multiply a matrix by it's transpose means the value at (i, j) is the dot product of row i and row j (user i and user j))
    """

    # Note: Null ratings don't effect dot products, since we made null ratings = 0.
    return usr_matrix @ usr_matrix.T


def get_dot_product(user1: int, user2: int, sim_matrix: np.ndarray):
    """Gets the dot product between two users by indexing a numpy array of symmetric square matrix.

    Representation Invariants:
    - Matrix must be AA^T Matrix
    """

    return sim_matrix[user1][user2]


def calculate_norms(usr_matrix: np.ndarray) -> np.ndarray:
    """Calculates norms for each row/user and stores them in an index sorted 1D array"""

    # axis 1 means for each row in our matrix
    return np.linalg.norm(usr_matrix, axis=1)


def similarity_matrix(usr_matrix: np.ndarray) -> np.ndarray:
    """Calculates cosine similarity between two users and stores them in matrix for easy access.

    Cosine similarity is between 0 and 1. 

    Cosine similarity between user i and user j is given by the value at row i, column j.
    """

    dot_products = calculate_dot_product(usr_matrix)
    norms = calculate_norms(usr_matrix)

    denominators = np.outer(norms, norms)

    sim_matrix = dot_products / denominators

    return sim_matrix


def get_weighted_avg(user_id: int, item_id: int, sim_matrix: np.ndarray, usr_matrix: np.ndarray):

    user_sims = sim_matrix[user_id]

    movie_ratings = usr_matrix[:, item_id]

    numerator = np.sum(user_sims * movie_ratings)  # Another 1D array output

    # Numpy creates an array with T/F values then compares index to index
    relevant_sims = user_sims[movie_ratings > 0]
    denominator = np.sum(relevant_sims)

    return round(numerator / denominator, 1) if denominator != 0 else 0


if __name__ == '__main__':

    usr_matrix = create_matrix('ratings.csv')
    dot_products = calculate_dot_product(usr_matrix)
    simil_matrix = similarity_matrix(usr_matrix)
    print(get_weighted_avg(1, 2, simil_matrix, usr_matrix))
