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

    norms = []
    for row in usr_matrix:
        norms.append(np.linalg.norm(row))

    return np.array(norms)


def similarity_matrix(usr_matrix: np.ndarray) -> np.ndarray:
    """Calculates cosine similarity between two users and stores them in matrix for easy access.

    Cosine similarity is between 0 and 1. 

    Cosine similarity between user i and user j is given by the value at row i, column j.
    """

    sym_matrix = calculate_dot_product(usr_matrix)
    norms = calculate_norms(usr_matrix)

    # Shape is User x User
    n = usr_matrix.shape[0]

    # Creating empty matrix with known size
    up_simil_matrix = np.zeros((n, n))

    for row in range(n):
        for col in range(row, n):

            # Cosine Similarity Formula
            cosine_sim = get_dot_product(
                row, col, sym_matrix) / (norms[row] * norms[col])

            up_simil_matrix[row][col] = cosine_sim

    # Reflect the Upper Triangle Matrix to Lower Half
    simil_matrix = up_simil_matrix + up_simil_matrix.T - \
        np.diag(np.diag(up_simil_matrix))

    return simil_matrix


def get_weighted_avg(user1: int, item: int, simil_matrix: np.ndarray, usr_matrix: np.ndarray):

    n = simil_matrix.shape[0]

    numerator = 0
    denominator = 0
    for i in range(n):
        # Another users rating for movie "item" from 0-5
        useri_rating_for_item = usr_matrix[i][item]
        if i != user1 and useri_rating_for_item != 0:
            # Cosine similarity between user1 and useri
            cosine_similarity = simil_matrix[user1][i]
            numerator += cosine_similarity * usr_matrix[i][item]
            denominator += cosine_similarity

    return round(numerator/denominator, 1)


if __name__ == '__main__':

    usr_matrix = create_matrix('ratings.csv')
    dot_products = calculate_dot_product(usr_matrix)
    simil_matrix = similarity_matrix(usr_matrix)
    print(get_weighted_avg(1, 2, simil_matrix, usr_matrix))
