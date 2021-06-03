import numpy as np
import itertools
from scipy import sparse

def create_matrix(size,degree,seed = 42):
    """
    Returns P matrix
    
    Parameters
    ------------
    size: size of the parity check matrix(n)
    degree: degree(d)
    
    Returns
    ------------
    P matrix of d X n
    """
    np.random.seed(seed)
    arr = np.tile(np.arange(size),(degree,1))
    return np.vstack([np.random.permutation(i) for i in arr])

def uni(records_array):
    # retruns the length of an array if the items are unique
    idx_sort = np.argsort(records_array)
    sorted_records_array = records_array[idx_sort]
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
    if len(vals) < len(records_array):
        res = np.split(idx_sort, idx_start[1:])
        return dict(zip(vals, res))

def swap_elements(matrix,row,col,newcol):
    """
    swap columns of a matrix in a particular row
    """
    matrix[row,[col,newcol]] = matrix[row,[newcol,col]]    
    
def has_two_loop(mat):
    """
    Checks if the matrix has two loops
    """
    rows = mat.shape[0]
    for i in range(mat.shape[1]):
        if len(np.unique(mat[:,i])) < rows:
            return True
    return False

def return_two_loop_cols(mat):
    col_id = []
    col_len = mat.shape[0]
    for col in range(mat.shape[1]):
        if len(np.unique(mat[:,col])) < col_len:
            col_id.append(col)
    return col_id

def return_duplicate_ids(array):
    """
    Given an array, returns the duplicated id 
    """
    arr = uni(array)
    ids = []
    if arr != None:
        for key in arr.keys():
            while len(arr[key]) > 1:
                idx, arr[key] = arr[key][-1], arr[key][:-1]
                ids.append(idx)
        return ids     
    
def remove_two_loop(mat):
    """
    Function to remove two loops
    """
    two_loop = has_two_loop(mat)
    if two_loop == True:
        while (two_loop == True):
            cols = return_two_loop_cols(mat)
            for col_id in cols:
                rows = return_duplicate_ids(mat[:,col_id])
                for row_id in rows:
                    new_col_id = np.random.choice([x for x in range(mat.shape[1]) if x != col_id])
                    mat[row_id,new_col_id], mat[row_id,col_id] = mat[row_id,col_id], mat[row_id,new_col_id]
            two_loop = has_two_loop(mat)
    return mat             


def has_four_loop(mat):
    """
    checks if a matrix has four loops or not
    """
    for i in range(mat.shape[1]):
        for col_id in range(i):
            intersect = np.intersect1d(mat[:,col_id],mat[:,i])
            if len(intersect) > 1 :
                return True
    return False

def remove_four_loop(mat):
    """
    Function to remove four loops
    """
    four_loop = has_four_loop(mat)
    while (four_loop == True):
        for i in range(mat.shape[1]):
            for col_id in range(i):
                #print (f'comparing col:{col_id}, col:{i}')
                list1 = list(itertools.permutations(mat[:,col_id],2))
                list2 = list(itertools.permutations(mat[:,i],2))
                intersect = list(set(list1).intersection(list2))
                if len(intersect) > 0 :
                    row_id = np.where(mat[:,col_id] == list1[0][0])
                    new_col_id = np.random.choice([x for x in range(mat.shape[1]) if x != col_id])
                    mat[row_id,new_col_id], mat[row_id,col_id] = mat[row_id,col_id], mat[row_id,new_col_id]
        four_loop = has_four_loop(mat)
    return mat  

def remove_two_loopv2(mat):
    """
    Function to remove two loops
    """
    two_loop = has_two_loop(mat)
    #if two_loop == True:
    while (two_loop == True):
        cols = return_two_loop_cols(mat)
#         print (final_mat)
        for col_id in cols:
            rows = return_duplicate_ids(mat[:,col_id])
            for row_id in rows:
                new_col_id = np.random.choice([x for x in range(mat.shape[1]) if x != col_id])
                swap_elements(mat,row = row_id,col = col_id,newcol = new_col_id)
        two_loop = has_two_loop(mat)

def remove_four_loopv2(final_mat):
    """
    Function to remove four loops
    """
    four_loop = has_four_loop(final_mat)
    while (four_loop == True):
        for i in range(final_mat.shape[1]):
            for col_id in range(i):
                intersect,x_ind,_ = np.intersect1d(final_mat[:,col_id],final_mat[:,i], return_indices=True)
                if len(intersect) > 1 :
                    new_col_id = np.random.choice([x for x in range(final_mat.shape[1]) if x != col_id and x != i])
                    #print (f'intersect = {intersect}, col1_id = {col_id}, col2_id = {i}, row = {x_ind}, new_col_id = {new_col_id},\n matrix = \n {final_mat}',end = '\r')
                    swap_elements(final_mat,row = x_ind[0],col = col_id,newcol = new_col_id)
                    four_loop = has_four_loop(final_mat)

test_matrix = create_matrix(size=128,degree = 7)


two_loop = has_two_loop(test_matrix)
four_loop = has_four_loop(test_matrix)
count = 0
while (two_loop == True) or (four_loop == True):
    print(f'iter: {count+1}', end='\r')
    remove_two_loopv2(test_matrix)
    #print ("two loop removed")
    remove_four_loopv2(test_matrix)
    #print ("four loop removed")
    #print (test_matrix)
    two_loop = has_two_loop(test_matrix)
    four_loop = has_four_loop(test_matrix)
    
    count += 1
print()


# two_loop = has_two_loop(matrix)
# four_loop = has_four_loop(matrix)

# while (two_loop == True) or (four_loop == True):
#     matrix = remove_two_loop(matrix)
#     matrix = remove_four_loop(matrix)
#     two_loop = has_two_loop(matrix)
#     four_loop = has_four_loop(matrix)


np.savetxt('generator_matrix.txt',test_matrix,fmt = '%i')

#np.savetxt('generator_matrix.txt',matrix)











