# decode a vector

# REF: https://arxiv.org/pdf/0704.1317.pdf

# import libraries
import numpy as np
import scipy
from scipy import interpolate, sparse
import utilsv2 as utils
from multiprocessing import Pool

matrix = np.loadtxt('./generator_matrix_d3.txt')
datafile = np.loadtxt('./dataset.txt')

datafile = datafile.reshape(-1,128,1)

def create_sparse_matrix(mat,data):
    """
    Creates the generator matrix
    """
    rows = mat.shape[0]
    cols = mat.shape[1]
    row_id = mat.ravel()
    col_id = np.tile(np.arange(cols),rows)
    mat_data = np.repeat(data,cols)
    mat_data = [each*np.random.choice([-1,1]) for each in mat_data]
    
    matrix = sparse.csr_matrix((mat_data, (row_id, col_id)), shape=(cols, cols)).toarray()
    
    return matrix

vals = [1/2.31 , 1/3.17, 1/5.11 , 1/7.33 , 1/11.71]

H = create_sparse_matrix(matrix,vals[:3])

H = H/np.abs(np.linalg.det(H))**(1/128.) # normalize H
G = np.linalg.inv(H)

np.random.seed(1234)
# b = np.random.randint(10,size = 128)
# x = np.dot(G,b)
# x = x.reshape(-1,1)
mu = 0# mean and standard deviation

x_input = np.linspace(-40.0,40,801)

sigma = 0.09

def error_count(data):
    x_data = []
    sig = 0.09
    data = data.reshape(128,1)
    w = np.random.normal(mu, sig, data.shape)
    

    y = data + w

    cnode = utils.CheckNode(H,x_input,y,sig)
    vnode = utils.VariableNode(H,x_input,y,sig)
    res = utils.init_message(x_input, H,y,sig)
    for i in range(2):
        q = cnode.Q(res)
        f = vnode.f(q)
        res = f
    f_final = vnode.final(q)
    x_b = np.argmax(f_final,axis = 1)
        #b_cal = np.rint(np.dot(H,x_input[x_b]))
        
#         print (b_cal[:10].astype(int),b[:10])
    x_data = x_input[x_b]
    return x_data
    
if __name__ == '__main__':
    with Pool(50) as p:
        res = p.map(error_count, datafile)




#print (res,sigma)

np.save('data_decoded.npy',res)