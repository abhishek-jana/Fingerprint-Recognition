import numpy as np
import time
import miniball
S = np.loadtxt('dataset.txt')
#S = np.random.randn(100,10)
tic = time.time()
C,r2 = miniball.get_bounding_ball(S)

#print (f'center is = {C}')
print (f'radius squared is = {r2}')
print (f'Total time = {time.time() - tic} sec')

np.savetxt('rad.txt',(C,r2))
