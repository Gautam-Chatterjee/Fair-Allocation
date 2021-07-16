import torch
import numpy as np

# np.load np.fromtxt
#valuations = np.array([[1, 0],[0, 1]]) 
#valuations = np.array([[4, 2, 3, 1, 5], [1, 2, 3, 5, 4], [2, 1, 3, 5, 4], [5, 2, 3, 4, 1], [3, 1, 2, 5, 4], [3, 5, 4, 1, 2], [3, 4, 1, 2, 5], [4, 2, 1, 5, 3], [2, 4, 5, 3, 1], [2, 3, 4, 5, 1]])
valuations = np.array([[7, 4, 5, 2, 3, 10, 9, 1, 6, 8], [6, 5, 1, 2, 7, 4, 8, 3, 9, 10], [1, 5, 3, 7, 10, 8, 2, 6, 4, 9], [6, 5, 7, 2, 8, 9, 10, 3, 1, 4], [1, 3, 5, 7, 9, 8, 2, 10, 4, 6], [10, 2, 9, 6, 4, 5, 3, 7, 8, 1], [5, 9, 6, 8, 2, 10, 4, 7, 3, 1], [3, 6, 4, 10, 5, 8, 2, 7, 1, 9], [4, 5, 10, 2, 8, 9, 7, 6, 1, 3], [3, 2, 5, 7, 10, 9, 8, 6, 4, 1]])
# Create variables B and V
# shape = [2, 2]
B = torch.randn(valuations.shape, requires_grad=True)
V = torch.from_numpy(valuations).float()
#V += np.random.rand(V.shape)*.001
print(V)

learning_rate = 1e-1
optimizer = torch.optim.Adam([B], lr=learning_rate)

for iter in range(10000):
    # TODO: compute loss function (-1 * USW)
    optimizer.zero_grad()
    
    # 10 -4
    # 3  7

    # e^10/(e^10 + e^3)     e^-4/(e^-4 + e^7)
    # e^3/(e^10 + e^3)      e^7/(e^-4 + e^7)


    A = torch.softmax(B, axis=0)

    # TODO: Look at Pytorch docs and figure out how to compute V A^T
    A_trans = torch.transpose(A,0,1) # Fix syntax
    W = torch.matmul(V, A_trans)

    # 1 2
    # 3 4
    D = torch.diag(W)
    # [1 4]

    loss = -1*torch.sum(D)

    #print(loss)
    

    # Compute gradient of loss with respect to B
    loss.backward()
    # Update B
    optimizer.step()
#print(A)
# print out B and A
# Hopefully B = [[ 100M -100M]
#                [ -100M 100M ]]

# Then A = [[ ~1  ~0]
#           [ ~0  ~1]]
print(A)
#torch.allclose(A, , rtol=1e-05, atol=1e-08, equal_nan=False)