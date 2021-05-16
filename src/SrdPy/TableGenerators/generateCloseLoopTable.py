import numpy as np
def generateCloseLoopTable(A_table, B_table,C, c_table, K_table,L_table, x_table, u_table):
   count = A_table.shape[0]
   n = A_table.shape[1]

   AA_table = np.zeros((count,n*2, n*2))
   cc_table = np.zeros((count,2*n))

   for i in range(count):
       AA_table[i]=np.block([
         [A_table[i],-B_table[i]@K_table[i]],
         [L_table[i]@C, A_table[i] - B_table[i]@K_table[i]-L_table[i]@C] 
       ])

       cc_table[i]=np.block([
         B_table[i] @ (K_table[i] @ x_table[i] + u_table[i]) + c_table[i],
         B_table[i] @ (K_table[i] @ x_table[i] + u_table[i]) + c_table[i]
       ])


   return AA_table, cc_table
