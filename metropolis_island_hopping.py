import random

# Codility (Prefix sum) inspired - MCMC
def get_posterior_from_Metropolis(A, N, start, burn_in):

    ''' A: the distribution to be predicted (a list).
        N: the number of trials / moves
        start: starting index in A
        burn_in: the number of  initial values to be discarded.
    '''

    Posterior = [0] * len(A) # to be returned
    cur_pos = start # starting index

    for i in range(N):
   
        # proposal distribution is (-1, +1) cordinates, takng into account the list boundaries

        proposal_ratio = 1 #The ratio of proposal probabilities, change when moving to or from list boundaries

        if cur_pos<len(A)-1 and cur_pos>0:
            proposed_positions = [cur_pos-1, cur_pos+1]
        elif cur_pos == len(A)-1:
            proposed_positions =[cur_pos-1] # if this is enforced the posterior does not look like A
            proposal_ratio = 0.5 # moving away from a boundary
            #proposed_positions = [0,cur_pos-1] This makes the islands cyclic, you can jump from the last island to the first.
        elif cur_pos == 0 :
            proposed_positions =[cur_pos+1]
            proposal_ratio = 0.5 # moving away from a boundary
            #proposed_positions = [len(A)-1, cur_pos+1] # if proposed_positions =[cur_pos+1], the posterior does not look like A

        proposed_pos = random.choice(proposed_positions)

        #  if moving to a boundary (0th or the nth element in the list)
        if proposed_pos == 0 or proposed_pos == len(A)-1:
            proposal_ratio = 2 # prob of moving away from boundary to proposed_pos (1) is twice the prob of moving into cur_pos from proposed_pos (1/2)

        # decide whether you will move
        r = (float(A[proposed_pos])/A[cur_pos]) * proposal_ratio

        if r>1: # definitely move
            cur_pos = proposed_pos
        else: # move with a prob proportional to how close A[cur_pos] is to A[proposed_pos] 

            u = random.random() # sample from uniform distribution from 0 to 1

            if r > u: # move to proposed
                 cur_pos = proposed_pos 
  
        # if past burn_in point, increment the location by 1 in the posterior distribution.
        if i > burn_in: 
            Posterior[cur_pos] +=  1

    return Posterior

A = [2,3,1,5,8,2,9] # distribution to be predicted.
posterior_A = get_posterior_from_Metropolis(A, 4000, 3, 1000)

# Validate the sampling works:
print [n/float(sum(A)) for n in A]
print [n/float(sum(posterior_A)) for n in posterior_A]
