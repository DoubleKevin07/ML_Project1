import math
import random
import numpy
def K_Means(X, K, mu):
    num_dimensions = len(X[0])

    # Check if mu is empty. If so, fill it with random points.
    if mu == [] or mu is None:

        # Fill with random vectors from the samples, K times.
        for x in range(K):        

            # Get random index
            random_index = random.randint(0, len(X) - 1)
            mu.append(X[random_index])

    # print("MU:", mu)
    converged = False

    C = []
    X_temp = []
    
    while not converged:
        # Fill X temp with K blank arrays.
        X_temp.clear()
        for center in mu:
            X_temp.append([])

        # For all vectors,
        for vector in X:
            min_dist = None
            min_dist_id = 0

            # - Find min dist
            id = 0
            for center in mu:
                dist = get_dist(vector, center)

                if dist == min_dist and random.randint(0,1) == 1:
                    min_dist = dist
                    min_dist_id = id                     
                elif min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_dist_id = id 
                id += 1
            
            # - Put in array with centroid id.
            X_temp[min_dist_id].append(vector)

            
        # - Recompute centers, put this in new array.
        for num in range(len(mu)):
            if len(X_temp[num]) != 0: # Only recalculate if not blank 
                mu[num] = get_average_from_set(X_temp[num])
            
        # - Check to see if this array matches C.
        same = True
        for vec1, vec2 in zip(C, mu):
            if vec1 != vec2:
                same = False

        if len(C) != len(mu):
            same = False
        # - If so, great, converged, return C.
        if same:
            converged = True

        # Otherwise, continue looping, update C.
        else:
            C.clear()
            for item in mu:
                C.append(item)
        
        # print("THE C:", C)

    return numpy.array(C)
                
def get_dist(vec_a, vec_b):
    presquarerooted = 0

    for numA, numB in zip(vec_a, vec_b):
        presquarerooted = presquarerooted + (numA - numB) * (numA - numB)

    return math.sqrt(presquarerooted)

def get_average_from_set(vec_set):

    num_dimensions = len(vec_set[0])

    return_vector = []
    # print("Getting average from vec set:", vec_set)
    for dimension in range(num_dimensions):
        total = 0
        count = 0
        for vector in vec_set:
            total += vector[dimension]
            count += 1
        return_vector.append(total/count)
    
    # print("Average: ", return_vector)
    return return_vector
