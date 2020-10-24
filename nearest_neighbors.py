import math
import random

def KNN_test(X_train,Y_train,X_test,Y_test,K):
    
    num_tested = 0
    num_correct = 0

    id = 0
    for vector in X_test:

        # Fill list of vectors we will test.
        vector_train_data = []
        for vec in X_train:
            vector_train_data.append(vec)

        label_data = []


        # print("Vector we're testing: ", vector)
        # Find closet vector in X_train, add label Y_train[id] to array, K times

        used_vectors = []
        for k in range(K):

            min_dist_id = 0
            min_dist = None
            min_vec = None

            
            # Find shortest distant vector
            curr_id = 0
            for vec in vector_train_data:

                # If we already tested the vector,
                if curr_id in used_vectors:
                    # print("skipping ", vec)
                    continue # Keep going.

                before_sqrt = 0
                for val1, val2 in zip(vector, vec):
                    before_sqrt += (val1 - val2) * (val1 - val2)
                sqrted = math.sqrt(before_sqrt)

                if min_dist is None or sqrted < min_dist or (sqrted == min_dist and random.randint(0,1) == 1):
                    min_dist = sqrted
                    min_dist_id = curr_id
                    min_vec = vec

                curr_id += 1

            # Add its label to the array.
            label_data.append(Y_train[min_dist_id])

            # Remove from list of vectors to test.
            used_vectors.append(min_dist_id)

            # print("Closest vector: ", min_vec)
            # print("Closest vector label: ", Y_train[min_dist_id])
            # print("Closest vector distance: ", min_dist)

        # Make decision
        num_pos = 0
        num_neg = 0
        for label in label_data:
            if label == -1:
                num_neg += 1
            if label == 1:
                num_pos += 1
        
        decision = 0
        if num_pos > num_neg or (num_pos == num_neg and random.randint(0,1) == 1):
            decision = 1
        else:
            decision = -1

        # print("KNN: Decision for ", vector)
        # print(decision)

        # Test with y[id]
        if decision == Y_test[id]:
            num_correct += 1
        num_tested += 1

        id += 1

    return num_correct / num_tested