import math
import random

class BinaryNode:

    def __init__(self, start_entropy):
        
        self.entropy = start_entropy
        self.left = None
        self.left_decision = None
        self.right = None
        self.right_decision = None
        self.feature_to_split = None


def DT_train_binary(X, Y, max_depth):
    entropy = get_entropy(Y)
    total = len(Y)
    node = BinaryNode(entropy)
    
    # print("Start entropy: ", entropy)
    # See where to split given data.

    # This is the data we will use to keep track of, later on.
    optimal_IG = 0
    optimal_feature = None
    optimal_decision_left = None
    optimal_entropy_left = None
    optimal_left_X = []
    optimal_left_Y = []
    optimal_decision_right = None
    optimal_entropy_right = None
    optimal_right_X = []
    optimal_right_Y = []
    num_features = len(X[0])

    # print("Num features: ", num_features)
    # Base case.
    if num_features == 0 or entropy == 0 or max_depth == 0:
        return None

    # Go through all features.
    for num in range(0, num_features):


        # Build datasets for choice "YES" and choice "NO"
        # Ex. num_yes_yes means, when we say that the feature = 1, and the label = 1.
        # num_yes_no means, feature = 1, label = 0.
        new_Y_yes = [] # Dataset containing labels for when feature is true
        new_X_yes = []
        num_yes_yes = 0
        num_yes_no = 0
        decision_yes = 0
        count_yes = 0

        new_Y_no = [] # Dataset containing labels for when feature is not true
        new_X_no = []
        num_no_yes = 0
        num_no_no = 0
        decision_no = 0
        count_no = 0

        # Go through all the rows and populate the data above, given feature "num"
        for row_number, row in zip(range(len(X)), X):
            if row[num] == 1:
                new_Y_yes.append(Y[row_number])
                new_X_yes.append(row)

                if Y[row_number] == 1:
                    num_yes_yes += 1
                elif Y[row_number] == 0:
                    num_yes_no += 1

            elif row[num] == 0:
                new_Y_no.append(Y[row_number])
                new_X_no.append(row)

                if Y[row_number] == 1:
                    num_no_yes += 1
                elif Y[row_number] == 0:
                    num_no_no += 1

        # Calculate entropy for both sides:
        entropy_left = get_entropy(new_Y_no)
        entropy_right = get_entropy(new_Y_yes)

        # print("For feature: ", num)
        # print("entropy_left: ", entropy_left)
        # print("entropy_right: ", entropy_right)
        # Make the decisions for both sides
        # Left (feature = 0):
        if num_no_no > num_no_yes:
            decision_no = 0
        else:
            decision_no = 1

        # Right (feature = 1):
        if num_yes_no > num_yes_yes:
            decision_yes = 0
        else:
            decision_yes = 1

        count_no = num_no_no + num_no_yes
        count_yes = num_yes_no + num_yes_yes

        # Calculate IG
        IG = entropy - ( (count_no/total) * entropy_left + (count_yes/total) * entropy_right )
        # print("IG: ", IG)
        if IG > optimal_IG:
            # Now we replace the optimal choice with everything.
            optimal_IG = IG
            optimal_feature = num # num = feature num. RFE???

            # Will need to rewrite this to build into the node directly.
            optimal_decision_left = decision_no
            optimal_entropy_left = entropy_left

            optimal_left_X.clear()
            for item in new_X_no:
                optimal_left_X.append(item)
            optimal_left_Y.clear()
            for item in new_Y_no:
                optimal_left_Y.append(item)

            optimal_decision_right = decision_yes
            optimal_entropy_right = entropy_right

            optimal_right_X.clear()
            for item in new_X_yes:
                optimal_right_X.append(item)
            optimal_right_Y.clear()
            for item in new_Y_yes:
                optimal_right_Y.append(item)

    # Now build the trees for the other branches.
    node.left_decision = optimal_decision_left
    node.right_decision = optimal_decision_right
    node.feature_to_split = optimal_feature
    # print("Splitting on feature: ", optimal_feature)
    node.left = DT_train_binary(optimal_left_X, optimal_left_Y, max_depth-1)
    node.right = DT_train_binary(optimal_right_X, optimal_right_Y, max_depth-1)

    return node



# Get entropy of all the given labels.
def get_entropy(Y):
    num_yes = 0
    num_no = 0
    for label in Y:
        if label == 1:
            num_yes += 1
        else:
            num_no += 1
    
    total = num_no + num_yes

    # If there is no data to do calculations with...
    if total == 0:
        return 0 # then our entropy is 0!

    probability_no = num_no/total
    probability_yes = num_yes/total

    # Avoid a math domain error by checking if the probability is 0.
    first_calc = 0
    if probability_no != 0:
        first_calc = -1 * probability_no * math.log2(probability_no)

    second_calc = 0
    if probability_yes != 0:
        second_calc = -1 * probability_yes * math.log2(probability_yes)

    return first_calc + second_calc

def DT_make_prediction(x, DT): # Let DT be our node.
    feature_id = 0
    for feature in x:
        if DT.feature_to_split == feature_id:

            # Check left
            if feature == 0:
                if DT.left is None:
                    return DT.left_decision
                else:
                    return DT_make_prediction(x, DT.left)

            # Check right
            if feature == 1:
                if DT.right is None:
                    return DT.right_decision
                else:
                    return DT_make_prediction(x, DT.right)

        feature_id += 1

def DT_test_binary(X,Y,DT):
    num_tested = 0
    num_correct = 0

    id = 0

    for sample in X:
        result = DT_make_prediction(sample, DT)
        if result == Y[id]:
            num_correct += 1
        num_tested += 1    
        id += 1

    return num_correct / num_tested

