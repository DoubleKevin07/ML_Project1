# Decision Trees
DT_train_binary(X, Y, max_depth)
Here is the process for how my DT_train_binary() function works. It works recursively, returning a tree of Binary Nodes.
	- We get the total entropy for current label set.
	- The entropy is calculated the same way as we have calculated it, thus far. 
	-1 * (probability_no) * log2 (probability_no) – (probability_yes) * log2 (probability_yes)

	- (Base case) If our entropy is 0, or we have run out of samples, or our max_depth is 0, we return nothing.
	- Otherwise, we figure out the optimal feature to split on, given the data set.
		+ Datasets are filled out for both sides, including the decision, a set of samples where the feature is true, the number of labels marked as 0/1 when the feature is true, a set of samples where the decision is false, and the number of labels marked as 0/1 when the feature is false.
		+ We calculate the Information Gain from the data in these sets.
		+ From there, we compare our IG calculation to our “optimal_IG.’, keeping track of the feature that yielded the largest IG. We also keep track of the decision and entropy that corresponds to the optimal IG.
	- Once the optimal feature is calculated, we create a new Binary Node, saving the feature we’re splitting on, the left decision, and the right decision, and the datasets corresponding to both sides.
	- (Recursive Call) We then call DT_train_binary() on the left and right leaves of the Node.



DT_train_binary(X, Y, max_depth)
DT_test_binary() works by calling DT_make_prediction() on each sample in X, and checks to see if the result matches the labels in Y.
	- We keep track of the number the number of tests we performed, and how many were successful.
	- We run DT_make_prediction() on each sample in X, and then test to see if the result matches the corresponding label in Y.
		+ If so, we add 1 to the number of correct answers.
		+ Either way, add 1 to the number of tests performed.
	- When we’re done, we return the accuracy, which is calculated by the number of correct answers, divided by the number of tests that were performed.

DT_make_prediction(x, DT)
DT_make_prediction() works recursively.
	- To make a prediction, we iterate through each of the features in a sample, to find the feature that the given decision tree would have us split on.
		+ Once we’ve found the feature that matches the feature the tree splits on, we check if the feature is 0 or 1.
		+ If the feature is “0”, we then check to see if the given decision tree has a left Node.
			=> (Recursive call) If we do, we call DT_make_prediction() on this Node.
			=> Otherwise, we return the decision the tree made on the left.
		+ Similarly, if the feature is “1”, we then check to see if the given decision tree has a right Node.
			=> (Recursive call) If we do, we call DT_make_prediction() on this Node.
			=> Otherwise, we return the decision the tree made on the right.
# Nearest Neighbors
KNN_test(X_train,Y_train,X_test,Y_test,K)
KNN_test() works by going through each of the test vectors, finding the K closest vectors from the training set, and making a decision based on the majority of the labels for the closest vectors. We then calculate the accuracy by dividing the number of correct decisions by the number of overall tests.
	- We go through each vector in X_test.
		+ We will find the vector in the training set that is closest to the one we’re testing, iterating up to K.
			=> We find the closest vector by calculating the distance between each training vector, and the vector we’re testing. We do this by keeping track of the minimum distance and its corresponding training vector.
				- We calculate distance by squaring the difference between each dimensional value, adding these altogether, and then square rooting the total. (For example, if we were comparing [1,2] with [3,4], we would get distance by doing √((1-3)^2+(2-4)^2 ) )
				- If the distance already matches the value we have stored for minimal distance, we will choose randomly whether or not we want to replace the old minimum distance vector with the new one.
			=> Each time we find the closest vector, we remove this vector from the set of vectors that we can choose from, in order to get K unique vectors.
			=> We also store the label for the closest vector in a list.
			=> We continue to do this until we have the K closest vectors.
		+ Once we are done building the list of labels, we will make a decision based on if the labels are predominantly positive, or negative. 
			=> If the number of positive labels matches the number of negative labels, we will make the decision randomly.
				- Once the decision has been made, we will store whether or not the decision was correct.
	- Once we make a decision for each test vector, we will calculate the accuracy by dividing the number of correct decisions by the number of decisions in total, and return that number.

# Clusters
K_Means(X, K, mu)
K_Means() works by calculating K cluster centers for the sample list X.
	- K_Means() takes mu as a list of centers to start with. However, if mu is empty, we will randomly choose K vectors from the dataset as our centers.
	- From there, we will recalculate the centers on the data points until we get the same set of centers, twice in a row. When we do, this will mean we converged.
		+ We will calculate which center is closest to each vector, and fill out an array with the results, with each entry being a list containing the vectors that are closest to a specific cluster center.
			=> We calculate distance using get_dist() (described below), passing the vector and the center we’re checking.
			=> We keep track of the minimum distance calculated so far, and its corresponding vector. If we calculate a distance that already matches the minimum distance, we make a random decision to choose whether or not to overwrite the minimum with our new vector.
		+ Once we fill out the array, we then recalculate the centers.
			=> We run each list of vectors through get_average_from_set() (defined below).
		+ After the centers are recalculated, we check to see if they match the set of centers we calculated before.
			=> If so, great! We converged, and we return the newly calculated centers.
			=> Otherwise, we store the newly calculated centers in a new array, and then we run through these steps again (starting from calculating which center is closest to each vector).

get_dist (vec_a, vec_b)
get_dist() works by calculating the distance between vec_a and vec_b.
	- We go through each dimension in vec_a and vec_b, and square the difference between them, and this to a total.
	- We then square root the total, and return this value.
		+ (For example, if we were comparing [1,2] with [3,4], we would get distance by doing √((1-3)^2+(2-4)^2 ) )

get_average_from_set(vec_set)
get_average_from_set() works by calculating the average value for each dimension in a vector set, and returning the average vector.
	- First, we calculate the number of dimensions by getting the length of the first vector in the vector set.
	- Then, for each dimension, we calculate the average for that specific dimension, across the entire vector set, and append it to a new array.
	- After calculating the average for all dimensions, we return the array, with the array being the new “average vector”, which should serve as the center between all of the other vectors.


