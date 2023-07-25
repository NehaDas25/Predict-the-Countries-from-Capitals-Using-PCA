# Work Report

## Information

- Name: <ins> DAS,NEHA </ins>
- CIN: <ins> 401457144  </ins>
- GitHub: <ins> NehaDas25 </ins>
- Email: <ins> nda@calstatela.edu </ins>


## Features

- Not Implemented:
  - PART 3: Plotting for compute_pca.
    - In this scatter plot, it was concluded that the word vectors for   gas, oil and petroleum appear related to each other, because their vectors are close to each other. Similarly, sad, joyful and happy all express emotions, and are also near each other.
  - PART 1: The word embeddings as a Python dictionary was loaded. As stated, these have already been obtained through a machine learning algorithm.

<br><br>

- Implemented:In the assignment, to achieve the end goal, all parts are implemented sequentially.


  - PART-1: Predict the Countries from the Capitals
  - PART-1.2: Implemented the Cosine Similarity function using the formula provided. This passed all the unit-test function.
  - Understood the cosine similarity, it's a function that takes in two word vectors and computes the cosine distance. 
  - dot variable is used the store dot product of A and B.
  - norma and normb are used to store the square roots of A and B.
  - cos variable has been used to store numerical number representing the cosine similarity between A and B.
  - Tested the function with different word, for example the words 'King' and 'queen'. Here word_embeddings has been loaded as a Python dictionary which also has been used to find cosine similarity.
  - This passed all the unit-test cases. 

  
  - PART 1.3: Implemented the Euclidean function d using the formula provided.
  - Here np.linalg.norm has been used , NumPy package contains the moddule numpy.linalg module that provides all the functionality for linear algebra.
  - d is the variable that stores numerical number representing the Euclidean distance between A and B ( both A and B are the numpy arrays that corresponds to word vectors).
  - This passed all the unit-test cases as well.


 - PART 1.4: Finding the Country of each Capital. Implemented the get_country function.
 - The parameters passed city1, country1, city2, embeddings, cosine_similarity().
 - city1, country1, city2 are stored in a set called group and the embeddings of the above mentioned three terms has been found out.
 - Using the word embeddings and a similarity function, predicting the relationship among the words has implemented.For example, to predict the capital we might want to look at the "King - Man + Woman = Queen".
 - So, the embedding of vec(country2) = country1_emb - city1_emb + city2_emb has been calculated.
 - the similarity has been initialized to -1. We will loop through all the words in the embeddings dictionary. Calculated cosine similarity between embedding of country2 and the word in the embeddings dictionary.
 - Checked if the cosine similarity is more similar than the previously best similarity , updated the similarity to the new, better similarity and stored the country as a tuple, country = (word, similarity).
 - This passed all the unit-test cases as well.


 - PART 1.5: MODEL ACURRACY.
 - Implemented the get_accuracy function along with the parameters passed that are word_embeddings, data, get_country().
 - num_correct has been initialized to 0 and looping through the rows in dataframe was done.
 - Used get_country() to find the predicted country2 and comparison was done between the predicted country2 and actual country2. num_correct was incremented to 1, if predicted_country2 == country2. 
 - the number of rows in the data dataframe was calculated that is m = len(data) and Accuracy was found by dividing the num_correct by m.
 - This passed all the unit-test cases.


 
 
 - PART 3: Plotting the vectors using PCA. Implemented the compute_pca() with the parameters passed as X, n_components=2.
    - Understood the PCA algorithm.
    - The word vectors are of dimension 300.Used PCA to change the 300 dimensions to n_components dimensions.The new matrix should be of dimension m, n_componentns. 
    - To demean the data,numpy.mean(a,axis=None) was used and axis was set to 0, to take the mean for each column and stored under X_demeaned.
    - Used numpy.cov(m, rowvar=False) to calculate the covariance matrix. Here rowvar is False because in our case, each row is a word vector observation, and each column is a feature (variable).
    - Used numpy.linalg.eigh(covariance_matrix, UPLO='L') to get eigen_vals, eigen_vecs.
    - Used numpy.argsort, that sorts the eigenvalue in an array from smallest to largest, then returns the indices from this sort and stored under idx_sorted.
    - Reversed the order so that it's from highest to lowest, stored under idx_sorted_decreasing and then sorted the eigen_vals by idx_sorted_decreasing and eigen_vecs using idx_sorted_decreasing indices.
    - Got a subset of the eigenvectors and transformed the data by multiplying the transpose of the eigenvectors with the transpose of the de-meaned data(X_demeaned).
    - The transpose was taken of the dot product and stored under X_reduced.
    - Testing was done using X = np.random.rand(3, 10).
    - This passed all the unit-test cases as well.
 



<br><br>

- Partly implemented:
  - utils.py which contains get_vectors has not been implemented, it was provided.
  - ws_unittest.py was also not implemented as part of assignment to pass all unit-tests for the graded functions().

<br><br>

- Bugs
  - No bugs

<br><br>


## Reflections

- Assignment is very good. Gives a thorough understanding of the basis of Vector space, PCA, Predicting the relation among words and word embeddings.


## Output

### output:

<pre>
<br/><br/>
  Out[2] - 
  
  city1	country1	city2	  country2
0	Athens	Greece	Bangkok	Thailand
1	Athens	Greece	Beijing	 China
2	Athens	Greece	Berlin	 Germany
3	Athens	Greece	Bern	  Switzerland
4	Athens	Greece	Cairo	  Egypt
    
  Out[3] - 243

  Out[4] - dimension: 300
  
  Out[6] - 0.651095680465667
  Expected Output: ≈ 0.651095
    
  Out[7] - All tests passed

  Out[9] - 2.4796925
  Expected Output: 2.4796925

  Out[10] - All tests passed
  
  Out[12] - ('Egypt', 0.7626821213022329)
  **Expected Output: (Approximately)**
    ('Egypt', 0.7626821)

  Out[13] - All tests passed
  
  Out[16] - Accuracy is 0.92
  Expected Output: ≈ 0.92
    
  Out[17] - All tests passed

  Out[27] - Your original matrix was (3, 10) and it became:
[[ 0.43437323  0.49820384]
 [ 0.42077249 -0.50351448]
 [-0.85514571  0.00531064]]

  Out[28] - All tests passed

  Out[29] - You have 11 words each of 300 dimensions thus X.shape is: (11, 300)
  
  Out[30] - Scatter Plot   
  <img src="https://user-images.githubusercontent.com/100334984/218356320-ea87869d-e580-4bf1-a44b-8e4b0969cf2b.png" />
<br/><br/>
</pre>
