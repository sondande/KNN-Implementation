[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8488461&assignment_repo_type=AssignmentRepo)
# hw1-knn
HW1: k-Nearest Neighbors

This assignment contains four data sets which are based on three publicly available benchmarks:

1. monks1.csv: A data set describing two classes of robots using all nominal attributes and a binary label.  This data set has a simple rule set for determining the label: if head_shape = body_shape  jacket_color = red, then yes, else no. Each of the attributes in the monks1 data set are nominal.  Monks1 was one of the first machine learning challenge problems (http://www.mli.gmu.edu/papers/91-95/91-28.pdf).  This data set comes from the UCI Machine Learning Repository: http://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems

2. penguins.csv: A data set describing observed measurements of different animals belonging to three species of penguins.  The four attributes are each continuous measurements, and the label is the species of penguin.  Special thanks and credit to Professor Allison Horst at the University of California Santa Barbara for making this data set public: see this Twitter post and thread with more information (https://twitter.com/allison_horst/status/1270046399418138625) and GitHub repository (https://github.com/allisonhorst/palmerpenguins).

3. mnist_100.csv: A data set of optical character recognition of numeric digits from images.  Each instance represents a different grayscale 28x28 pixel image of a handwritten numeric digit (from 0 through 9).  The attributes are the intensity values of the 784 pixels. Each attribute is ordinal (treat them as continuous for the purpose of this assignment) and a nominal label.  This version of MNIST contains 100 instances of each handwritten numeric digit, randomly sampled from the original training data for MNIST.  The overall MNIST data set is one of the main benchmarks in machine learning: http://yann.lecun.com/exdb/mnist/.  It was converted to CSV file using the python code provided at: https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html

4. mnist_1000.csv: The same as mnist_100, except containing 1000 instances of each handwritten numeric digit.

## Run program:

Run the following commands in terminal to run program

```bash
# Install virtualenv
pip install virtualenv

# Create Virtual Environment
virtualenv <your_virtual_env>

# Activate Virtual Environment
source <your_virtual_env>/bin/activate

# Install Dependencies
pip install -r requirements.txt

# Run Program
python knn.py -f <data_set> -d <Distance_Function> -k <k_value> -t <test_set_percentage>

# Example Run Command with p value and random seed included
python knn.py -f data-sets/penguins.csv -d M -k 2 -t 0.7 -r 5
```

## Research Questions
Please use your program to answer these research questions and record your answers in a README.md file:
1) Pick a single random seed and a single training set percentage (document both in your README) and run k-Nearest Neighbors with a k = 1 on each of the four data sets. What is the accuracy you observed on each data set?
    - Random Seed: 12
    - Training Set Percentage: 0.75
    - Results: 
      * **monks1.csv** : 73.14% _(0.7314814814814815)_ accuracy rate
      * **penguins.csv** : 91.86% _(0.9186046511627907)_ accuracy rate
      * **mnist_100.csv** : 85.6% _(0.856)_ accuracy rate
      * **mnist_1000.csv** : 93.96% _(0.9396)_ accuracy rate
2) Using the accuracies from Question 1, calculate a 95% confidence interval around the accuracy on each data set.
   * **monks1.csv**:
     * With accuracy: 73.14%
       * [0.4429955609873762, 0.6310785130866979]
   * **penguins.csv**:
     * With accuracy: 91.86% 
       * [0.8608122205639201, 0.9763970817616612]
   * **mnist_100.csv**:
     * With accuracy: 85.6%
       * [0.8124784659093914, 0.8995215340906085]
   * **mnist_1000.csv** : 
     * With accuracy: 93.96% 
       * [0.9302615232817337, 0.9489384767182663]
3) How did your accuracy compare between the mnist_100 and mnist_1000 data sets? Which had the higher average? Why do you think you observed this result? Did their confidence intervals overlap? What conclusion can we draw based on their confidence intervals?
    - I found the accuracy of mnist_1000 was larger than mnist_100 by 8%. I think that came with having more data in this situation resulted in better results due to how we are searching for the smallest distance to help with our predictions. From this, we got more values that allowed us to find answers that are closer to the one we are searching for and could have helped those outlier cases. Alongside this, I found that the confidence intervals don't overlap, resulting in the performance of mnist_1000.csv is greater than mnist_100. In other words, there is a  statistically significant difference in the performances of _mnist_100_ and _mnist_1000_ (at the 1-95%= 0.05 level)
Since the two confidence intervals do not overlap, we conclude that the second algorithm statistically
significantly outperformed the first (at a 0.05 level).

4) Pick one data set and three different values of k (document both in your README). Run the program with each value of k on that data set and compare the accuracy values observed. Did changing the value of k have much of an effect on your results? Speculate as to why or why not that observation occurred?
   - Data Set Chosen: **monks1.csv**
   - Training Set Percent: 75% _(0.75)_
   - Random Seed: 12
     - K Value: 
       - 3
         - Accuracy Rate: 85.19% _(0.8518518518518519)_
       - 30
         - Accuracy Rate: 76.85% _(0.7685185185185185)_
       - 100
         - Accuracy Rate: 53.70% _(0.5370370370370371)_

   - This makes sense to me because we are searching for the nearest neighbor to our value. From this, once we find the closest values, it becomes a find the the highest value from our k values. From this, if we have more neighbors, it extends our range of distance which can result in the answer to be less accurate. 
   From this, if there are more results that are the opposite of the one we want, we will predict the wrong value due to the majority of values are of the wrong label compared to the ones that are more closely similar to the test instance we are evaluating. 

5) (BONUS) Pick 10 different random seeds (document them in your README file) and rerun k- Nearest Neighbors with a k = 1 on the penguins.csv data. Record the average of the accuracy across the 10 runs.
Next, rerun the program on the same 10 seeds but only consider two attributes at a time (ignoring the other two attributes not in the chosen pair). Record the average accuracy for each pair of attributes across the 10 seeds. Since there are four attributes, there are six possible pairs of attributes (e.g., bill_length_mm-bill_depth_mm is one pair, so flipper_length_mm and body_mass_g would be ignored for this pair).
Finally, compare the average accuracy results between (1-6) all six pairs of attributes and (7) the results using all four attributes. Did any pairs of attributes do as well (or better) than learning using all four attributes? Speculate why you observed your results.

**Arguments Used**
- k_value:  1
- Training Percent: 0.75

- Random Seeds: 
  - 1:
    - Accuracy: 81.39% (0.813953488372093)
    - Accuracy with pair (bill_length_mm, bill_depth_mm): 93.0% (0.9302325581395349)
    - Accuracy with pair (bill_depth_mm,flipper_length_mm): 75.58% (0.7558139534883721)
    - Accuracy with pair (flipper_length_mm,body_mass_g): 68.60% (0.686046511627907)
    - Accuracy with pair (bill_length_mm,body_mass_g): 68.60% (0.686046511627907)
    - Accuracy with pair (bill_length_mm,flipper_length_mm): 95.34% (0.9534883720930233)
    - Accuracy with pair (bill_depth_mm, body_mass_g): 68.60% (0.686046511627907)
  - 2:
    - Accuracy: 87.39% (0.872093023255814)
    - Accuracy with pair (bill_length_mm, bill_depth_mm): 89.53% (0.8953488372093024)
    - Accuracy with pair (bill_depth_mm,flipper_length_mm): 73.25% (0.7325581395348837)
    - Accuracy with pair (flipper_length_mm,body_mass_g): 69.76% (0.6976744186046512)
    - Accuracy with pair (bill_length_mm,body_mass_g): 68.60% (0.686046511627907)
    - Accuracy with pair (bill_length_mm,flipper_length_mm): 93.02% (0.9302325581395349)
    - Accuracy with pair (bill_depth_mm, body_mass_g): 68.60% (0.686046511627907)
  - 3:
    - Accuracy: 80.23% (0.8023255813953488)
    - Accuracy with pair (bill_length_mm, bill_depth_mm): 94.18% (0.9418604651162791)
    - Accuracy with pair (bill_depth_mm,flipper_length_mm): 72.09% (0.7209302325581395)
    - Accuracy with pair (flipper_length_mm,body_mass_g): 66.27% (0.6627906976744186)
    - Accuracy with pair (bill_length_mm,body_mass_g): 73.25%(0.7325581395348837)
    - Accuracy with pair (bill_length_mm,flipper_length_mm): 95.34% (0.9534883720930233)
    - Accuracy with pair (bill_depth_mm, body_mass_g): 73.25% (0.7325581395348837)
  - 4:
    - Accuracy: 82.55% (0.8255813953488372)
    - Accuracy with pair (bill_length_mm, bill_depth_mm): 94.18% (0.9418604651162791)
    - Accuracy with pair (bill_depth_mm,flipper_length_mm): 72.09% (0.7209302325581395)
    - Accuracy with pair (flipper_length_mm,body_mass_g): 72.09% (0.7209302325581395)
    - Accuracy with pair (bill_length_mm,body_mass_g): 73.25% (0.7325581395348837)
    - Accuracy with pair (bill_length_mm,flipper_length_mm): 90.69% (0.9069767441860465)
    - Accuracy with pair (bill_depth_mm, body_mass_g): 73.25% (0.7325581395348837)
  - 5:
    - Accuracy: 84.88% (0.8488372093023255)
    - Accuracy with pair (bill_length_mm, bill_depth_mm): 96.51% (0.9651162790697675)
    - Accuracy with pair (bill_depth_mm,flipper_length_mm): 72.09% (0.7209302325581395)
    - Accuracy with pair (flipper_length_mm,body_mass_g): 69.76% (0.6976744186046512)
    - Accuracy with pair (bill_length_mm,body_mass_g): 74.41% (0.7441860465116279)
    - Accuracy with pair (bill_length_mm,flipper_length_mm): 93.02% (0.9302325581395349)
    - Accuracy with pair (bill_depth_mm, body_mass_g): 74.41% (0.7441860465116279)
  - 6:
    - Accuracy: 86.04% (0.8604651162790697)
    - Accuracy with pair (bill_length_mm, bill_depth_mm): 93.02% (0.9302325581395349)
    - Accuracy with pair (bill_depth_mm,flipper_length_mm): 72.09% (0.7209302325581395)
    - Accuracy with pair (flipper_length_mm,body_mass_g): 68.60% (0.686046511627907)
    - Accuracy with pair (bill_length_mm,body_mass_g): 69.76% (0.6976744186046512)
    - Accuracy with pair (bill_length_mm,flipper_length_mm): 94.18% (0.9418604651162791)
    - Accuracy with pair (bill_depth_mm, body_mass_g): 69.76% (0.6976744186046512)
  - 7:
    - Accuracy: 87.20% (0.872093023255814)
    - Accuracy with pair (bill_length_mm, bill_depth_mm): 95.34% (0.9534883720930233)
    - Accuracy with pair (bill_depth_mm,flipper_length_mm): 69.76% (0.6976744186046512)
    - Accuracy with pair (flipper_length_mm,body_mass_g): 69.76% (0.6976744186046512)
    - Accuracy with pair (bill_length_mm,body_mass_g): 63.95% (0.6395348837209303)
    - Accuracy with pair (bill_length_mm,flipper_length_mm): 91.86% (0.9186046511627907)
    - Accuracy with pair (bill_depth_mm, body_mass_g): 63.95% (0.6395348837209303)
  - 8:
    - Accuracy: 89.53% (0.8953488372093024)
    - Accuracy with pair (bill_length_mm, bill_depth_mm): 97.67% (0.9767441860465116)
    - Accuracy with pair (bill_depth_mm,flipper_length_mm): 76.74% (0.7674418604651163)
    - Accuracy with pair (flipper_length_mm,body_mass_g): 80.23% (0.8023255813953488)
    - Accuracy with pair (bill_length_mm,body_mass_g): 73.25% (0.7325581395348837)
    - Accuracy with pair (bill_length_mm,flipper_length_mm): 95.34% (0.9534883720930233)
    - Accuracy with pair (bill_depth_mm, body_mass_g): 73.25% (0.7325581395348837)
  - 9:
    - Accuracy: 77.90% (0.7790697674418605)
    - Accuracy with pair (bill_length_mm, bill_depth_mm): 93.02% (0.9302325581395349)
    - Accuracy with pair (bill_depth_mm,flipper_length_mm): 75.58% (0.7558139534883721)
    - Accuracy with pair (flipper_length_mm,body_mass_g): 68.60% (0.686046511627907)
    - Accuracy with pair (bill_length_mm,body_mass_g): 68.60% (0.686046511627907)
    - Accuracy with pair (bill_length_mm,flipper_length_mm): 96.51% (0.9651162790697675)
    - Accuracy with pair (bill_depth_mm, body_mass_g): 68.60% (0.686046511627907)
  - 10:
    - Accuracy: 81.39% (0.813953488372093)
    - Accuracy with pair (bill_length_mm, bill_depth_mm): 93.02% (0.9302325581395349)
    - Accuracy with pair (bill_depth_mm,flipper_length_mm): 70.93% (0.7093023255813954)
    - Accuracy with pair (flipper_length_mm,body_mass_g): 69.76% (0.6976744186046512)
    - Accuracy with pair (bill_length_mm,body_mass_g): 67.44% (0.6744186046511628)
    - Accuracy with pair (bill_length_mm,flipper_length_mm): 96.51% (0.9651162790697675)
    - Accuracy with pair (bill_depth_mm, body_mass_g): 67.44% (0.6744186046511628)

I found that there were some attributes that performed better than other. The pairs of attributes that performed better than when 
using all 4 the attributes were:
- (bill_length_mm,flipper_length_mm)
- (bill_length_mm, bill_depth_mm)

I believe these pair of attributes performed better as these are key features for distinguishing penguins. I can see that compared to body mass which is more individualized,
I can see that the key things that define certain species of ducks could be generalized and pinpointed by the length of their ill and their tail. 
Overall, these let the algorithm perform better by clearing up noise in the dataset to pinpoint the key features that can help with the classification. 

## 2) A short paragraph describing your experience during the assignment (what did you enjoy,what was difficult, etc.)
I really enjoyed this assignment. I enjoyed the process of reading in the csv file and understanding the format that the files came in, how to optimizer the code to make it more efficient, and also the overall analysis of the algorithm. I found finding ways to make my code faster difficult, but I know once we learn more things about how to optimize code in python, I'll come back to this assignment and revise it to the best of my ability.

## 3) An estimation of how much time you spent on the assignment
I spent about 20 hours on this assignment.

## 4) Honor Code
I have adhered to the honor code on this assignment.
- Sagana Ondande
