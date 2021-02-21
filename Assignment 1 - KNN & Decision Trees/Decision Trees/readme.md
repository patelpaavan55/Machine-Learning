Problem 2: Decision Tree (50 points)
====================================

Remember from lecture, we learned that we can use decision tree to solve classification and regression problem. Mostly we focus on classification. In problem 1 we used KNN to do classification. We could also use decision tree algorithm to do the same job. For Decision Tree, you will be asked to implement ID3 algorithm. It's guaranteed that all features are discrete in this assignment. **After finishing the implementation, you can use dt\_test\_script.py to test the correctness of your functions.**

Grading Guideline
----------------

1.  Information\_Gain function - 1\*10 = 10 points we will test your Infomation Gain function on ten sets of inputs. To receive full credits of this part, your outputs should be within a difference of 1e-10 with ours.
2.  Train your decision tree - 3\*5 = 15 points we will test your decision tree on five datasets. To receive full credit of this part, your algorithm should generate the identical decision tree as ours on each dataset.
3.  Prediction of decision tree - 2\*5 = 10 points we will test your predict function after you build the tree on (2). To receive full credit of this part, your algorithm should have right prediction.
4.  Pruning of decision tree - 3\*5 = 15 points we will test your reduceErrorPruning function on 5 datasets. To receive full credit of this part, your algorithm should generate the identical decision tree as ours on each dataset.

What to submit
--------------

1.  hw1\_dt.py

1.  utils.py

Part 2.1 Implementation
-----------------------

### 2.1.1 Information Gain (10 points)

-   In ID3 algorithm, we use Entropy to measure the uncertainty in the data set. We use Information Gain to measure the quality of a split.
-   Entropy:
-   Information\_Gain:
-   see more detail on [ID3 Algorithm](https://urldefense.proofpoint.com/v2/url?u=https-3A__en.wikipedia.org_wiki_ID3-5Falgorithm&d=DwMFaQ&c=clK7kQUTWtAVEOVIgvi0NU5BOUHhpN0H8p7CSfnc_gI&r=yYXh4HckzYFWw0_zu9Nv6g&m=jSeQoESPLgNl7UYVyluLogJjjceJYUpB1D-AkAVjrc0&s=IoqAGixMKM_zbRP7nTMwEnkZAniPWAzq6KM9c7931q8&e=) In this section, you need to implement Information\_Gain function on utils.py.

``` {.md-fences .md-end-block .ty-contain-cm .modeLoaded spellcheck="false" lang=""}
x
```

``` {.CodeMirror-line role="presentation"}
def Information_Gain(S, branches):
```

``` {.CodeMirror-line role="presentation"}
# calculate information_gain according to branches seperated by one feature
```

``` {.CodeMirror-line role="presentation"}
# input:
```

``` {.CodeMirror-line role="presentation"}
    -S: float Entropy of current state
```

``` {.CodeMirror-line role="presentation"}
    -branches: List[List[int]] num_attribute_values*num_classes : Count of data points belonging to each attribute and class. 
```

``` {.CodeMirror-line role="presentation"}
    e.g. [[1,2],[1,1][2,0]] represents there are three attribute values. For attribute 0, there is one data point that belongs to class 0 and two data points that belong to class 1.
```

``` {.CodeMirror-line role="presentation"}
# return: float
```

### 2.1.2 Grow decision Tree and predict (25 points)

-   In ID3 algorithm, we use the largest information gain to split the set S. Please consult the Lecture 2 notes page 23.

-   Implement TreeNode split function and TreeNode predict function in hw1\_dt.py:

    -   TreeNode.split

        -   In TreeNode class, variable features represents all data points in current TreeNode. Variable labels represents corresponding labels for all data points. Variable children is a list of TreeNode after split the current node on best attribute. This should be a recursive process that once we call the split function, the TreeNode will keep spliting until we get the whole decision tree.
        -   **When there is a tie of information gain on comparing all attributes of current TreeNode, always choose the attribute which has more attribute values. If they have same number of attribute values, use the one with small index.**
        -   **Build child list of each TreeNode with increasing order on attribute values, the order matters because it will be used when comparing two decision trees.**

    -   TreeNode.predict

        -   This function will be called once you have got the decision tree. It will take one single data point as a parameter, your code should process that data point and go through your tree to a leaf node and make prediction. This function should return a predicted lable.

-   You don't need to implement Decision Tree predict and train function in hw1\_dt.py. We will provide these two function in the statercode. Reading the train and predict function would help you understand functions that you need to implement.

    -   DecisionTree.train
    -   DecisionTree.predict
-   This is the comarision logic we will use to check your decision tree.

    ``` {.md-fences .md-end-block .ty-contain-cm .modeLoaded spellcheck="false" lang=""}
    x
    ```

    ``` {.CodeMirror-line role="presentation"}
    def compareDecisionTree(dTree_node, my_dTree_node):
    ```

    ``` {.CodeMirror-line role="presentation"}
        if dTree_node.splittable != my_dTree_node.splittable:
    ```

    ``` {.CodeMirror-line role="presentation"}
            return False
    ```

    ``` {.CodeMirror-line role="presentation"}
        if dTree_node.cls_max != my_dTree_node.cls_max:
    ```

    ``` {.CodeMirror-line role="presentation"}
            return False
    ```

    ``` {.CodeMirror-line role="presentation"}
        if dTree_node.splittable:
    ```

    ``` {.CodeMirror-line role="presentation"}
            if len(dTree_node.children) != len(my_dTree_node.children):
    ```

    ``` {.CodeMirror-line role="presentation"}
                return False
    ```

    ``` {.CodeMirror-line role="presentation"}
            for idx, child in enumerate(dTree_node.children):
    ```

    ``` {.CodeMirror-line role="presentation"}
                if not compareDecisionTree(child, my_dTree_node.children[idx]):
    ```

    ``` {.CodeMirror-line role="presentation"}
                    return False
    ```

    ``` {.CodeMirror-line role="presentation"}
        return True
    ```

Part 2.2 Pruning Decision Tree (15 points)
------------------------------------------

Sometimes, in order to prevent overfitting. We need to prune our Decision Tree. There are several approaches to avoid overfitting in building decision trees.

-   Pre-pruning: stop growing the tree earlier, before it perfectly classifies the training set.
-   Post-pruning: grows decision tree completely on training set, and then post prune decision tree.

Practically, the second approach post-pruning is more useful, also it is not easy to precisely estimate when to stop growing the tree by using pre-pruning. We will use Reduced Error Pruning, as one of Post-pruning in this assignment.

``` {.md-fences .md-end-block .ty-contain-cm .modeLoaded spellcheck="false" lang=""}
x
```

``` {.CodeMirror-line role="presentation"}
Reduced Error Pruning
```

``` {.CodeMirror-line role="presentation"}
0. Split data into training and validation sets.
```

``` {.CodeMirror-line role="presentation"}
1. Grow the decision tree until further pruning is harmful:
```

``` {.CodeMirror-line role="presentation"}
2. Evaluate decision tree on validation dataset.
```

``` {.CodeMirror-line role="presentation"}
3. Greedily remove TreeNode (from leaf to root) that most improves prediction accuracy on validation set.
```

For Pruning of Decision Tree, you can refer [Reduce Error Pruning](https://urldefense.proofpoint.com/v2/url?u=http-3A__jmvidal.cse.sc.edu_talks_decisiontrees_reducederrorprun.html-3Fstyle-3DWhite&d=DwMFaQ&c=clK7kQUTWtAVEOVIgvi0NU5BOUHhpN0H8p7CSfnc_gI&r=yYXh4HckzYFWw0_zu9Nv6g&m=jSeQoESPLgNl7UYVyluLogJjjceJYUpB1D-AkAVjrc0&s=m8oZfwPt7lK7dHjzo71WBzdiYIu22fIBn7h4KHwRvF4&e=) and P69 of Textbook: Machine Learning -Tom Mitchell.

### 2.2.1 Reduce Error Pruning

**Hint: in this part, you can add other parameters or functions in TreeNode class and DecisionTree class to help you in pruning the decision tree. But your changes should not influence results of previous parts.** Implement the reduced\_error\_pruning function in util.py.

``` {.md-fences .md-end-block .ty-contain-cm .modeLoaded spellcheck="false" lang=""}
x
```

``` {.CodeMirror-line role="presentation"}
def reduced_error_pruning(decitionTree):
```

``` {.CodeMirror-line role="presentation"}
# input: 
```

``` {.CodeMirror-line role="presentation"}
    - decitionTree: decitionTree trained based on training data set.
```

``` {.CodeMirror-line role="presentation"}
    - X_test: List[List[any]] test data, num_cases*num_attributes
```

``` {.CodeMirror-line role="presentation"}
    - y_test: List[any] test labels, num_cases*1
```

Good Luck! :)
