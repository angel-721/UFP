# UFP
The Ultimate Fight Predictor- A fun way to predict the outcome of UFC fights.

### Warning(For now)
The current models were trained on scikit-learn 1.20 so predicting on later versions might not work due to pickle.
- model1.pkl is trained on the latest version of scikit-learn!

### Use 
Install requirements with <code>pip install -r requirements.txt</code>
Enter fighter data in the way specified by <code>./data/example.csv</code> and run main.py with the data you have as a command line argument and predict as the action. 

For example just run <code>python main.py</code> and it should use model1.pkl to predict everything in example.csv. 

![example picks](https://github.com/angel-721/UFP/assets/75283919/a33f609d-f6a7-40d5-a31c-237a62fd091b)

## Data for training
The model was trained on the [Ultimate UFC Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset?select=ufc-master.csv). If you want to train your own model with the ground work please download the data first.

### Background information
- model1.pkl is trained on a [SVM](https://scikit-learn.org/stable/modules/svm.html#svm-classification) using [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- The problem is a binary classification where the label is Winner which can be Blue or Red
  - The features are described in the dataset and include some aspects such as betting odds, wins, losses, and height
- All of the training is done on a 80/20(train/test) split with cross validation on 15 folds
