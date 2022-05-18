import pickle

# Variable is loaded once, when this module is imported
dtree_iris = pickle.load(open('dtree_iris.pkl', 'rb'))

def predict(flower):
    '''
    **TODO** docstring pattern
    Args: Flower as described as a JSON representing its feature vector {"<feature_name>: <value>"}
    Returns: label (0,1,2)
    '''
    global dtree_iris

    return 100