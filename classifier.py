import pickle

# Variable is loaded once, when this module is imported
dtree_iris = pickle.load(open('dtree_iris.pkl', 'rb'))

def predict(flower):
    '''
    Args: Flower as a list of floats, i.e. feature vector
    Returns: label in range (0,1,2)
    '''
    global dtree_iris
    label = 1
    # TODO implement prediction
    # label = dtree_iris.predict()

    return label


if __name__ == "__main__":
    print("predict() function was not implemented, will return 1")
    species = predict([0,1,2,3])
    print(species)