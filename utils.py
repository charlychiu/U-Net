import pickle


def save_variable_to_pickle(variable, name, path='./'):
    if name is not None:
        file = open(path + name + '.pickle', 'wb')
        pickle.dump(variable, file)
        file.close()
        print('Save to {} successfully'.format(str(name + '.pickle')))
    else:
        print('Error arg')
