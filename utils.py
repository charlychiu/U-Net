def save_variable_to_pickle(variable, name, path='./'):
    import pickle
    if name is not None:
        file = open(path + name + '.pickle', 'wb')
        pickle.dump(variable, file)
        file.close()
        print('Save to {} successfully'.format(str(name + '.pickle')))
    else:
        print('Error arg')

def save_npy_array_to_picture(np_array, path='./data/'):
    from PIL import Image
    for idx, each_picture in enumerate(np_array):
        img = Image.fromarray(each_picture)
        img.mode = 'I'
        img.save(path + str(idx) + '.png')
        