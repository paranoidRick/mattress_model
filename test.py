import pickle

if __name__ == '__main__':
    resnet_acc = open('test_acc_lists/resnet_acc', 'rb')
    alex_acc = open('test_acc_lists/alex_acc', 'rb')
    vgg_acc = open('test_acc_lists/vgg_acc', 'rb')

    resnet_list = pickle.load(resnet_acc)
    alex_list = pickle.load(alex_acc)
    vgg_list = pickle.load(vgg_acc)

    print(resnet_list)
    print(alex_list)
    print(vgg_list)
