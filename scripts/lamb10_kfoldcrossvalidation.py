import matplotlib.pyplot as plt

from datasets.default_trainingsets import get_10lamb_6patches


if __name__ == '__main__':
    k_fold_train_data = get_10lamb_6patches(5)

    plt.figure()
    for i in range(6):
        train_data_i = k_fold_train_data.k_split_i(i)

        x = train_data_i.get_x_train()
        y_tr = train_data_i.get_y_train()

        x_te = train_data_i.get_x_test()
        y_te = train_data_i.get_y_test()
        
        plt.subplot(3, 2, i+1)
        plt.imshow(y_tr[0, ..., 1])
        
    plt.show()
