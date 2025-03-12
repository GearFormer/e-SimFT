
import pandas as pd

if __name__ == "__main__":

    data1 = pd.read_csv("train_models/dataset/val_data.csv", header=None)
    data2 = pd.read_csv("train_models/dataset/test_data.csv", header=None)

    data = pd.concat([data1, data2], ignore_index=True)
    data_size = len(data.index)
    shuffled_data = data.sample(frac=1).reset_index(drop=True)

    test_size = 184 #~2.5% of the datas
    simft_test_idx = test_size
    paret_test_idx = test_size * 2
    simft_train_idx = int(test_size * 2 + (data_size - test_size * 2)/2)

    simft_test = shuffled_data[:simft_test_idx]
    pareto_test = shuffled_data[simft_test_idx:paret_test_idx]
    sft_train = shuffled_data[paret_test_idx:simft_train_idx]
    pref_train = shuffled_data[simft_train_idx:]

    print(len(simft_test))
    print(len(pareto_test))
    print(len(sft_train))
    print(len(pref_train))

    print("storing files...")

    df = pd.DataFrame(simft_test)
    df.to_pickle("esimft_data/simft_test.pkl")
    df = pd.DataFrame(pareto_test)
    df.to_pickle("esimft_data/pareto_test.pkl")
    df = pd.DataFrame(sft_train)
    df.to_pickle("esimft_data/sft_train.pkl")
    df = pd.DataFrame(pref_train)
    df.to_pickle("esimft_data/pref_train.pkl")
