import argparse
import time
import os
from timeit import default_timer as timer

from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import torch
import torchvision
import torchvision.utils as vutils
import shutil

from pretrained_classification import Model


if __name__ == '__main__':
    start = timer()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-emotion', default='happiness', type=str)
    args, _ = parser.parse_known_args()

    EMOTION = args.emotion

    # TODO remove
    # RUN_PATH = f'{int(time.time())}_output'
    # if os.path.exists(RUN_PATH):
    #     shutil.rmtree(RUN_PATH)
    # os.makedirs(RUN_PATH)

    # Setup pretrained model
    # Model class outputs: [neutral, happiness, sadness, anger]
    model = Model(CLASS_NUM=4, in_image_channels=1, pretrained=False)
    model.load_state_dict(torch.load('./classificator_models/model-best_4classes.pt', map_location=torch.device('cpu')))
    model.eval()

    classifier_emo_dict = {
        'neutral': 0,
        'happiness': 1,
        'sadness': 2,
        'anger': 3
    }
    classifier_emo_label = classifier_emo_dict[EMOTION]

    ### Prepare data to be stored
    fer_emo_dict = {
        'anger': 0,
        'disgust': 1,
        'fear': 2,
        'happiness': 3,
        'sadness': 4,
        'surprise': 5,
        'neutral': 6
    }
    fer_emo_label = fer_emo_dict[EMOTION]
    print(os.getcwd())
    # Emotions [0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral]
    df = pd.read_csv('./data/fer/fer2013/fer2013.csv', sep=',', header=None)
    df = df.loc[df[0] == str(fer_emo_label)]  # filter out entries for label

    # TODO remove
    # for idx, val in enumerate(df.values):
    #     img_flat = np.fromstring(val[1], dtype=int, sep=' ')
    #     img_np = np.reshape(img_flat, (48, 48))
    #     img_t = torch.FloatTensor(img_np)
    #     vutils.save_image(img_t,
    #                       os.path.join(RUN_PATH, 'samples_{}.png'.format(idx)),
    #                       nrow=1,
    #                       normalize=True)

    # Anomalies for emotions in fer dataset
    anomaly_idxs = {
        'happiness': [7777, 7388, 6441, 6270, 5956, 5890, 5691, 5672, 5453, 5160, 4752,
                      4391, 4123, 3994, 3964, 3834, 3754, 3613, 3314, 3040, 2864, 2819,
                      2647, 2541, 2458, 2250, 2226, 1933, 1817, 1691, 1296, 1091],
        'neutral': [77, 135, 213, 219, 261, 363, 589, 825, 1011, 1058, 1095, 1207, 1760,
                    1774, 2062, 2075, 2156, 2256, 2371, 2458, 2503, 2512, 2545, 2657, 2732,
                    2751, 2997, 3022, 3079, 3118, 3253, 3274, 3392, 3639, 3672, 3767, 3806,
                    3941, 3977, 4074, 4105, 4186, 4268, 4277, 4326, 4352, 4369, 4473, 4606,
                    4660, 4849, 4859, 4879, 5020, 5034, 5037, 5100, 5188, 5210, 5289, 5383,
                    5491, 5532, 5652, 5708, 5723, 5726, 5751, 5811, 5942, 5962, 6123]
    }

    # # TODO remove
    # # Remove anomaly indexes
    # # anomalies = [1091]
    # batch = []
    # for i in range(10):
    #     img_flat = np.fromstring(df.values[150+i][1], dtype=int, sep=' ')
    #     img_np = np.reshape(img_flat, (1, 48, 48))
    #     img_t = torch.FloatTensor(img_np)
    #     batch.append(img_t)
    # batch = torch.vstack(batch)
    #
    # y_prim = model.forward(batch.unsqueeze(dim=1)).squeeze(dim=0)
    # # loss = -torch.log(y_prim.squeeze(dim=0))
    # y_idx = torch.argmax(y_prim, dim=1)
    # # y_idx = torch.tensor([1, 0, 2, 3, 1, 0])
    # acc = torch.mean((y_idx == 1) * 1.0)

    # # if neutral emotion loss is close to chosen then let it pass (absolute value in bounds of 5.0)
    # prc = y_prim[classifier_emo_label]

    # obtain valid sample indexes using classifier
    train_idxs = []
    test_idxs = []
    data_len = len(df.values)
    iter_done = 0
    for idx, val in enumerate(df.values):
        img_flat = np.fromstring(val[1], dtype=int, sep=' ')
        img_np = np.reshape(img_flat, (48, 48))
        img_t = torch.FloatTensor(img_np)
        y_prim = model.forward(img_t.view(1, 1, img_t.size(0), img_t.size(1))).squeeze(dim=0)
        if EMOTION != 'neutral':  # Ignore neutral emotion
            y_real = classifier_emo_dict[EMOTION] - 1
            y_idx = torch.argmax(y_prim[1:])
        else:
            y_real = classifier_emo_dict[EMOTION]
            y_idx = torch.argmax(y_prim)

        if idx not in anomaly_idxs[EMOTION] and y_idx == y_real:
            if val[2] == 'Training':
                train_idxs.append(idx)
                iter_done += 1
            else:
                test_idxs.append(idx)
                iter_done += 1
            if iter_done % 20 == 0:
                print(f'Classifier iterations done: {iter_done}')

    # open a hdf5 file and create arrays
    if not os.path.exists('../data/'):
        os.makedirs('../data/')
    hdf5_path = f'./data/fer_48_{EMOTION}.hdf5'
    f = h5py.File(hdf5_path, mode='w')

    f.create_dataset("train_labels", (len(train_idxs),), np.uint8)
    f.create_dataset("test_labels", (len(test_idxs),), np.uint8)
    f.create_dataset("train_imgs", (len(train_idxs), 48, 48), np.float32)
    f.create_dataset("test_imgs", (len(test_idxs), 48, 48), np.float32)

    # Write train data
    for i, val_idx in enumerate(train_idxs):
        f["train_labels"][i] = int(df.values[val_idx][0])
        img_flat = np.fromstring(df.values[val_idx][1], dtype=int, sep=' ')
        img_np = np.reshape(img_flat, (48, 48))
        f["train_imgs"][i] = img_np

    # Write test data
    for i, val_idx in enumerate(test_idxs):
        f["test_labels"][i] = int(df.values[val_idx][0])
        img_flat = np.fromstring(df.values[val_idx][1], dtype=int, sep=' ')
        img_np = np.reshape(img_flat, (48, 48))
        f["test_imgs"][i] = img_np

    f.close()
    end = timer()
    print(f'Elapsed time: {end - start}')