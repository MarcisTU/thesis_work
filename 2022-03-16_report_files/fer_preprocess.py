import argparse
import time
import os

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import h5py
import torch
import torchvision
import torch.nn.functional as nnf
import torchvision.utils as vutils
import shutil

from pretrained_segmentator import Model as Segmentator
from pretrained_seg_classificator import Model as Classificator
# from pretrained_classification import Model


def check_masks_exist(masks: np.ndarray):
    masks_exist = False
    eye_mask = masks[0]
    eyebrow_mask = masks[1]
    lips_mask = masks[3]

    eye_mask_px_count = len(np.argwhere(eye_mask > 0.5))
    eyebrow_mask_px_count = len(np.argwhere(eyebrow_mask > 0.5))
    lips_mask_px_count = len(np.argwhere(lips_mask > 0.5))

    if eye_mask_px_count > 200 and eyebrow_mask_px_count > 200 and lips_mask_px_count > 200:
        masks_exist = True

    return masks_exist


def process_data(run_path, emotion, fer_emotion_label, is_soruce_emo):
    classifier_emo_dict = {
        'neutral': 0,
        'happiness': 1,
        'sadness': 2,
        'anger': 3
    }

    # Emotions [0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral]
    df = pd.read_csv('../data/fer2013.csv', sep=',', header=None)
    # df = pd.read_csv('../data/fer/fer2013/fer2013.csv', sep=',', header=None)
    df = df.loc[df[0] == str(fer_emotion_label)]  # filter out entries for label

    # obtain valid sample indexes using classifier and segmentator
    train_idxs = []
    test_idxs = []
    data_len = len(df.values)
    iter_done = 0
    print(f'{emotion} samples before processing: {data_len}')
    for idx, val in enumerate(df.values):
        img_flat = np.fromstring(val[1], dtype=int, sep=' ')
        img_np = np.reshape(img_flat, (48, 48))
        img_t = torch.FloatTensor(img_np)
        img = nnf.interpolate(img_t.view(1, 1, img_t.size(0), img_t.size(1)), size=(256, 256), mode='bicubic')

        mid = img.max() / 2
        img = (img - mid) / mid

        img = img.to(DEVICE)
        with torch.no_grad():
            y_prim = segmentator.forward(img)
        y_prim = y_prim.cpu()
        # 0:eyes, 1:eyebrows, 2:mouth, 3:lips
        masks = y_prim[0, :-1, :, :]

        # Filter bad mask samples
        masks_exist = check_masks_exist(masks.numpy())
        if not masks_exist:
            vutils.save_image(img_t,
                              os.path.join(run_path, f'bad_masks_{idx}.png'),
                              nrow=1,
                              normalize=True)
            continue

        # Filter target and other emotions from source
        if is_soruce_emo:
            with torch.no_grad():
                masks = masks.to(DEVICE)
                m_prim = classificator.forward(masks.unsqueeze(dim=0))
            m_idx = torch.argmax(m_prim, dim=1)
            if not m_idx == classifier_emo_dict[emotion]:
                vutils.save_image(img_t,
                                  os.path.join(run_path, f'target_emo_{idx}.png'),
                                  nrow=1,
                                  normalize=True)
                continue

        if val[2] == 'Training':
            train_idxs.append(idx)
            iter_done += 1
        else:
            test_idxs.append(idx)
            iter_done += 1
        if iter_done % 50 == 0:
            print(f'iterations done: {iter_done}')

    print(f'{emotion} samples left after processing: {len(train_idxs) + len(test_idxs)}')

    # # TODO remove
    # for idx in train_idxs + test_idxs:
    #     img_flat = np.fromstring(df.values[idx][1], dtype=int, sep=' ')
    #     img_np = np.reshape(img_flat, (48, 48))
    #     img_t = torch.FloatTensor(img_np)
    #     vutils.save_image(img_t,
    #                       os.path.join(RUN_PATH, f'samples_{idx}.png'),
    #                       nrow=1,
    #                       normalize=True)

    # open a hdf5 file and create arrays
    if not os.path.exists('../data/'):
        os.makedirs('../data/')
    hdf5_path = f'../data/fer_48_{emotion}.hdf5'
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-source_emotion', default='neutral', type=str)
    parser.add_argument('-target_emotion', default='happiness', type=str)
    args, _ = parser.parse_known_args()

    SOURCE_EMOTION = args.source_emotion
    TARGET_EMOTION = args.target_emotion
    DEVICE = 'cuda'
    if not torch.cuda.is_available():
        DEVICE = 'cpu'

    RUN_PATH_TARGET = f'{int(time.time())}_{TARGET_EMOTION}_output'
    if os.path.exists(RUN_PATH_TARGET):
        shutil.rmtree(RUN_PATH_TARGET)
    os.makedirs(RUN_PATH_TARGET)
    RUN_PATH_SOURCE = f'{int(time.time())}_{SOURCE_EMOTION}_output'
    if os.path.exists(RUN_PATH_SOURCE):
        shutil.rmtree(RUN_PATH_SOURCE)
    os.makedirs(RUN_PATH_SOURCE)

    # Setup pretrained model
    # Model class outputs: [neutral, happiness, sadness, anger]
    # model = Model(CLASS_NUM=4, in_image_channels=1, pretrained=False)
    # model.load_state_dict(torch.load('./classificator_models/model-best_4classes.pt', map_location=torch.device('cpu')))
    # model.eval()

    # Setup pretrained segmentator model
    segmentator = Segmentator(classes=5).to(DEVICE)
    segmentator.load_state_dict(torch.load('./seg-model-best.pt', map_location=torch.device(DEVICE)))
    segmentator.eval()
    # Setup pretrained segmentation classificator model
    classificator = Classificator(CLASS_NUM=4, in_image_channels=4).to(DEVICE)
    classificator.load_state_dict(torch.load('./model-best_4m_4c.pt', map_location=torch.device(DEVICE)))
    classificator.eval()

    classifier_emo_dict = {
        'neutral': 0,
        'happiness': 1,
        'sadness': 2,
        'anger': 3
    }
    # classifier_emo_label = classifier_emo_dict[EMOTION]

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
    fer_source_emo_label = fer_emo_dict[SOURCE_EMOTION]
    fer_target_emo_label = fer_emo_dict[TARGET_EMOTION]

    # Process soource emotion data
    process_data(run_path=RUN_PATH_SOURCE, emotion=SOURCE_EMOTION,
                 fer_emotion_label=fer_source_emo_label, is_soruce_emo=True)

    # Process target emotion data
    process_data(run_path=RUN_PATH_TARGET, emotion=TARGET_EMOTION,
                 fer_emotion_label=fer_target_emo_label, is_soruce_emo=False)
