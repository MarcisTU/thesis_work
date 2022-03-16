import os
import argparse
import shutil
import time
from timeit import default_timer as timer

from tqdm import tqdm
import numpy as np
import h5py
import torch
import torchvision
import torch.nn.functional as nnf
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils import data
import cv2
import matplotlib.pyplot as plt

from pretrained_segmentator import Model as Segmentator
from pretrained_seg_classificator import Model as Classificator
from loader import Dataset


def calculate_square_padding(top, bottom, left, right):
    height = abs(top - bottom)
    width = abs(left - right)
    p_top = top
    p_bottom = bottom
    p_left = left
    p_right = right

    diff = abs(height - width)

    if width > height: # adjust top and bottom
        if diff % 2 == 0:
            delta = diff // 2
            p_top -= delta
            p_bottom += delta
        else:  # not even number
            delta = (diff - 1) // 2
            p_top -= delta
            p_bottom += delta
            p_right -= 1
    elif height > width:
        if diff % 2 == 0:
            delta = diff // 2
            p_left -= delta
            p_right += delta
        else:  # not even number
            delta = (diff - 1) // 2
            p_bottom -= 1
            p_right += delta
            p_left -= delta

    padding = 8  # Change this from results
    if p_top - padding < 0:
        p_top = 0
    else:
        p_top -= padding
    if p_bottom + padding > 255:
        p_bottom = 255
    else:
        p_bottom += padding
    if p_left - padding < 0:
        p_left = 0
    else:
        p_left -= padding
    if p_right + padding > 255:
        p_right = 255
    else:
        p_right += padding

    return p_top, p_bottom, p_left, p_right


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


if __name__ == '__main__':
    start = timer()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-emotion', default='happiness', type=str)
    args, _ = parser.parse_known_args()

    DEVICE = 'cuda'
    if not torch.cuda.is_available():
        DEVICE = 'cpu'
    EMOTION = args.emotion

    # RUN_PATH = f'{int(time.time())}_{EMOTION}_output'
    # if os.path.exists(RUN_PATH):
    #     shutil.rmtree(RUN_PATH)
    # os.makedirs(RUN_PATH)

    classifier_emo_dict = {
        'neutral': 0,
        'happiness': 1,
        'sadness': 2,
        'anger': 3
    }
    classifier_emo_label = classifier_emo_dict[EMOTION]

    # Setup pretrained segmentator model
    segmentator = Segmentator(classes=5).to(DEVICE)
    segmentator.load_state_dict(torch.load('./seg-model-best.pt', map_location=torch.device(DEVICE)))
    segmentator.eval()
    # Setup pretrained segmentation classificator model
    classificator = Classificator(CLASS_NUM=4, in_image_channels=4).to(DEVICE)
    classificator.load_state_dict(torch.load('./model-best_4m_4c.pt', map_location=torch.device(DEVICE)))
    classificator.eval()

    # Segmentator works only with pretrained image size which is (256, 256)
    transform = torchvision.transforms.Compose([
        transforms.Resize(256)
    ])
    # dataset = ImageDataset('../EmotionSegmantation-master/CelebAMask-HQ/CelebA-HQ-img/', transform=transform)
    loader_params = {'batch_size': 1, 'shuffle': True, 'num_workers': 4}
    dataset_train = Dataset(f'../data/fer_48_{EMOTION}.hdf5', mode='train', img_size=48, transform=transform)
    dataloader_train = data.DataLoader(dataset=dataset_train, **loader_params)
    loader_params = {'batch_size': 1, 'shuffle': True, 'num_workers': 4}
    dataset_test = Dataset(f'../data/fer_48_{EMOTION}.hdf5', mode='test', img_size=48, transform=transform)
    dataloader_test = data.DataLoader(dataset=dataset_test, **loader_params)

    hdf5_path = f'../data/fer_64_segmen_{EMOTION}.hdf5'
    f = h5py.File(hdf5_path, mode='w')
    d_train_roi = f.create_dataset("train_roi", shape=(len(dataset_train), 64, 64), maxshape=(None, 64, 64), dtype=np.float32)
    d_test_roi = f.create_dataset("test_roi", shape=(len(dataset_test), 64, 64), maxshape=(None, 64, 64), dtype=np.float32)
    d_train_imgs = f.create_dataset("train_imgs", shape=(len(dataset_train), 64, 64), maxshape=(None, 64, 64), dtype=np.float32)
    d_test_imgs = f.create_dataset("test_imgs", shape=(len(dataset_test), 64, 64), maxshape=(None, 64, 64), dtype=np.float32)
    d_train_roi_pos = f.create_dataset("train_roi_pos", shape=(len(dataset_train), 4), maxshape=(None, 4), dtype=np.uint8)
    d_test_roi_pos = f.create_dataset("test_roi_pos", shape=(len(dataset_test), 4), maxshape=(None, 4), dtype=np.uint8)
    # f.create_dataset("train_roi", shape=(len(dataset_train), 64, 64), dtype=np.float32)
    # f.create_dataset("test_roi", shape=(len(dataset_test), 64, 64), dtype=np.float32)
    # f.create_dataset("train_imgs", shape=(len(dataset_train), 64, 64), dtype=np.float32)
    # f.create_dataset("test_imgs", shape=(len(dataset_test), 64, 64), dtype=np.float32)
    # f.create_dataset("train_roi_pos", shape=(len(dataset_train), 4), dtype=np.uint8)
    # f.create_dataset("test_roi_pos", shape=(len(dataset_test), 4), dtype=np.uint8)

    idx = 0
    for x, _ in tqdm(dataloader_train):
        x = x.to(DEVICE)
        with torch.no_grad():
            y_prim = segmentator.forward(x)
        y_prim = y_prim.cpu()
        # 0:eyes, 1:eyebrows, 2:mouth, 3:lips?
        masks = y_prim[0, :-1, :, :]

        masks_exist = check_masks_exist(masks.numpy())
        if not masks_exist:
            continue

        # Classify masks
        # with torch.no_grad():
        #     masks = torch.FloatTensor(masks).unsqueeze(dim=0).to(DEVICE)
        #     m_prim = classificator.forward(masks)
        # m_idx = torch.argmax(m_prim, dim=1).cpu()

        # if m_idx == classifier_emo_dict[EMOTION]:
        masks = torch.sum(masks, dim=0)
        masks = np.array(masks > 0.5).astype(np.int32)
        masks = torch.FloatTensor(masks).squeeze()

        left = 0
        right = 0
        top = 0
        bottom = 0
        is_first_value = True
        # top -> bottom
        for idx_m, row in enumerate(masks):
            if row.any() > 0.5:
                if is_first_value:
                    top = idx_m
                    is_first_value = False
                else:
                    bottom = idx_m
                continue  # skip when encountering the first mask value (pixel with value > 0.5)

        is_first_value = True
        # left -> right
        for idx_m, col in enumerate(masks.T):
            if col.any() > 0.5:
                if is_first_value:
                    left = idx_m
                    is_first_value = False
                else:
                    right = idx_m
                continue  # skip when encountering the first mask value (pixel with value 1.0)

        # if abs(bottom - top) == 0 or abs(right - left) == 0:
        #     continue  # skip faulty sample
        top, bottom, left, right = calculate_square_padding(top, bottom, left, right)
        img = x.cpu().squeeze()
        roi_img = img[top:bottom, left:right]
        resized_roi = nnf.interpolate(roi_img.view(1, 1, roi_img.size(0), roi_img.size(1)), size=(64, 64), mode='bicubic')
        resized_roi = resized_roi.squeeze()

        img = nnf.interpolate(img.view(1, 1, img.size(0), img.size(1)), size=(64, 64), mode='bicubic')
        img = img.squeeze()

        resized_roi = resized_roi.numpy()
        img = img.numpy()
        f["train_roi"][idx] = resized_roi
        f["train_imgs"][idx] = img
        f["train_roi_pos"][idx] = np.array([top, bottom, left, right]).astype(np.uint8)
        idx += 1

    d_train_roi.resize((idx, 64, 64))
    d_train_imgs.resize((idx, 64, 64))
    d_train_roi_pos.resize((idx, 4))

    idx = 0
    for x, _ in tqdm(dataloader_test):
        x = x.to(DEVICE)
        with torch.no_grad():
            y_prim = segmentator.forward(x)
        y_prim = y_prim.cpu()
        masks = y_prim[0, :-1, :, :]

        masks_exist = check_masks_exist(masks.numpy())
        if not masks_exist:
            continue

        # Classify masks
        # with torch.no_grad():
        #     masks = torch.FloatTensor(masks).unsqueeze(dim=0).to(DEVICE)
        #     m_prim = classificator.forward(masks)
        # m_idx = torch.argmax(m_prim, dim=1).cpu()

        # if m_idx == classifier_emo_dict[EMOTION]:
        masks = torch.sum(masks, dim=0)
        masks = np.array(masks > 0.5).astype(np.int32)
        masks = torch.FloatTensor(masks).squeeze()

        left = 0
        right = 0
        top = 0
        bottom = 0
        is_first_value = True
        # top -> bottom
        for idx_m, row in enumerate(masks):
            if row.any() > 0.5:
                if is_first_value:
                    top = idx_m
                    is_first_value = False
                else:
                    bottom = idx_m
                continue  # skip when encountering the first mask value (pixel with value > 0.5)

        is_first_value = True
        # left -> right
        for idx_m, col in enumerate(masks.T):
            if col.any() > 0.5:
                if is_first_value:
                    left = idx_m
                    is_first_value = False
                else:
                    right = idx_m
                continue  # skip when encountering the first mask value (pixel with value 1.0)

        # if abs(bottom - top) == 0 or abs(right - left) == 0:
        #     continue  # skip faulty sample
        top, bottom, left, right = calculate_square_padding(top, bottom, left, right)
        img = x.cpu().squeeze()
        roi_img = img[top:bottom, left:right]
        resized_roi = nnf.interpolate(roi_img.view(1, 1, roi_img.size(0), roi_img.size(1)), size=(64, 64),
                                      mode='bicubic')
        resized_roi = resized_roi.squeeze()

        img = nnf.interpolate(img.view(1, 1, img.size(0), img.size(1)), size=(64, 64), mode='bicubic')
        img = img.squeeze()

        resized_roi = resized_roi.numpy()
        img = img.numpy()
        f["test_roi"][idx] = resized_roi
        f["test_imgs"][idx] = img
        f["test_roi_pos"][idx] = np.array([top, bottom, left, right]).astype(np.uint8)
        idx += 1

    d_test_roi.resize((idx, 64, 64))
    d_test_imgs.resize((idx, 64, 64))
    d_test_roi_pos.resize((idx, 4))

    f.close()
    end = timer()
    print(f'Elapsed time: {end - start}')