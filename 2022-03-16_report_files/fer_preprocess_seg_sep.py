import os
import argparse
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
from skimage import measure
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

    padding = 8 # Change this from results
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


def calculate_initial_bounds(mask):
    left = 0
    right = 0
    top = 0
    bottom = 0
    is_first_value = True
    # top -> bottom
    for idx_m, row in enumerate(mask):
        if row.any() == 1.0:
            if is_first_value:
                top = idx_m
                is_first_value = False
            else:
                bottom = idx_m
            continue  # skip when encountering the first mask value (pixel with value > 0.5)

    is_first_value = True
    # left -> right
    for idx_m, col in enumerate(mask.T):
        if col.any() == 1.0:
            if is_first_value:
                left = idx_m
                is_first_value = False
            else:
                right = idx_m
            continue  # skip when encountering the first mask value (pixel with value 1.0)

    return top, bottom, left, right


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


def cleanup_mask(thresh_mask):
    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh_mask, background=0)
    mask = np.zeros(thresh_mask.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh_mask.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 100:
            mask = cv2.add(mask, labelMask)

    return mask


if __name__ == '__main__':
    start = timer()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-emotion', default='neutral', type=str)
    args, _ = parser.parse_known_args()

    DEVICE = 'cuda'
    if not torch.cuda.is_available():
        DEVICE = 'cpu'
    EMOTION = args.emotion

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

    hdf5_path = f'../data/fer_64_segmen_masks_{EMOTION}.hdf5'
    f = h5py.File(hdf5_path, mode='w')
    d_train_masks_roi = f.create_dataset("train_masks_roi", shape=(len(dataset_train), 2, 64, 64), maxshape=(None, 2, 64, 64), dtype=np.float32)
    d_test_masks_roi = f.create_dataset("test_masks_roi", shape=(len(dataset_test), 2, 64, 64), maxshape=(None, 2, 64, 64), dtype=np.float32)
    d_train_imgs = f.create_dataset("train_imgs", shape=(len(dataset_train), 64, 64), maxshape=(None, 64, 64), dtype=np.float32)
    d_test_imgs = f.create_dataset("test_imgs", shape=(len(dataset_test), 64, 64), maxshape=(None, 64, 64), dtype=np.float32)
    d_train_masks_roi_pos = f.create_dataset("train_masks_roi_pos", shape=(len(dataset_train), 2, 4), maxshape=(None, 2, 4), dtype=np.uint8)
    d_test_masks_roi_pos = f.create_dataset("test_masks_roi_pos", shape=(len(dataset_test), 2, 4), maxshape=(None, 2, 4), dtype=np.uint8)
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

        eye_eyebrow_masks = torch.cat([masks[0].unsqueeze(dim=0), masks[1].unsqueeze(dim=0)], dim=0)
        eye_eyebrow_mask = torch.sum(eye_eyebrow_masks, dim=0)
        eye_eyebrow_mask = np.array(eye_eyebrow_mask > 0.95).astype(np.int32)
        eye_eyebrow_mask = torch.FloatTensor(eye_eyebrow_mask).squeeze()

        mouth_lips_masks = torch.cat([masks[2].unsqueeze(dim=0), masks[3].unsqueeze(dim=0)], dim=0)
        mouth_lips_mask = torch.sum(mouth_lips_masks, dim=0)
        mouth_lips_mask = np.array(mouth_lips_mask > 0.95).astype(np.int32)
        mouth_lips_mask = cleanup_mask(thresh_mask=mouth_lips_mask)
        mouth_lips_mask = torch.FloatTensor(mouth_lips_mask).squeeze()

        e_top, e_bottom, e_left, e_right = calculate_initial_bounds(mask=eye_eyebrow_mask)
        m_top, m_bottom, m_left, m_right = calculate_initial_bounds(mask=mouth_lips_mask)

        e_pad_top, e_pad_bottom, e_pad_left, e_pad_right = calculate_square_padding(e_top, e_bottom, e_left, e_right)
        m_pad_top, m_pad_bottom, m_pad_left, m_pad_right = calculate_square_padding(m_top, m_bottom, m_left, m_right)

        e_width = e_pad_bottom - e_pad_top
        e_height = e_pad_right - e_pad_left
        m_width = m_pad_bottom - m_pad_top
        m_height = m_pad_right - m_pad_left

        img = x.cpu().squeeze()
        e_roi_img = img[e_pad_top:e_pad_bottom, e_pad_left:e_pad_right]
        m_roi_img = img[m_pad_top:m_pad_bottom, m_pad_left:m_pad_right]

        resized_e_roi = nnf.interpolate(e_roi_img.view(1, 1, e_roi_img.size(0), e_roi_img.size(1)), size=(64, 64), mode='bicubic')
        resized_m_roi = nnf.interpolate(m_roi_img.view(1, 1, m_roi_img.size(0), m_roi_img.size(1)), size=(64, 64), mode='bicubic')

        masks_roi = np.concatenate((resized_e_roi.squeeze(dim=0), resized_m_roi.squeeze(dim=0)), axis=0, dtype=np.float32)
        e_roi_pos = np.array([[e_pad_top, e_pad_bottom, e_pad_left, e_pad_right]]).astype(np.uint8)
        m_roi_pos = np.array([[m_pad_top, m_pad_bottom, m_pad_left, m_pad_right]]).astype(np.uint8)
        masks_roi_pos = np.concatenate((e_roi_pos, m_roi_pos), axis=0, dtype=np.uint8)
        img = nnf.interpolate(img.view(1, 1, img.size(0), img.size(1)), size=(64, 64), mode='bicubic')
        img = img.numpy().squeeze()

        f["train_masks_roi"][idx] = masks_roi
        f["train_imgs"][idx] = img
        f["train_masks_roi_pos"][idx] = masks_roi_pos
        idx += 1

    d_train_masks_roi.resize((idx, 2, 64, 64))
    d_train_imgs.resize((idx, 64, 64))
    d_train_masks_roi_pos.resize((idx, 2, 4))

    idx = 0
    for x, _ in tqdm(dataloader_test):
        x = x.to(DEVICE)
        with torch.no_grad():
            y_prim = segmentator.forward(x)
        y_prim = y_prim.cpu()
        # 0:eyes, 1:eyebrows, 2:mouth, 3:lips?
        masks = y_prim[0, :-1, :, :]

        masks_exist = check_masks_exist(masks.numpy())
        if not masks_exist:
            continue

        eye_eyebrow_masks = torch.cat([masks[0].unsqueeze(dim=0), masks[1].unsqueeze(dim=0)], dim=0)
        eye_eyebrow_mask = torch.sum(eye_eyebrow_masks, dim=0)
        eye_eyebrow_mask = np.array(eye_eyebrow_mask > 0.95).astype(np.int32)
        eye_eyebrow_mask = torch.FloatTensor(eye_eyebrow_mask).squeeze()

        mouth_lips_masks = torch.cat([masks[2].unsqueeze(dim=0), masks[3].unsqueeze(dim=0)], dim=0)
        mouth_lips_mask = torch.sum(mouth_lips_masks, dim=0)
        mouth_lips_mask = np.array(mouth_lips_mask > 0.95).astype(np.int32)
        mouth_lips_mask = cleanup_mask(thresh_mask=mouth_lips_mask)
        mouth_lips_mask = torch.FloatTensor(mouth_lips_mask).squeeze()

        e_top, e_bottom, e_left, e_right = calculate_initial_bounds(mask=eye_eyebrow_mask)
        m_top, m_bottom, m_left, m_right = calculate_initial_bounds(mask=mouth_lips_mask)

        e_pad_top, e_pad_bottom, e_pad_left, e_pad_right = calculate_square_padding(e_top, e_bottom, e_left, e_right)
        m_pad_top, m_pad_bottom, m_pad_left, m_pad_right = calculate_square_padding(m_top, m_bottom, m_left, m_right)

        e_width = e_pad_bottom - e_pad_top
        e_height = e_pad_right - e_pad_left
        m_width = m_pad_bottom - m_pad_top
        m_height = m_pad_right - m_pad_left

        img = x.cpu().squeeze()
        e_roi_img = img[e_pad_top:e_pad_bottom, e_pad_left:e_pad_right]
        m_roi_img = img[m_pad_top:m_pad_bottom, m_pad_left:m_pad_right]

        resized_e_roi = nnf.interpolate(e_roi_img.view(1, 1, e_roi_img.size(0), e_roi_img.size(1)), size=(64, 64),
                                        mode='bicubic')
        resized_m_roi = nnf.interpolate(m_roi_img.view(1, 1, m_roi_img.size(0), m_roi_img.size(1)), size=(64, 64),
                                        mode='bicubic')

        masks_roi = np.concatenate((resized_e_roi.squeeze(dim=0), resized_m_roi.squeeze(dim=0)), axis=0,
                                   dtype=np.float32)
        e_roi_pos = np.array([[e_pad_top, e_pad_bottom, e_pad_left, e_pad_right]]).astype(np.uint8)
        m_roi_pos = np.array([[m_pad_top, m_pad_bottom, m_pad_left, m_pad_right]]).astype(np.uint8)
        masks_roi_pos = np.concatenate((e_roi_pos, m_roi_pos), axis=0, dtype=np.uint8)
        img = nnf.interpolate(img.view(1, 1, img.size(0), img.size(1)), size=(64, 64), mode='bicubic')
        img = img.numpy().squeeze()

        f["test_masks_roi"][idx] = masks_roi
        f["test_imgs"][idx] = img
        f["test_masks_roi_pos"][idx] = masks_roi_pos
        idx += 1

    d_test_masks_roi.resize((idx, 2, 64, 64))
    d_test_imgs.resize((idx, 64, 64))
    d_test_masks_roi_pos.resize((idx, 2, 4))

    f.close()
    end = timer()
    print(f'Elapsed time: {end - start}')