import argparse

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as nnf
import torchvision.utils as vutils
from torch.utils import data

from pretrained_classification import Model
from loader_all_feat import Dataset


def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.
    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


def insert_roi_to_image(img_batch, roi_batch, roi_pos_batch):
    img_out = []

    for idx, img in enumerate(img_batch):
        img = nnf.interpolate(img.unsqueeze(dim=0), size=(256, 256), mode='bicubic').squeeze()
        roi_pos = roi_pos_batch[idx].squeeze().type(torch.IntTensor)
        roi = roi_batch[idx]

        width = roi_pos[3] - roi_pos[2]
        height = roi_pos[1] - roi_pos[0]
        resized_m = nnf.interpolate(
            roi.unsqueeze(dim=0),
            size=(height, width), mode='bicubic'
        ).squeeze()

        # Normalize back to 0-255
        resized_m = (resized_m * 127.5) + 127.5
        img = (img * 127.5) + 127.5
        # Perform blending
        alpha_mask = resized_m[:, :] / 255.0
        img_result = img[:, :]
        y = roi_pos[0]
        x = roi_pos[2]
        overlay_image_alpha(img_result, resized_m, x, y, alpha_mask)

        # img[roi_pos[0]:roi_pos[1], roi_pos[2]:roi_pos[3]] = resized_m
        img_result = nnf.interpolate(
            img_result.view(1, 1, img_result.size(0), img_result.size(1)),
            size=(64, 64), mode='bicubic'
        )
        img_out.append(img_result)
    img_out = torch.vstack(img_out)

    return img_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-emotion', default='neutral', type=str)
    args, _ = parser.parse_known_args()

    DEVICE = 'cuda'
    if not torch.cuda.is_available():
        DEVICE = 'cpu'
    EMOTION = args.emotion

    mode = 'train'

    ### Test dataloader
    # source_dataset_test = Dataset(f'../data/fer_64_segmen_happiness.hdf5', mode='test')
    # loader_params = {'batch_size': 8, 'shuffle': True, 'num_workers': 2}
    # dataloader_test_source = data.DataLoader(source_dataset_test, **loader_params)
    #
    # img_s, roi_s, roi_pos_s = next(iter(dataloader_test_source))
    # roi_s = roi_s.to(DEVICE)
    # img_s = img_s.to(DEVICE)
    # roi_pos_s = roi_pos_s.to(DEVICE)
    #
    # img_g_t = insert_roi_to_image(img_batch=img_s,
    #                               roi_batch=roi_s, roi_pos_batch=roi_pos_s)
    # img_g_s = insert_roi_to_image(img_batch=img_s,
    #                               roi_batch=roi_s, roi_pos_batch=roi_pos_s)
    #
    # img_s = (img_s * 127.5) + 127.5
    # viz_sample = torch.cat((img_s, img_g_t, img_g_s), dim=0)
    # vutils.save_image(viz_sample,
    #                   f'samples_1_test-acc_1.png',
    #                   nrow=4,
    #                   normalize=True)
    #
    # classificator = Model(CLASS_NUM=4, in_image_channels=1, pretrained=False).to(DEVICE)
    # classificator.load_state_dict(torch.load('./model-best_4classes.pt', map_location=torch.device(DEVICE)))
    # classificator.eval()
    # y_prim = classificator.forward(img_g_t)
    # y_idxs = torch.argmax(y_prim, dim=1)

    ### all features data test
    file = h5py.File('../data/fer_64_segmen_happiness.hdf5')
    print(f"Train data length: {len(file['train_roi'])}")
    print(f"Test data length: {len(file['test_roi'])}")
    roi = file[f"{mode}_roi"][60]
    img = file[f"{mode}_imgs"][60]
    roi_pos = file[f"{mode}_roi_pos"][60]

    plt.imshow(roi)
    plt.show()
    plt.imshow(img)
    plt.show()
    vutils.save_image(torch.FloatTensor(roi),
                      f'roi.png',
                      nrow=1,
                      normalize=True)
    vutils.save_image(torch.FloatTensor(img),
                      f'img.png',
                      nrow=1,
                      normalize=True)

    ### Simulate post-processing step of puting mask back into image
    img_t = torch.FloatTensor(img)
    img_t = nnf.interpolate(img_t.view(1, 1, img_t.size(0), img_t.size(1)), size=(256, 256), mode='bicubic').squeeze()

    feat_mask = torch.FloatTensor(roi)
    roi_pos = torch.IntTensor(roi_pos)

    width = roi_pos[3] - roi_pos[2]
    height = roi_pos[1] - roi_pos[0]
    resized_m = nnf.interpolate(
        feat_mask.view(1, 1, feat_mask.size(0), feat_mask.size(1)),
        size=(height, width), mode='bicubic').squeeze()


    def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
        """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.
        `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
        """
        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        # Blend overlay within the determined ranges
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha

        img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

    ### insert mask into original img and alpha blend it
    # Normalize back to 0-255
    img = img_t.squeeze()
    resized_m = (resized_m * 127.5) + 127.5
    img = (img * 127.5) + 127.5
    # Perform blending
    alpha_mask = resized_m[:, :] / 255.0
    img_result = img[:, :]
    # y = roi_pos[0].astype(np.int8)
    # x = roi_pos[2].astype(np.int8)
    y = roi_pos[0]
    x = roi_pos[2]
    overlay_image_alpha(img_result, resized_m, x, y, alpha_mask)

    img_res = nnf.interpolate(img_result.view(1, 1, img_result.size(0), img_result.size(1)), size=(64, 64), mode='bicubic').squeeze()
    plt.imshow(img_res.numpy())
    plt.show()
    print('a')
    ### separate features data test
    # file = h5py.File('../data/fer_64_segmen_masks_happiness.hdf5')
    # print(f"Train data length: {len(file['train_masks_roi'])}")
    # print(f"Test data length: {len(file['test_masks_roi'])}")
    # roi = file[f"{mode}_masks_roi"][255]
    # img = file[f"{mode}_imgs"][254]
    # roi_pos = file[f"{mode}_masks_roi_pos"][255]
    #
    # for r in roi:
    #     plt.imshow(r)
    #     plt.show()
    # plt.imshow(img)
    # plt.show()
    #
    # print(roi_pos)
    #
    # ### Simulate post-processing step of puting mask back into image
    # img_t = torch.FloatTensor(img)
    # img_t = nnf.interpolate(img_t.view(1, 1, img_t.size(0), img_t.size(1)), size=(256, 256), mode='bicubic').squeeze()
    #
    # eye_eyebrow_mask = torch.FloatTensor(roi[0])
    # mouth_lips_mask = torch.FloatTensor(roi[1])
    # e_roi_pos = roi_pos[0]
    # m_roi_pos = roi_pos[1]
    #
    # e_width = e_roi_pos[3] - e_roi_pos[2]
    # e_height = e_roi_pos[1] - e_roi_pos[0]
    # eye_eyebrow_resized_m = nnf.interpolate(
    #     eye_eyebrow_mask.view(1, 1, eye_eyebrow_mask.size(0), eye_eyebrow_mask.size(1)),
    #     size=(e_height, e_width), mode='bicubic').squeeze()
    #
    # m_width = m_roi_pos[3] - m_roi_pos[2]
    # m_height = m_roi_pos[1] - m_roi_pos[0]
    # mouth_lips_resized_m = nnf.interpolate(
    #     mouth_lips_mask.view(1, 1, mouth_lips_mask.size(0), mouth_lips_mask.size(1)),
    #     size=(m_height, m_width), mode='bicubic').squeeze()
    #
    # ### insert masks into original img
    # img_t[e_roi_pos[0]:e_roi_pos[1], e_roi_pos[2]:e_roi_pos[3]] = eye_eyebrow_resized_m
    # img_t[m_roi_pos[0]:m_roi_pos[1], m_roi_pos[2]:m_roi_pos[3]] = mouth_lips_resized_m
    #
    # img_t = nnf.interpolate(img_t.view(1, 1, img_t.size(0), img_t.size(1)), size=(64, 64), mode='bicubic').squeeze()
    # plt.imshow(img_t.numpy())
    # plt.show()

    print('a')
