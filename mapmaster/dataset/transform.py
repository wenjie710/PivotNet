import cv2
import mmcv
import torch
import numpy as np
from PIL import Image
from collections.abc import Sequence

class Resize(object):
    def __init__(self, img_scale=None, backend="cv2", interpolation="bilinear"):
        self.size = img_scale
        self.backend = backend
        self.interpolation = interpolation
        self.cv2_interp_codes = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        self.pillow_interp_codes = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "box": Image.BOX,
            "lanczos": Image.LANCZOS,
            "hamming": Image.HAMMING,
        }

    def __call__(self, data_dict):
        """Call function to resize images.

        Args:
            data_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized data_dict, 'scale_factor' keys are added into result dict.
        """

        imgs = []
        for img in data_dict["images"]:
            img = self.im_resize(img, self.size, backend=self.backend)
            imgs.append(img)
        data_dict["images"] = imgs

        new_h, new_w = imgs[0].shape[:2]
        h, w = data_dict["images"][0].shape[:2]
        w_scale = new_w / w
        h_scale = new_h / h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        data_dict["extra_infos"].update({"scale_factor": scale_factor})

        return data_dict

    def im_resize(self, img, size, return_scale=False, interpolation="bilinear", out=None, backend="cv2"):
        """Resize image to a given size.
        Args:
            img (ndarray): The input image.
            size (tuple[int]): Target size (w, h).
            return_scale (bool): Whether to return `w_scale` and `h_scale`.
            interpolation (str): Interpolation method, accepted values are
                "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
                backend, "nearest", "bilinear" for 'pillow' backend.
            out (ndarray): The output destination.
            backend (str | None): The image resize backend type. Options are `cv2`,
                `pillow`, `None`.
        Returns:
            tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
        """
        h, w = img.shape[:2]
        if backend not in ["cv2", "pillow"]:
            raise ValueError(
                f"backend: {backend} is not supported for resize." f"Supported backends are 'cv2', 'pillow'"
            )

        if backend == "pillow":
            assert img.dtype == np.uint8, "Pillow backend only support uint8 type"
            pil_image = Image.fromarray(img)
            pil_image = pil_image.resize(size, self.pillow_interp_codes[interpolation])
            resized_img = np.array(pil_image)
        else:
            resized_img = cv2.resize(img, size, dst=out, interpolation=self.cv2_interp_codes[interpolation])
        if not return_scale:
            return resized_img
        else:
            w_scale = size[0] / w
            h_scale = size[1] / h
            return resized_img, w_scale, h_scale

class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, data_dict):
        imgs = []
        for img in data_dict["images"]:
            if self.to_rgb:
                img = img.astype(np.float32) / 255.0
            img = self.im_normalize(img, self.mean, self.std, self.to_rgb)
            imgs.append(img)
        data_dict["images"] = imgs
        data_dict["extra_infos"]["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return data_dict

    @staticmethod
    def im_normalize(img, mean, std, to_rgb=True):
        img = img.copy().astype(np.float32)
        assert img.dtype != np.uint8  # cv2 inplace normalization does not accept uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img


class ToTensor(object):
    """Default formatting bundle."""

    def __call__(self, data_dict):
        """Call function to transform and format common fields in data_dict.

        Args:
            data_dict (dict): Data dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with default bundle.
        """

        for k in ["images", "extrinsic", "intrinsic", "ida_mats"]:
            if k == "images":
                data_dict[k] = np.stack([img.transpose(2, 0, 1) for img in data_dict[k]], axis=0)
            data_dict[k] = self.to_tensor(np.ascontiguousarray(data_dict[k]))

        for k in ["masks", "points", "labels"]:
            data_dict["targets"][k] = self.to_tensor(np.ascontiguousarray(data_dict["targets"][k]))

        return data_dict

    @staticmethod
    def to_tensor(data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, Sequence) and not mmcv.is_str(data):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")

class ToTensor_Pivot(object):
    """Default formatting bundle."""

    def __call__(self, data_dict):
        """Call function to transform and format common fields in data_dict.

        Args:
            data_dict (dict): Data dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with default bundle.
        """
        if "images" in data_dict:
            if isinstance(data_dict["images"], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in data_dict["images"]]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                data_dict["images"] = self.to_tensor(imgs)
            else:
                img = np.ascontiguousarray(data_dict["img"].transpose(2, 0, 1))
                data_dict["images"] = self.to_tensor(img)

        for k in ["masks"]:
            data_dict["targets"][k] = self.to_tensor(np.ascontiguousarray(data_dict["targets"][k]))

        return data_dict

    @staticmethod
    def to_tensor(data):
        """Convert objects of various python types to :obj:`torch.Tensor`.
        Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
        :class:`Sequence`, :class:`int` and :class:`float`.
        Args:
            data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
                be converted.
        """

        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, Sequence) and not mmcv.is_str(data):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")



class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size_divisor is not None

    def __call__(self, data_dict):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            data_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        padded_img = None
        padded_imgs = []
        for img in data_dict["images"]:
            padded_img = self.im_pad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
            padded_imgs.append(padded_img)
        data_dict["images"] = padded_imgs
        data_dict["extra_infos"].update(
            {
                "pad_shape": padded_img.shape,
                "pad_size_divisor": self.size_divisor if self.size_divisor is not None else "None",
            }
        )
        return data_dict

    def im_pad_to_multiple(self, img, divisor, pad_val=0):
        """Pad an image to ensure each edge to be multiple to some number.
        Args:
            img (ndarray): Image to be padded.
            divisor (int): Padded image edges will be multiple to divisor.
            pad_val (Number | Sequence[Number]): Same as :func:`impad`.
        Returns:
            ndarray: The padded image.
        """
        pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
        pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
        return self.im_pad(img, shape=(pad_h, pad_w), pad_val=pad_val)
