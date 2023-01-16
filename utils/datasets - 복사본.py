# # YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# """
# Dataloaders and dataset utils
# """

# import glob
# import hashlib
# import json
# import math
# import os
# import random
# import shutil
# import time
# from itertools import repeat
# from multiprocessing.pool import Pool, ThreadPool
# from pathlib import Path
# from threading import Thread
# from urllib.parse import urlparse
# from zipfile import ZipFile

# import numpy as np
# import torch
# import torch.nn.functional as F
# import yaml
# from PIL import ExifTags, Image, ImageOps
# from torch.utils.data import DataLoader, Dataset, dataloader, distributed
# from tqdm import tqdm

# from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
# from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
#                            cv2, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
# from utils.torch_utils import torch_distributed_zero_first

# # Parameters
# HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
# IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
# VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
# BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
# LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html

# # Get orientation exif tag
# for orientation in ExifTags.TAGS.keys():
#     if ExifTags.TAGS[orientation] == 'Orientation':
#         break


# def get_hash(paths):
#     # Returns a single hash value of a list of paths (files or dirs)
#     size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
#     h = hashlib.md5(str(size).encode())  # hash sizes
#     h.update(''.join(paths).encode())  # hash paths
#     return h.hexdigest()  # return hash


# def exif_size(img):
#     # Returns exif-corrected PIL size
#     s = img.size  # (width, height)
#     try:
#         rotation = dict(img._getexif().items())[orientation]
#         if rotation == 6:  # rotation 270
#             s = (s[1], s[0])
#         elif rotation == 8:  # rotation 90
#             s = (s[1], s[0])
#     except Exception:
#         pass

#     return s


# def exif_transpose(image):
#     """
#     Transpose a PIL image accordingly if it has an EXIF Orientation tag.
#     Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

#     :param image: The image to transpose.
#     :return: An image.
#     """
#     exif = image.getexif()
#     orientation = exif.get(0x0112, 1)  # default 1
#     if orientation > 1:
#         method = {
#             2: Image.FLIP_LEFT_RIGHT,
#             3: Image.ROTATE_180,
#             4: Image.FLIP_TOP_BOTTOM,
#             5: Image.TRANSPOSE,
#             6: Image.ROTATE_270,
#             7: Image.TRANSVERSE,
#             8: Image.ROTATE_90,}.get(orientation)
#         if method is not None:
#             image = image.transpose(method)
#             del exif[0x0112]
#             image.info["exif"] = exif.tobytes()
#     return image


# def create_dataloader(path,
#                       imgsz,
#                       batch_size,
#                       stride,
#                       single_cls=False,
#                       hyp=None,
#                       augment=False,
#                       cache=False,
#                       pad=0.0,
#                       rect=False,
#                       rank=-1,
#                       workers=8,
#                       image_weights=False,
#                       quad=False,
#                       prefix='',
#                       shuffle=False):
#     if rect and shuffle:
#         LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
#         shuffle = False
#     with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
#         dataset = LoadImagesAndLabels(
#             path,
#             imgsz,
#             batch_size,
#             augment=augment,  # augmentation
#             hyp=hyp,  # hyperparameters
#             rect=rect,  # rectangular batches
#             cache_images=cache,
#             single_cls=single_cls,
#             stride=int(stride),
#             pad=pad,
#             image_weights=image_weights,
#             prefix=prefix)

#     batch_size = min(batch_size, len(dataset))
#     nd = torch.cuda.device_count()  # number of CUDA devices
#     nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
#     sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
#     loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
#     return loader(dataset,
#                   batch_size=batch_size,
#                   shuffle=shuffle and sampler is None,
#                   num_workers=nw,
#                   sampler=sampler,
#                   pin_memory=True,
#                   collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset


# class InfiniteDataLoader(dataloader.DataLoader):
#     """ Dataloader that reuses workers

#     Uses same syntax as vanilla DataLoader
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
#         self.iterator = super().__iter__()

#     def __len__(self):
#         return len(self.batch_sampler.sampler)

#     def __iter__(self):
#         for i in range(len(self)):
#             yield next(self.iterator)


# class _RepeatSampler:
#     """ Sampler that repeats forever

#     Args:
#         sampler (Sampler)
#     """

#     def __init__(self, sampler):
#         self.sampler = sampler

#     def __iter__(self):
#         while True:
#             yield from iter(self.sampler)


# class LoadImages:
#     # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
#     def __init__(self, path, img_size=416, stride=32, auto=True):
#         p = str(Path(path).resolve())  # os-agnostic absolute path
#         if '*' in p:
#             files = sorted(glob.glob(p, recursive=True))  # glob
#         elif os.path.isdir(p):
#             files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
#         elif os.path.isfile(p):
#             files = [p]  # files
#         else:
#             raise Exception(f'ERROR: {p} does not exist')

#         images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
#         videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
#         ni, nv = len(images), len(videos)

#         self.img_size = img_size
#         self.stride = stride
#         self.files = images + videos
#         self.nf = ni + nv  # number of files
#         self.video_flag = [False] * ni + [True] * nv
#         self.mode = 'image'
#         self.auto = auto
#         if any(videos):
#             self.new_video(videos[0])  # new video
#         else:
#             self.cap = None
#         assert self.nf > 0, f'No images or videos found in {p}. ' \
#                             f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

#     def __iter__(self):
#         self.count = 1
#         return self

#     def __next__(self):
#         if self.count == self.nf:
#             raise StopIteration
#         path = self.files[self.count]

#         if self.video_flag[self.count]:
#             # Read video
#             self.mode = 'video'
#             ret_val, img0 = self.cap.read()
#             while not ret_val:
#                 self.count += 1
#                 self.cap.release()
#                 if self.count == self.nf:  # last video
#                     raise StopIteration
#                 else:
#                     path = self.files[self.count]
#                     self.new_video(path)
#                     ret_val, img0 = self.cap.read()

#             self.frame += 1
#             s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

#         else:
#             # Read image
#             self.count += 1
#             img0 = cv2.imread(path)  # BGR
#             assert img0 is not None, f'Image Not Found {path}'
#             s = f'image {self.count}/{self.nf} {path}: '

#         # Padded resize
#         img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

#         # Convert
#         img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#         img = np.ascontiguousarray(img)

#         return path, img, img0, self.cap, s

#     def new_video(self, path):
#         self.frame = 1
#         self.cap = cv2.VideoCapture(path)
#         self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     def __len__(self):
#         return self.nf  # number of files


# class LoadWebcam:  # for inference
#     # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
#     def __init__(self, pipe='0', img_size=416, stride=32):
#         self.img_size = img_size
#         self.stride = stride
#         self.pipe = eval(pipe) if pipe.isnumeric() else pipe
#         self.cap = cv2.VideoCapture(self.pipe)  # video capture object
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

#     def __iter__(self):
#         self.count = -1
#         return self

#     def __next__(self):
#         self.count += 1
#         if cv2.waitKey(1) == ord('q'):  # q to quit
#             self.cap.release()
#             cv2.destroyAllWindows()
#             raise StopIteration

#         # Read frame
#         ret_val, img0 = self.cap.read()
#         img0 = cv2.flip(img0, 1)  # flip left-right

#         # Print
#         assert ret_val, f'Camera Error {self.pipe}'
#         img_path = 'webcam.jpg'
#         s = f'webcam {self.count}: '

#         # Padded resize
#         img = letterbox(img0, self.img_size, stride=self.stride)[0]

#         # Convert
#         img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#         img = np.ascontiguousarray(img)

#         return img_path, img, img0, None, s

#     def __len__(self):
#         return 0


# class LoadStreams:
#     # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
#     def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
#         self.mode = 'stream'
#         self.img_size = img_size
#         self.stride = stride

#         if os.path.isfile(sources):
#             with open(sources) as f:
#                 sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
#         else:
#             sources = [sources]

#         n = len(sources)
#         self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
#         self.sources = [clean_str(x) for x in sources]  # clean source names for later
#         self.auto = auto
#         for i, s in enumerate(sources):  # index, source
#             # Start thread to read frames from video stream
#             st = f'{i + 1}/{n}: {s}... '
#             if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
#                 check_requirements(('pafy', 'youtube_dl==2020.12.2'))
#                 import pafy
#                 s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
#             s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
#             cap = cv2.VideoCapture(s)
#             assert cap.isOpened(), f'{st}Failed to open {s}'
#             w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
#             self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
#             self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

#             _, self.imgs[i] = cap.read()  # guarantee first frame
#             self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
#             LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
#             self.threads[i].start()
#         LOGGER.info('')  # newline

#         # check for common shapes
#         s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
#         self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
#         if not self.rect:
#             LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

#     def update(self, i, cap, stream):
#         # Read stream `i` frames in daemon thread
#         n, f, read = 1, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
#         while cap.isOpened() and n < f:
#             n += 1
#             # _, self.imgs[index] = cap.read()
#             cap.grab()
#             if n % read == 0:
#                 success, im = cap.retrieve()
#                 if success:
#                     self.imgs[i] = im
#                 else:
#                     LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
#                     self.imgs[i] = np.zeros_like(self.imgs[i])
#                     cap.open(stream)  # re-open stream if signal was lost
#             time.sleep(1 / self.fps[i])  # wait time

#     def __iter__(self):
#         self.count = -1
#         return self

#     def __next__(self):
#         self.count += 1
#         if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
#             cv2.destroyAllWindows()
#             raise StopIteration

#         # Letterbox
#         img0 = self.imgs.copy()
#         img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

#         # Stack
#         img = np.stack(img, 0)

#         # Convert
#         img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
#         img = np.ascontiguousarray(img)

#         return self.sources, img, img0, None, ''

#     def __len__(self):
#         return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


# def img2label_paths(img_paths):
#     # Define label paths as a function of image paths
#     sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
#     return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


# class LoadImagesAndLabels(Dataset):
#     # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
#     cache_version = 0.6  # dataset labels *.cache version

#     def __init__(self,
#                  path,
#                  img_size=640,
#                  batch_size=16,
#                  augment=False,
#                  hyp=None,
#                  rect=False,
#                  image_weights=False,
#                  cache_images=False,
#                  single_cls=False,
#                  stride=32,
#                  pad=0.0,
#                  prefix=''):
#         self.img_size = img_size
#         self.augment = augment
#         self.hyp = hyp
#         self.image_weights = image_weights
#         self.rect = False if image_weights else rect
#         self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
#         self.mosaic_border = [-img_size // 2, -img_size // 2]
#         self.stride = stride
#         self.path = path
#         self.albumentations = Albumentations() if augment else None

#         try:
#             f = []  # image files
#             for p in path if isinstance(path, list) else [path]:
#                 p = Path(p)  # os-agnostic
#                 if p.is_dir():  # dir
#                     f += glob.glob(str(p / '**' / '*.*'), recursive=True)
#                     # f = list(p.rglob('*.*'))  # pathlib
#                 elif p.is_file():  # file
#                     with open(p) as t:
#                         t = t.read().strip().splitlines()
#                         parent = str(p.parent) + os.sep
#                         f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
#                         # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
#                 else:
#                     raise Exception(f'{prefix}{p} does not exist')
#             self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
#             # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
#             assert self.im_files, f'{prefix}No images found'
#         except Exception as e:
#             raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

#         # Check cache
#         self.label_files = img2label_paths(self.im_files)  # labels
#         cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
#         try:
#             cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
#             assert cache['version'] == self.cache_version  # same version
#             assert cache['hash'] == get_hash(self.label_files + self.im_files)  # same hash
#         except Exception:
#             cache, exists = self.cache_labels(cache_path, prefix), False  # cache

#         # Display cache
#         nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
#         if exists and LOCAL_RANK in (-1, 0):
#             d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
#             tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
#             if cache['msgs']:
#                 LOGGER.info('\n'.join(cache['msgs']))  # display warnings
#         assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

#         # Read cache
#         [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
#         labels, shapes, self.segments = zip(*cache.values())
#         self.labels = list(labels)
#         self.shapes = np.array(shapes, dtype=np.float64)
#         self.im_files = list(cache.keys())  # update
#         self.label_files = img2label_paths(cache.keys())  # update
#         n = len(shapes)  # number of images
#         bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
#         nb = bi[-1] + 1  # number of batches
#         self.batch = bi  # batch index of image
#         self.n = n
#         self.indices = range(n)

#         # Update labels
#         include_class = []  # filter labels to include only these classes (optional)
#         include_class_array = np.array(include_class).reshape(1, -1)
#         for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
#             if include_class:
#                 j = (label[:, 0:1] == include_class_array).any(1)
#                 self.labels[i] = label[j]
#                 if segment:
#                     self.segments[i] = segment[j]
#             if single_cls:  # single-class training, merge all classes into 0
#                 self.labels[i][:, 0] = 0
#                 if segment:
#                     self.segments[i][:, 0] = 0

#         # Rectangular Training
#         if self.rect:
#             # Sort by aspect ratio
#             s = self.shapes  # wh
#             ar = s[:, 1] / s[:, 0]  # aspect ratio
#             irect = ar.argsort()
#             self.im_files = [self.im_files[i] for i in irect]
#             self.label_files = [self.label_files[i] for i in irect]
#             self.labels = [self.labels[i] for i in irect]
#             self.shapes = s[irect]  # wh
#             ar = ar[irect]

#             # Set training image shapes
#             shapes = [[1, 1]] * nb
#             for i in range(nb):
#                 ari = ar[bi == i]
#                 mini, maxi = ari.min(), ari.max()
#                 if maxi < 1:
#                     shapes[i] = [maxi, 1]
#                 elif mini > 1:
#                     shapes[i] = [1, 1 / mini]

#             self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

#         # Cache images into RAM/disk for faster training (WARNING: large datasets may exceed system resources)
#         self.ims = [None] * n
#         self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
#         if cache_images:
#             gb = 0  # Gigabytes of cached images
#             self.im_hw0, self.im_hw = [None] * n, [None] * n
#             fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
#             results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
#             pbar = tqdm(enumerate(results), total=n, bar_format=BAR_FORMAT, disable=LOCAL_RANK > 0)
#             for i, x in pbar:
#                 if cache_images == 'disk':
#                     gb += self.npy_files[i].stat().st_size
#                 else:  # 'ram'
#                     self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
#                     gb += self.ims[i].nbytes
#                 pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
#             pbar.close()

#     def cache_labels(self, path=Path('./labels.cache'), prefix=''):
#         # Cache dataset labels, check images and read shapes
#         x = {}  # dict
#         nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
#         desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
#         with Pool(NUM_THREADS) as pool:
#             pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
#                         desc=desc,
#                         total=len(self.im_files),
#                         bar_format=BAR_FORMAT)
#             for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
#                 nm += nm_f
#                 nf += nf_f
#                 ne += ne_f
#                 nc += nc_f
#                 if im_file:
#                     x[im_file] = [lb, shape, segments]
#                 if msg:
#                     msgs.append(msg)
#                 pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

#         pbar.close()
#         if msgs:
#             LOGGER.info('\n'.join(msgs))
#         if nf == 0:
#             LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
#         x['hash'] = get_hash(self.label_files + self.im_files)
#         x['results'] = nf, nm, ne, nc, len(self.im_files)
#         x['msgs'] = msgs  # warnings
#         x['version'] = self.cache_version  # cache version
#         try:
#             np.save(path, x)  # save cache for next time
#             path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
#             LOGGER.info(f'{prefix}New cache created: {path}')
#         except Exception as e:
#             LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
#         return x

#     def __len__(self):
#         return len(self.im_files)

#     # def __iter__(self):
#     #     self.count = -1
#     #     print('ran dataset iter')
#     #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
#     #     return self

#     def __getitem__(self, index):
#         index = self.indices[index]  # linear, shuffled, or image_weights

#         hyp = self.hyp
#         mosaic = self.mosaic and random.random() < hyp['mosaic']
#         if mosaic:
#             # Load mosaic
#             img, labels = self.load_mosaic(index)
#             shapes = None

#             # MixUp augmentation
#             if random.random() < hyp['mixup']:
#                 img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

#         else:
#             # Load image
#             img, (h0, w0), (h, w) = self.load_image(index)

#             # Letterbox
#             shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
#             img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
#             shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

#             labels = self.labels[index].copy()
#             if labels.size:  # normalized xywh to pixel xyxy format
#                 labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

#             if self.augment:
#                 img, labels = random_perspective(img,
#                                                  labels,
#                                                  degrees=hyp['degrees'],
#                                                  translate=hyp['translate'],
#                                                  scale=hyp['scale'],
#                                                  shear=hyp['shear'],
#                                                  perspective=hyp['perspective'])

#         nl = len(labels)  # number of labels
#         if nl:
#             labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

#         if self.augment:
#             # Albumentations
#             img, labels = self.albumentations(img, labels)
#             nl = len(labels)  # update after albumentations

#             # HSV color-space
#             augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

#             # Flip up-down
#             if random.random() < hyp['flipud']:
#                 img = np.flipud(img)
#                 if nl:
#                     labels[:, 2] = 1 - labels[:, 2]

#             # Flip left-right
#             if random.random() < hyp['fliplr']:
#                 img = np.fliplr(img)
#                 if nl:
#                     labels[:, 1] = 1 - labels[:, 1]

#             # Cutouts
#             # labels = cutout(img, labels, p=0.5)
#             # nl = len(labels)  # update after cutout

#         labels_out = torch.zeros((nl, 6))
#         if nl:
#             labels_out[:, 1:] = torch.from_numpy(labels)

#         # Convert
#         img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#         img = np.ascontiguousarray(img)

#         return torch.from_numpy(img), labels_out, self.im_files[index], shapes

#     def load_image(self, i):
#         # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
#         im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
#         if im is None:  # not cached in RAM
#             if fn.exists():  # load npy
#                 im = np.load(fn)
#             else:  # read image
#                 im = cv2.imread(f)  # BGR
#                 assert im is not None, f'Image Not Found {f}'
#             h0, w0 = im.shape[:2]  # orig hw
#             r = self.img_size / max(h0, w0)  # ratio
#             if r != 1:  # if sizes are not equal
#                 im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
#                                 interpolation=cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA)
#             return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
#         else:
#             return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

#     def cache_images_to_disk(self, i):
#         # Saves an image as an *.npy file for faster loading
#         f = self.npy_files[i]
#         if not f.exists():
#             np.save(f.as_posix(), cv2.imread(self.im_files[i]))

#     def load_mosaic(self, index):
#         # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
#         labels4, segments4 = [], []
#         s = self.img_size
#         yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
#         indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
#         random.shuffle(indices)
#         for i, index in enumerate(indices):
#             # Load image
#             img, _, (h, w) = self.load_image(index)

#             # place img in img4
#             if i == 0:  # top left
#                 img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
#                 x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
#                 x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
#             elif i == 1:  # top right
#                 x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
#                 x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
#             elif i == 2:  # bottom left
#                 x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
#                 x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
#             elif i == 3:  # bottom right
#                 x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
#                 x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

#             img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
#             padw = x1a - x1b
#             padh = y1a - y1b

#             # Labels
#             labels, segments = self.labels[index].copy(), self.segments[index].copy()
#             if labels.size:
#                 labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
#                 segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
#             labels4.append(labels)
#             segments4.extend(segments)

#         # Concat/clip labels
#         labels4 = np.concatenate(labels4, 0)
#         for x in (labels4[:, 1:], *segments4):
#             np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
#         # img4, labels4 = replicate(img4, labels4)  # replicate

#         # Augment
#         img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
#         img4, labels4 = random_perspective(img4,
#                                            labels4,
#                                            segments4,
#                                            degrees=self.hyp['degrees'],
#                                            translate=self.hyp['translate'],
#                                            scale=self.hyp['scale'],
#                                            shear=self.hyp['shear'],
#                                            perspective=self.hyp['perspective'],
#                                            border=self.mosaic_border)  # border to remove

#         return img4, labels4

#     def load_mosaic9(self, index):
#         # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
#         labels9, segments9 = [], []
#         s = self.img_size
#         indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
#         random.shuffle(indices)
#         hp, wp = -1, -1  # height, width previous
#         for i, index in enumerate(indices):
#             # Load image
#             img, _, (h, w) = self.load_image(index)

#             # place img in img9
#             if i == 0:  # center
#                 img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
#                 h0, w0 = h, w
#                 c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
#             elif i == 1:  # top
#                 c = s, s - h, s + w, s
#             elif i == 2:  # top right
#                 c = s + wp, s - h, s + wp + w, s
#             elif i == 3:  # right
#                 c = s + w0, s, s + w0 + w, s + h
#             elif i == 4:  # bottom right
#                 c = s + w0, s + hp, s + w0 + w, s + hp + h
#             elif i == 5:  # bottom
#                 c = s + w0 - w, s + h0, s + w0, s + h0 + h
#             elif i == 6:  # bottom left
#                 c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
#             elif i == 7:  # left
#                 c = s - w, s + h0 - h, s, s + h0
#             elif i == 8:  # top left
#                 c = s - w, s + h0 - hp - h, s, s + h0 - hp

#             padx, pady = c[:2]
#             x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

#             # Labels
#             labels, segments = self.labels[index].copy(), self.segments[index].copy()
#             if labels.size:
#                 labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
#                 segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
#             labels9.append(labels)
#             segments9.extend(segments)

#             # Image
#             img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
#             hp, wp = h, w  # height, width previous

#         # Offset
#         yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
#         img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

#         # Concat/clip labels
#         labels9 = np.concatenate(labels9, 0)
#         labels9[:, [1, 3]] -= xc
#         labels9[:, [2, 4]] -= yc
#         c = np.array([xc, yc])  # centers
#         segments9 = [x - c for x in segments9]

#         for x in (labels9[:, 1:], *segments9):
#             np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
#         # img9, labels9 = replicate(img9, labels9)  # replicate

#         # Augment
#         img9, labels9 = random_perspective(img9,
#                                            labels9,
#                                            segments9,
#                                            degrees=self.hyp['degrees'],
#                                            translate=self.hyp['translate'],
#                                            scale=self.hyp['scale'],
#                                            shear=self.hyp['shear'],
#                                            perspective=self.hyp['perspective'],
#                                            border=self.mosaic_border)  # border to remove

#         return img9, labels9

#     @staticmethod
#     def collate_fn(batch):
#         im, label, path, shapes = zip(*batch)  # transposed
#         for i, lb in enumerate(label):
#             lb[:, 0] = i  # add target image index for build_targets()
#         return torch.stack(im, 0), torch.cat(label, 0), path, shapes

#     @staticmethod
#     def collate_fn4(batch):
#         img, label, path, shapes = zip(*batch)  # transposed
#         n = len(shapes) // 4
#         im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

#         ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
#         wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
#         s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
#         for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
#             i *= 4
#             if random.random() < 0.5:
#                 im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
#                                    align_corners=False)[0].type(img[i].type())
#                 lb = label[i]
#             else:
#                 im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
#                 lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
#             im4.append(im)
#             label4.append(lb)

#         for i, lb in enumerate(label4):
#             lb[:, 0] = i  # add target image index for build_targets()

#         return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4


# # Ancillary functions --------------------------------------------------------------------------------------------------
# def create_folder(path='./new'):
#     # Create folder
#     if os.path.exists(path):
#         shutil.rmtree(path)  # delete output folder
#     os.makedirs(path)  # make new output folder


# def flatten_recursive(path=DATASETS_DIR / 'coco128'):
#     # Flatten a recursive directory by bringing all files to top level
#     new_path = Path(str(path) + '_flat')
#     create_folder(new_path)
#     for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
#         shutil.copyfile(file, new_path / Path(file).name)


# def extract_boxes(path=DATASETS_DIR / 'coco128'):  # from utils.datasets import *; extract_boxes()
#     # Convert detection dataset into classification dataset, with one directory per class
#     path = Path(path)  # images dir
#     shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
#     files = list(path.rglob('*.*'))
#     n = len(files)  # number of files
#     for im_file in tqdm(files, total=n):
#         if im_file.suffix[1:] in IMG_FORMATS:
#             # image
#             im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
#             h, w = im.shape[:2]

#             # labels
#             lb_file = Path(img2label_paths([str(im_file)])[0])
#             if Path(lb_file).exists():
#                 with open(lb_file) as f:
#                     lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

#                 for j, x in enumerate(lb):
#                     c = int(x[0])  # class
#                     f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
#                     if not f.parent.is_dir():
#                         f.parent.mkdir(parents=True)

#                     b = x[1:] * [w, h, w, h]  # box
#                     # b[2:] = b[2:].max()  # rectangle to square
#                     b[2:] = b[2:] * 1.2 + 3  # pad
#                     b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

#                     b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
#                     b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
#                     assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


# def autosplit(path=DATASETS_DIR / 'coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
#     """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
#     Usage: from utils.datasets import *; autosplit()
#     Arguments
#         path:            Path to images directory
#         weights:         Train, val, test weights (list, tuple)
#         annotated_only:  Only use images with an annotated txt file
#     """
#     path = Path(path)  # images dir
#     files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
#     n = len(files)  # number of files
#     random.seed(0)  # for reproducibility
#     indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

#     txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
#     [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

#     print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
#     for i, img in tqdm(zip(indices, files), total=n):
#         if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
#             with open(path.parent / txt[i], 'a') as f:
#                 f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


# def verify_image_label(args):
#     # Verify one image-label pair
#     im_file, lb_file, prefix = args
#     nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
#     try:
#         # verify images
#         im = Image.open(im_file)
#         im.verify()  # PIL verify
#         shape = exif_size(im)  # image size
#         assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
#         assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
#         if im.format.lower() in ('jpg', 'jpeg'):
#             with open(im_file, 'rb') as f:
#                 f.seek(-2, 2)
#                 if f.read() != b'\xff\xd9':  # corrupt JPEG
#                     ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
#                     msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

#         # verify labels
#         if os.path.isfile(lb_file):
#             nf = 1  # label found
#             with open(lb_file) as f:
#                 lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
#                 if any(len(x) > 6 for x in lb):  # is segment
#                     classes = np.array([x[0] for x in lb], dtype=np.float32)
#                     segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
#                     lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
#                 lb = np.array(lb, dtype=np.float32)
#             nl = len(lb)
#             if nl:
#                 assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
#                 assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
#                 assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
#                 _, i = np.unique(lb, axis=0, return_index=True)
#                 if len(i) < nl:  # duplicate row check
#                     lb = lb[i]  # remove duplicates
#                     if segments:
#                         segments = segments[i]
#                     msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
#             else:
#                 ne = 1  # label empty
#                 lb = np.zeros((0, 5), dtype=np.float32)
#         else:
#             nm = 1  # label missing
#             lb = np.zeros((0, 5), dtype=np.float32)
#         return im_file, lb, shape, segments, nm, nf, ne, nc, msg
#     except Exception as e:
#         nc = 1
#         msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
#         return [None, None, None, None, nm, nf, ne, nc, msg]


# def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
#     """ Return dataset statistics dictionary with images and instances counts per split per class
#     To run in parent directory: export PYTHONPATH="$PWD/yolov5"
#     Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
#     Usage2: from utils.datasets import *; dataset_stats('path/to/coco128_with_yaml.zip')
#     Arguments
#         path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
#         autodownload:   Attempt to download dataset if not found locally
#         verbose:        Print stats dictionary
#     """

#     def round_labels(labels):
#         # Update labels to integer class and 6 decimal place floats
#         return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

#     def unzip(path):
#         # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
#         if str(path).endswith('.zip'):  # path is data.zip
#             assert Path(path).is_file(), f'Error unzipping {path}, file not found'
#             ZipFile(path).extractall(path=path.parent)  # unzip
#             dir = path.with_suffix('')  # dataset directory == zip name
#             return True, str(dir), next(dir.rglob('*.yaml'))  # zipped, data_dir, yaml_path
#         else:  # path is data.yaml
#             return False, None, path

#     def hub_ops(f, max_dim=1920):
#         # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
#         f_new = im_dir / Path(f).name  # dataset-hub image filename
#         try:  # use PIL
#             im = Image.open(f)
#             r = max_dim / max(im.height, im.width)  # ratio
#             if r < 1.0:  # image too large
#                 im = im.resize((int(im.width * r), int(im.height * r)))
#             im.save(f_new, 'JPEG', quality=75, optimize=True)  # save
#         except Exception as e:  # use OpenCV
#             print(f'WARNING: HUB ops PIL failure {f}: {e}')
#             im = cv2.imread(f)
#             im_height, im_width = im.shape[:2]
#             r = max_dim / max(im_height, im_width)  # ratio
#             if r < 1.0:  # image too large
#                 im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
#             cv2.imwrite(str(f_new), im)

#     zipped, data_dir, yaml_path = unzip(Path(path))
#     with open(check_yaml(yaml_path), errors='ignore') as f:
#         data = yaml.safe_load(f)  # data dict
#         if zipped:
#             data['path'] = data_dir  # TODO: should this be dir.resolve()?
#     check_dataset(data, autodownload)  # download dataset if missing
#     hub_dir = Path(data['path'] + ('-hub' if hub else ''))
#     stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
#     for split in 'train', 'val', 'test':
#         if data.get(split) is None:
#             stats[split] = None  # i.e. no test set
#             continue
#         x = []
#         dataset = LoadImagesAndLabels(data[split])  # load dataset
#         for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
#             x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
#         x = np.array(x)  # shape(128x80)
#         stats[split] = {
#             'instance_stats': {
#                 'total': int(x.sum()),
#                 'per_class': x.sum(0).tolist()},
#             'image_stats': {
#                 'total': dataset.n,
#                 'unlabelled': int(np.all(x == 0, 1).sum()),
#                 'per_class': (x > 0).sum(0).tolist()},
#             'labels': [{
#                 str(Path(k).name): round_labels(v.tolist())} for k, v in zip(dataset.im_files, dataset.labels)]}

#         if hub:
#             im_dir = hub_dir / 'images'
#             im_dir.mkdir(parents=True, exist_ok=True)
#             for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.im_files), total=dataset.n, desc='HUB Ops'):
#                 pass

#     # Profile
#     stats_path = hub_dir / 'stats.json'
#     if profile:
#         for _ in range(1):
#             file = stats_path.with_suffix('.npy')
#             t1 = time.time()
#             np.save(file, stats)
#             t2 = time.time()
#             x = np.load(file, allow_pickle=True)
#             print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

#             file = stats_path.with_suffix('.json')
#             t1 = time.time()
#             with open(file, 'w') as f:
#                 json.dump(stats, f)  # save stats *.json
#             t2 = time.time()
#             with open(file) as f:
#                 x = json.load(f)  # load hyps dict
#             print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

#     # Save, print and return
#     if hub:
#         print(f'Saving {stats_path.resolve()}...')
#         with open(stats_path, 'w') as f:
#             json.dump(stats, f)  # save stats.json
#     if verbose:
#         print(json.dumps(stats, indent=2, sort_keys=False))
#     return stats


# # YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# """
# Run inference on images, videos, directories, streams, etc.

# Usage - sources:
#     $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
#                                                              img.jpg        # image
#                                                              vid.mp4        # video
#                                                              path/          # directory
#                                                              path/*.jpg     # glob
#                                                              'https://youtu.be/Zgi9g1ksQHc'  # YouTube
#                                                              'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

# Usage - formats:
#     $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
#                                          yolov5s.torchscript        # TorchScript
#                                          yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
#                                          yolov5s.xml                # OpenVINO
#                                          yolov5s.engine             # TensorRT
#                                          yolov5s.mlmodel            # CoreML (macOS-only)
#                                          yolov5s_saved_model        # TensorFlow SavedModel
#                                          yolov5s.pb                 # TensorFlow GraphDef
#                                          yolov5s.tflite             # TensorFlow Lite
#                                          yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
# """

# import argparse
# import os
# import sys
# from ast import keyword
# from collections import deque
# from logging import Logger, LogRecord
# from pathlib import Path
# from pickle import APPEND
# from re import A
# from tabnanny import filename_only
# #import time #kwj
# from time import time
# from tracemalloc import start
# from urllib.request import AbstractDigestAuthHandler

# import cv2  # kwj
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn

# #import keyboard
# from gui_buttons import Buttons

# #Initialize Buttons 
# button = Buttons()
# button.add_button("wound", 400, 20)
# button.add_button("dot", 400, 80)

# colors = button.colors


# global im0
# wound_que = deque(maxlen = 4)


# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from models.common import DetectMultiBackend
# from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
# from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
# from utils.plots import Annotator, colors, save_one_box
# from utils.torch_utils import select_device, time_sync


# # Button
# def click_button(event, x, y, flags, params):
#     global button_wound
#     if event == cv2.EVENT_LBUTTONDOWN:
#         button.button_click(x, y)
#         #print(x,y)
#     # global button_dot
#     # if event == cv2.EVENT_LBUTTONDOWN:
#     #     button.button_click(x, y)
        
        
# @torch.no_grad()
# def run(
#         #weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
#         weights=ROOT / 'best.pt',  # model.pt path(s)
#         #source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
#         source = cv2.CAP_DSHOW + 0,
#         #data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
#         data=ROOT / 'data/data.yaml',
#         #imgsz=(640, 640),  # inference size (height, width)
#         imgsz = (416, 416),
#         conf_thres=0.7,  # confidence threshold
#         iou_thres=0.45,  # NMS IOU threshold
#         max_det=7,  # maximum detections per image
#         device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#         view_img=False,  # show results
#         save_txt=False,  # save results to *.txt
#         save_conf=False,  # save confidences in --save-txt labels
#         save_crop=False,  # save cropped prediction boxes
#         nosave=True,  # do not save images/videos
#         classes=None,  # filter by class: --class 0, or --class 0 2 3
#         agnostic_nms=False,  # class-agnostic NMS
#         augment=False,  # augmented inference
#         visualize=False,  # visualize features
#         update=False,  # update all models
#         project=ROOT / 'runs/detect',  # save results to project/name
#         name='exp',  # save results to project/name
#         exist_ok=False,  # existing project/name ok, do not increment
#         line_thickness=3,  # bounding box thickness (pixels)
#         hide_labels=False,  # hide labels
#         hide_conf=False,  # hide confidences
#         half=False,  # use FP16 half-precision inference
#         dnn=False,  # use OpenCV DNN for ONNX inferenceqq
# ):
#     source = str(source)
#     save_img = not nosave and not source.endswith('.txt')  # save inference images
#     is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
#     is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
#     webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
#     if is_url and is_file:
#         source = check_file(source)  # download

#     # Directories
#     save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#     # Load model
#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
#     stride, names, pt = model.stride, model.names, model.pt
#     imgsz = check_img_size(imgsz, s=stride)  # check image size



#     # Dataloader
#     if webcam:
#         view_img = check_imshow()
#         cudnn.benchmark = True  # set True to speed up constant image size inference
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
#         bs = len(dataset)  # batch_size
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
#         bs = 1  # batch_size
#     vid_path, vid_writer = [None] * bs, [None] * bs

    

#     # Run inference
#     model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
#     dt, seen = [0.0, 0.0, 0.0], 0
#     for path, im, im0s, vid_cap, s in dataset:
#         start_time = time() # kwj_fps
#         t1 = time_sync()
#         im = torch.from_numpy(im).to(device)
#         im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#         im /= 255  # 0 - 255 to 0.0 - 1.0
#         if len(im.shape) == 3:
#             im = im[None]  # expand for batch dim
#         t2 = time_sync()
#         dt[0] += t2 - t1

#         # Inference
#         visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#         pred = model(im, augment=augment, visualize=visualize)
#         t3 = time_sync()
#         dt[1] += t3 - t2

#         # NMS
#         pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#         dt[2] += time_sync() - t3

#         # Second-stage classifier (optional)
#         # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
#         active_buttons = button.active_buttons_list() #Button

#         # Process predictions
#         for i, det in enumerate(pred):  # per image
#             seen += 1
#             if webcam:  # batch_size >= 1
#                 p, im0, frame = path[i], im0s[i].copy(), dataset.count
#                 s += f'{i}: '
#             else:
#                 p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 1)

#             p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # im.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
#             s += '%gx%g ' % im.shape[2:]  # print string #창의 크기 출력
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             imc = im0.copy() if save_crop else im0  # for save_crop
#             annotator = Annotator(im0, line_width=line_thickness, example=str(names))
#             cv2.putText(im0, str(frame), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2) # 몇 번째 frame인지(frame)

#             end_time = time() # kwj_fps
#             fps = 1 / np.round(end_time - start_time, 2) #kwj_fps
#             cv2.putText(im0, f'FPS: {int(fps)}', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2) #kwj_fps



#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                

#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    


#                 # Get active buttons list
#                 #active_buttons = button.active_buttons_list()
#                 #print("Active buttons", active_buttons)

#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                         with open(txt_path + '.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')

#                     if save_img or save_crop or view_img:  # Add bbox to image
#                         c = int(cls)  # integer class
#                         #label = None if hiqde_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') 
#                         #annotator.box_label(xyxy, label, color=colors(c, True))
                        
#                         # fps = 1 / np.round(end_time - start_time, 2) #kwj_fps
#                         # cv2.putText(im0, f'FPS: {int(fps)}', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2) #kwj_fps

#                         # # node 프레임마다 감지 
#                         # count = 0
#                         # node_point = 5


                        

#                         # if ((int(frame)+1) % node_point == 0 ):
#                         #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') # 라벨링
#                         #     annotator.box_label(xyxy, label, color=colors(c, True)) # 박스 그리기
#                         #     LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)') #kwj 몇개, 탐지 시간 출력
#                         #     cv2.putText(im0, f'{s}Done',(150,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # kwj_total_wound_per_frame
#                         #     count += 1

#                         #     #print("n_type :", n.dtype)
#                         #     #print("n :", n)
#                         #     node_frame_wound = n.tolist()
#                         #     #print("node_frame_wound : ", type(node_frame_wound))
                            
#                         #     # wound_que.append(node_frame_wound)

#                         #     if (node_frame_wound == 0):
#                         #         wound_que.append[(0)]
#                         #     else:
#                         #         wound_que.append(node_frame_wound)


#                         #     print("---------길이------------ length of list :", len(wound_que))
#                         #     print("frame_wound : ", wound_que)
                            

#                         # # [5frame_wound, 10frame_wound, 15frame_wound 20frame_wound]
#                         # if ((int(frame)+1) % 20 == 0):
                            
#                         #     wound_per_20frame = sum(wound_que)
#                         #     print("-------20프레임합계----------- 20frame_sum_total :", wound_per_20frame)
#                         #     cv2.putText(im0, f'20frame_wound : {int(wound_per_20frame)}', (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,200), 2) # kwj_sum_wound for 20frame
#                         #     #print(type(frame_wound))
#                         #     if (len(wound_que) != 0):
#                         #         wound_que.clear() # que초기화
#                         #     print("------초기화------reset_wound: ", wound_que)


#                         #     #품질
#                         #     if (wound_per_20frame <= 3):
#                         #         print("-------------상------------- 총 개수 : ", wound_per_20frame)
#                         #         cv2.putText(im0, f'BEST', (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4) # 상

#                         #     elif (wound_per_20frame <= 5):
#                         #         print("-------------중------------- 총 개수 : ", wound_per_20frame)
#                         #         cv2.putText(im0, f'BETTER', (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4) # 중
                            
#                         #     else:
#                         #         print("-------------하-------------총 개수 : ", wound_per_20frame)
#                         #         cv2.putText(im0, f'GOOD', (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4) # 하



                            
#                             #0914
#                             # print("n_type :", n.dtype)
#                             # print("n :", n)

#                             # n_kwj = n.tolist()

#                             # print("n_kwj_type : ", type(n_kwj))
#                             # print("n_kwj : ", n_kwj)

#                             # frame_wound = [n_kwj]
#                             # print("frame_wound_1 : ", type(frame_wound))
#                             # #frame_wound.append(n_kwj)
#                             # #print("frame_wound_2 : ", type(frame_wound))
#                             # print("frame_wound : ", frame_wound)
#                             # # print(len(frame_wound))
                            
                            

#                             #total_wound = [n_kwj] #kwj_total wound

#                             # if (count % 5 == 0):
#                             #     total_wound.append(n_kwj(0))
#                             #     print(total_wound)
                                
#                             # elif (count % 10 == 0):
#                             #     total_wound.append(n_kwj(0))
                                
#                             # total_wound.append(total_wound(0))
#                             #real_total = sum(total_wound)
#                             #print(real_total)
#                             # LOGGER.info(f'{int(real_total)}')
                                
                                

#                         # Button
#                         if names[c] in active_buttons:
                            
                            
#                             #kwj_origin
#                             # #c = int(cls)  # integer class
#                             # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') # 라벨링
#                             # annotator.box_label(xyxy, label, color=colors(c, True)) # 박스 그리기
#                             # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)') #kwj
#                             # cv2.putText(im0, f'{s}Done',(150,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # kwj_total_wound_per_frame

#                             # 20 frame
#                             # count = 0 
#                             # if (int(frame) % 20 == 0):
#                             #     print('Saved frame number : ' + str(int(frame)))
#                             #     cv2.imwrite("images/20frame/frame%d.jpg" % count, im0)
#                             #     print('Saved frame%d.jpg' % count)
#                             #     count += 1


#                             # node 프레임마다 감지 
#                             count = 0
#                             node_point = 5


#                             if ((int(frame)+1) % node_point == 0 ):
#                                 label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') # 라벨링
#                                 annotator.box_label(xyxy, label, color=colors(c, True)) # 박스 그리기
#                                 LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)') #kwj 몇개, 탐지 시간 출력
#                                 cv2.putText(im0, f'{s}Done',(150,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # kwj_total_wound_per_frame
#                                 count += 1

#                                 #print("n_type :", n.dtype)
#                                 #print("n :", n)
#                                 node_frame_wound = n.tolist()
#                                 #print("node_frame_wound : ", type(node_frame_wound))
                                
#                                 # wound_que.append(node_frame_wound)

#                                 if (node_frame_wound == 0):
#                                     wound_que.append[(0)]
#                                 else:
#                                     wound_que.append(node_frame_wound)


#                                 print("---------길이------------ length of list :", len(wound_que))
#                                 print("frame_wound : ", wound_que)
                                

#                             # [5frame_wound, 10frame_wound, 15frame_wound 20frame_wound]
#                             if ((int(frame)+1) % 20 == 0):
                                
#                                 wound_per_20frame = sum(wound_que)
#                                 print("-------20프레임합계----------- 20frame_sum_total :", wound_per_20frame)
#                                 cv2.putText(im0, f'20frame_wound : {int(wound_per_20frame)}', (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,200), 2) # kwj_sum_wound for 20frame
#                                 #print(type(frame_wound))
#                                 if (len(wound_que) != 0):
#                                     wound_que.clear() # que초기화
#                                 print("------초기화------reset_wound: ", wound_que)

                                
#                                 #품질
#                                 if (wound_per_20frame <= 3):
#                                     print("-------------상------------- 총 개수 : ", wound_per_20frame)
#                                     # 일정 프레임동안 화면상에 표시하고 싶음
#                                     cv2.putText(im0, f'BEST', (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4) # 상

#                                 elif (wound_per_20frame <= 5):
#                                     print("-------------중------------- 총 개수 : ", wound_per_20frame)
#                                     cv2.putText(im0, f'BETTER', (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4) # 중
                                
#                                 else:
#                                     print("-------------하-------------총 개수 : ", wound_per_20frame)
#                                     cv2.putText(im0, f'GOOD', (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4) # 하


#                             if save_crop:
#                                 save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
    
                        
            
#             #Stream results
#             im0 = annotator.result()
#             if view_img:
#                 #cv2.imshow(str(p), im0)

#                 # # Get active buttons list
#                 # active_buttons = button.active_buttons_list()
#                 # print("Active buttons", active_buttons)

                
                
#                 # Display buttons
#                 button.display_buttons(im0)
#                 cv2.namedWindow("woojin")
#                 cv2.setMouseCallback("woojin",click_button)
#                 cv2.imshow("woojin",im0)

#                 cv2.waitKey(1)  # 1 millisecond
                

#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'image':
#                     cv2.imwrite(save_path, im0)
#                 else:  # 'video' or 'stream'
#                     if vid_path[i] != save_path:  # new video
#                         vid_path[i] = save_path
#                         if isinstance(vid_writer[i], cv2.VideoWriter):
#                             vid_writer[i].release()  # release previous video writer
#                         if vid_cap:  # video
#                             #fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             fps = frame
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = fps, im0.shape[1], im0.shape[0]  #kwj_fps_save
#                         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
#                         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                     vid_writer[i].write(im0)


#         # Print time (inference-only)
#         print("frame : " , frame)
#         print("Active buttons", active_buttons) 
#         # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
               
               
    
        
        

#     # Print results
#     t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
#     LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
#     if save_txt or save_img:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
#     if update:
#         strip_optimizer(weights)  # update model (to fix SourceChangeWarning).


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
#     #parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob, 0 for webcam')
#     parser.add_argument('--source', type=str, default= cv2.CAP_DSHOW + 0, help='file/dir/URL/glob, 0 for webcam')
#     parser.add_argument('--data', type=str, default=ROOT / 'data/data.yaml', help='(optional) dataset.yaml path')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=7, help='maximum detections per image')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='show results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--visualize', action='store_true', help='visualize features')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#     print_args(vars(opt))
#     return opt

# a = 'finish' # 마지막 출력 문장 여기서 조정

# def main(opt):
#     #print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
#     check_requirements(exclude=('tensorboard', 'thop'))
#     run(**vars(opt))
#     return a


# #실행을 담당 # 마우스 눌렀을 때의 이벤트 추가 (ex. detect 껐다 켜기)
# if __name__ == "__main__": 
#     opt = parse_opt()
#     _main = main(opt)
#     print(_main)




# # 키보드 / 마우스를 눌렀을 때의 detecting 껐다가 키기
# # 그러기 위해서는 model(video)의 방법에 대해서 알고 있어야 함
# # model()의 인자? / 인자들은 어떻게 설정하나?


    


#     # def mouse_click(event):
#     #     if event == cv2.EVENT_FLAG_LBUTTON:
#     #         a = 1
#     #         hide_labels = True
#     #         hide_conf = True
#     #     elif event == cv2.EVENT_FLAG_RBUTTON:
#     #         a = 0
#     #         hide_labels = False
#     #         hide_conf = False
#     # cv2.setMouseCallback('0', mouse_click, run.im0)      

# #예시
# # while True:
# #     if keyboard.is_pressed("1"):
# #         print("hello")
# #         break


# #kwj
# # CAMERA_ID = 0

# # capture = cv2.VideoCapture(CAMERA_ID)
# # if capture.isOpened() == False: # 카메라 정상상태 확인
# #     print(f'Can\'t open the Camera({CAMERA_ID})')
# #     exit()

# # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
# # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
# # capture.set(cv2.CAP_PROP_FPS, 30)

# # prev_time = 0
# # total_frames = 0
# # start_time = time.time()
# # while cv2.waitKey(1) < 0:
# #     curr_time = time.time()

# #     ret, frame = capture.read()
# #     total_frames = total_frames + 1

# #     term = curr_time - prev_time
# #     fps = 1 / term
# #     prev_time = curr_time
# #     fps_string = f'term = {term:.3f},  FPS = {fps:.2f}'
# #     print(fps_string)

# #     cv2.putText(frame, fps_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
# #     cv2.imshow("VideoFrame", frame)

# # end_time = time.time()
# # fps = total_frames / (start_time - end_time)
# # print(f'total_frames = {tocd..tal_frames},  avg FPS = {fps:.2f}')

# # capture.release()
# # cv2.destroyAllWindows()



# # def mouse_click(event):
# #     if event == cv2.EVENT_FLAG_LBUTTON:
# #         hide_labels = True
# #         hide_conf = True
# #     elif event == cv2.EVENT_FLAG_RBUTTON:
# #         hide_labels = False
# #         hide_conf = False
# #cv2.setMouseCallback('0', mouse_click, im0)

# #Stream results # 캡쳐화면 q 
#             # im0 = annotator.result()
#             # if view_img:
#             #     while(True):
#             #         cv2.imshow(str(p), im0)
#             #         cv2.putText(im0, str(frame) , (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2)
#             #         #cv2.waitKey(1)  # 1 millisecond
#             #         if cv2.waitKey() == ord('q'):
#             #             break
#             #     cv2.destroyAllWindows() #kwj
















# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations_copy import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, is_colab, is_kaggle, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,}.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.new_video(path)
                ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, s

    def __len__(self):
        return 0


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl==2020.12.2'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0:
                assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
                assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 10  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into RAM/disk for faster training (WARNING: large datasets may exceed system resources)
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    gb += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.im_files),
                        bar_format=BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.im_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        else:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4, labels4 = random_perspective(img4,
                                           labels4,
                                           segments4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4

    def load_mosaic9(self, index):
        # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9 = random_perspective(img9,
                                           labels9,
                                           segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                   align_corners=False)[0].type(img[i].type())
                lb = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path=DATASETS_DIR / 'coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(str(path) + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / 'coco128'):  # from utils.dataloaders import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path=DATASETS_DIR / 'coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.dataloaders import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.dataloaders import *; dataset_stats('path/to/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def _round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def _find_yaml(dir):
        # Return data.yaml file
        files = list(dir.glob('*.yaml')) or list(dir.rglob('*.yaml'))  # try root level first and then recursive
        assert files, f'No *.yaml file found in {dir}'
        if len(files) > 1:
            files = [f for f in files if f.stem == dir.stem]  # prefer *.yaml files that match dir name
            assert files, f'Multiple *.yaml files found in {dir}, only 1 *.yaml file allowed'
        assert len(files) == 1, f'Multiple *.yaml files found: {files}, only 1 *.yaml file allowed in {dir}'
        return files[0]

    def _unzip(path):
        # Unzip data.zip
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix('')  # dataset directory == zip name
            assert dir.is_dir(), f'Error unzipping {path}, {dir} not found. path/to/abc.zip MUST unzip to path/to/abc/'
            return True, str(dir), _find_yaml(dir)  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def _hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=75, optimize=True)  # save
        except Exception as e:  # use OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = _unzip(Path(path))
    try:
        with open(check_yaml(yaml_path), errors='ignore') as f:
            data = yaml.safe_load(f)  # data dict
            if zipped:
                data['path'] = data_dir  # TODO: should this be dir.resolve()?`
    except Exception:
        raise Exception("error/HUB/dataset_stats/yaml_load")

    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  # shape(128x80)
        stats[split] = {
            'instance_stats': {
                'total': int(x.sum()),
                'per_class': x.sum(0).tolist()},
            'image_stats': {
                'total': dataset.n,
                'unlabelled': int(np.all(x == 0, 1).sum()),
                'per_class': (x > 0).sum(0).tolist()},
            'labels': [{
                str(Path(k).name): _round_labels(v.tolist())} for k, v in zip(dataset.im_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(_hub_ops, dataset.im_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats








