pip install torch torchvision opencv-python numpy scipy scikit-image albumentations pydicom SimpleITK
import os
import cv2
import numpy as np
import pydicom
from torch.utils.data import Dataset


class MRIDataset(Dataset):

    def __init__(self, image_dir, mask_dir=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = os.path.join(
            self.image_dir,
            self.images[idx]
        )

        if image_path.endswith(".dcm"):
            image = pydicom.dcmread(
                image_path
            ).pixel_array
        else:
            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

        image = image.astype(np.float32)
import cv2
import numpy as np


class MRIPreprocessor:

    def __init__(self,
                 target_size=(256,256)):

        self.target_size = target_size

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )

    def slice_standardization(self,
                              image):

        image = cv2.resize(
            image,
            self.target_size
        )

        return image

    def zscore_normalization(self,
                             image):

        mean = image.mean()
        std = image.std()

        image = (image - mean) / (std + 1e-8)

        return image

    def clahe_enhancement(self,
                          image):

        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        image = image.astype(np.uint8)

        image = self.clahe.apply(image)

        return image

    def process(self, image):

        image = self.slice_standardization(
            image
        )

        image = self.zscore_normalization(
            image
        )

        image = self.clahe_enhancement(
            image
        )

        return image
import albumentations as A


mri_aug = A.Compose([

    A.Rotate(
        limit=20,
        p=0.5
    ),

    A.Affine(
        scale=(0.9,1.1),
        p=0.5
    ),

    A.ElasticTransform(
        alpha=50,
        sigma=5,
        p=0.3
    ),

    A.RandomBrightnessContrast(
        p=0.5
    )

])
import cv2
import os


def extract_frames(
        video_path,
        save_folder,
        fps=10):

    os.makedirs(
        save_folder,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        video_path
    )

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    interval = int(
        video_fps / fps
    )

    count = 0
    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if count % interval == 0:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            cv2.imwrite(
                f"{save_folder}/{frame_id}.png",
                gray
            )

            frame_id += 1

        count += 1

    cap.release()
from scipy.ndimage import gaussian_filter
import numpy as np


def anisotropic_diffusion(
        img,
        iterations=10,
        kappa=50,
        gamma=0.1):

    img = img.astype(np.float32)

    for i in range(iterations):

        north = np.roll(
            img,
            -1,
            axis=0
        ) - img

        south = np.roll(
            img,
            1,
            axis=0
        ) - img

        east = np.roll(
            img,
            -1,
            axis=1
        ) - img

        west = np.roll(
            img,
            1,
            axis=1
        ) - img

        cN = np.exp(
            -(north/kappa)**2
        )

        cS = np.exp(
            -(south/kappa)**2
        )

        cE = np.exp(
            -(east/kappa)**2
        )

        cW = np.exp(
            -(west/kappa)**2
        )

        img += gamma * (
            cN*north +
            cS*south +
            cE*east +
            cW*west
        )

    return img
def minmax_normalization(img):

    img = img.astype(np.float32)

    img = (
        img - img.min()
    ) / (
        img.max() - img.min() + 1e-8
    )

    return img
import albumentations as A

echo_aug = A.Compose([

    A.Affine(
        scale=(0.9,1.1),
        rotate=(-15,15),
        p=0.5
    ),

    A.RandomBrightnessContrast(
        p=0.5
    ),

    A.GaussNoise(
        p=0.4
    )

])
def preprocess_echo(frame):

    frame = cv2.resize(
        frame,
        (256,256)
    )

    frame = anisotropic_diffusion(
        frame
    )

    frame = minmax_normalization(
        frame
    )

    return frame
        if self.mask_dir is not None:

            mask_path = os.path.join(
                self.mask_dir,
                self.images[idx]
            )

            mask = cv2.imread(
                mask_path,
                cv2.IMREAD_GRAYSCALE
            )

            mask = (mask > 0).astype(np.float32)

            return image, mask

        return image
pip install timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.swin_transformer import SwinTransformer
class ResidualBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.skip = nn.Sequential()

        if in_channels != out_channels:

            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1
                ),
                nn.BatchNorm2d(
                    out_channels
                )
            )

    def forward(self,x):

        identity = self.skip(x)

        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = self.bn2(
            self.conv2(x)
        )

        x += identity

        return F.relu(x)
class SharedEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.e1 = ResidualBlock(1,64)

        self.e2 = ResidualBlock(64,128)

        self.e3 = ResidualBlock(128,256)

        self.e4 = ResidualBlock(256,512)

        self.e5 = ResidualBlock(512,1024)

        self.pool = nn.MaxPool2d(2)

    def forward(self,x):

        f1 = self.e1(x)

        f2 = self.e2(
            self.pool(f1)
        )

        f3 = self.e3(
            self.pool(f2)
        )

        f4 = self.e4(
            self.pool(f3)
        )

        f5 = self.e5(
            self.pool(f4)
        )

        return f1,f2,f3,f4,f5
class MRITransformerBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.transformer = SwinTransformer(
            img_size=16,
            patch_size=1,
            in_chans=1024,
            embed_dim=256,
            depths=(2,2),
            num_heads=(4,8)
        )

    def forward(self,x):

        b,c,h,w = x.shape

        x = self.transformer.patch_embed(x)

        for layer in self.transformer.layers:

            x = layer(x)

        return x
class TemporalAttention(nn.Module):

    def __init__(self,
                 channels):

        super().__init__()

        self.query = nn.Linear(
            channels,
            channels
        )

        self.key = nn.Linear(
            channels,
            channels
        )

        self.value = nn.Linear(
            channels,
            channels
        )

    def forward(self,x):

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        score = torch.softmax(
            torch.matmul(
                Q,
                K.transpose(-2,-1)
            ) /
            (Q.shape[-1]**0.5),
            dim=-1
        )

        out = torch.matmul(
            score,
            V
        )

        return out
class EchoBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.src = nn.Conv2d(
            1024,
            1024,
            3,
            padding=1
        )

        self.temporal = TemporalAttention(
            1024
        )

    def forward(self,x):

        x = self.src(x)

        b,c,h,w = x.shape

        x = x.view(
            b,
            c,
            h*w
        ).transpose(1,2)

        x = self.temporal(x)

        return x
class CSAAM(nn.Module):

    def __init__(self,
                 channels):

        super().__init__()

        self.cavity = nn.Sequential(
            nn.Conv2d(channels,
                      channels,
                      1),
            nn.Sigmoid()
        )

        self.myo = nn.Sequential(
            nn.Conv2d(channels,
                      channels,
                      1),
            nn.Sigmoid()
        )

        self.boundary = nn.Sequential(
            nn.Conv2d(channels,
                      channels,
                      1),
            nn.Sigmoid()
        )

    def forward(self,x):

        cavity = x * self.cavity(x)

        myo = cavity * self.myo(cavity)

        boundary = myo * self.boundary(myo)

        return boundary
class DecoderBlock(nn.Module):

    def __init__(self,
                 in_ch,
                 skip_ch,
                 out_ch):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            2,
            stride=2
        )

        self.conv = nn.Sequential(

            nn.Conv2d(
                out_ch + skip_ch,
                out_ch,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_ch
            ),

            nn.ReLU(),

            nn.Conv2d(
                out_ch,
                out_ch,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_ch
            ),

            nn.ReLU()

        )

    def forward(self,x,skip):

        x = self.up(x)

        x = torch.cat(
            [x,skip],
            dim=1
        )

        return self.conv(x)
class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.d4 = DecoderBlock(
            1024,
            512,
            512
        )

        self.d3 = DecoderBlock(
            512,
            256,
            256
        )

        self.d2 = DecoderBlock(
            256,
            128,
            128
        )

        self.d1 = DecoderBlock(
            128,
            64,
            64
        )

        self.final = nn.Conv2d(
            64,
            1,
            1
        )

    def forward(self,
                x,
                f1,f2,f3,f4):

        x = self.d4(x,f4)

        x = self.d3(x,f3)

        x = self.d2(x,f2)

        x = self.d1(x,f1)

        return torch.sigmoid(
            self.final(x)
        )
class MATSNet(nn.Module):

    def __init__(self,
                 modality="mri"):

        super().__init__()

        self.modality = modality

        self.encoder = SharedEncoder()

        self.mri_branch = MRITransformerBranch()

        self.echo_branch = EchoBranch()

        self.csaam = CSAAM(
            1024
        )

        self.decoder = Decoder()

    def forward(self,x):

        f1,f2,f3,f4,f5 = self.encoder(x)

        if self.modality == "mri":

            bottleneck = self.mri_branch(f5)

        else:

            bottleneck = self.echo_branch(f5)

        bottleneck = bottleneck.reshape(
            f5.shape
        )

        bottleneck = self.csaam(
            bottleneck
        )

        mask = self.decoder(
            bottleneck,
            f1,f2,f3,f4
        )

        return mask
class DiceLoss(nn.Module):

    def forward(self,
                pred,
                target):

        smooth = 1

        pred = pred.view(-1)

        target = target.view(-1)

        inter = (
            pred * target
        ).sum()

        dice = (
            2*inter + smooth
        ) / (
            pred.sum() +
            target.sum() +
            smooth
        )

        return 1-dice
class HybridLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.dice = DiceLoss()

        self.bce = nn.BCELoss()

    def forward(self,
                pred,
                target):

        d = self.dice(
            pred,
            target
        )

        b = self.bce(
            pred,
            target
        )

        return d + b
# myocardial_refinement.py

import cv2
import numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from scipy import ndimage


# ==========================================================
# ACTIVE CONTOUR REFINEMENT
# ==========================================================

class ActiveContourRefinement:

    def __init__(
            self,
            alpha=0.015,
            beta=10,
            gamma=0.001):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def refine(self, image, mask):

        image = gaussian(image, sigma=3)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if len(contours) == 0:
            return mask

        contour = max(
            contours,
            key=cv2.contourArea
        )

        contour = contour.squeeze()

        if len(contour.shape) < 2:
            return mask

        snake = active_contour(
            image,
            contour,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma
        )

        refined = np.zeros_like(mask)

        snake = snake.astype(np.int32)

        cv2.fillPoly(
            refined,
            [snake],
            1
        )

        return refined.astype(np.uint8)


# ==========================================================
# MORPHOLOGICAL CLEANUP
# ==========================================================

class MorphologicalRefinement:

    def __init__(self):

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

    def process(self, mask):

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel
        )

        mask = cv2.dilate(
            mask,
            self.kernel,
            iterations=1
        )

        return mask


# ==========================================================
# GRAPH CUT OPTIMIZATION
# ==========================================================

class GraphCutOptimization:

    def optimize(
            self,
            image,
            mask):

        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        gc_mask = np.where(
            mask > 0,
            cv2.GC_PR_FGD,
            cv2.GC_BGD
        ).astype("uint8")

        bgdModel = np.zeros(
            (1, 65),
            np.float64
        )

        fgdModel = np.zeros(
            (1, 65),
            np.float64
        )

        rect = (
            1,
            1,
            image.shape[1]-2,
            image.shape[0]-2
        )

        try:

            cv2.grabCut(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2BGR
                ),
                gc_mask,
                rect,
                bgdModel,
                fgdModel,
                5,
                cv2.GC_INIT_WITH_MASK
            )

            result = np.where(
                (gc_mask == 2) |
                (gc_mask == 0),
                0,
                1
            )

            return result.astype(np.uint8)

        except:
            return mask


# ==========================================================
# LARGEST CONNECTED COMPONENT
# ==========================================================

class LargestComponent:

    def keep_largest(
            self,
            mask):

        lbl = label(mask)

        props = regionprops(lbl)

        if len(props) == 0:
            return mask

        largest = max(
            props,
            key=lambda x: x.area
        )

        final = (
            lbl == largest.label
        ).astype(np.uint8)

        return final


# ==========================================================
# AHA 17 SEGMENT PARTITION
# ==========================================================

class AHA17SegmentMapper:

    def __init__(self):

        self.num_segments = 17

    def create_segments(
            self,
            myocardium_mask):

        h, w = myocardium_mask.shape

        center_y, center_x = ndimage.center_of_mass(
            myocardium_mask
        )

        yy, xx = np.mgrid[0:h, 0:w]

        angles = np.arctan2(
            yy - center_y,
            xx - center_x
        )

        angles = (
            np.degrees(angles) + 360
        ) % 360

        segment_map = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        # Basal ring
        basal_mask = (
            myocardium_mask > 0
        )

        radius = np.sqrt(
            (yy-center_y)**2 +
            (xx-center_x)**2
        )

        max_radius = radius[
            myocardium_mask > 0
        ].max()

        r_norm = radius / (
            max_radius + 1e-8
        )

        # Basal 1-6
        basal = (
            basal_mask &
            (r_norm > 0.66)
        )

        for i in range(6):

            start = i * 60
            end = (i + 1) * 60

            region = (
                basal &
                (angles >= start) &
                (angles < end)
            )

            segment_map[
                region
            ] = i + 1

        # Mid 7-12
        mid = (
            basal_mask &
            (r_norm > 0.33) &
            (r_norm <= 0.66)
        )

        for i in range(6):

            start = i * 60
            end = (i + 1) * 60

            region = (
                mid &
                (angles >= start) &
                (angles < end)
            )

            segment_map[
                region
            ] = i + 7

        # Apical 13-16
        apex = (
            basal_mask &
            (r_norm <= 0.33)
        )

        for i in range(4):

            start = i * 90
            end = (i + 1) * 90

            region = (
                apex &
                (angles >= start) &
                (angles < end)
            )

            segment_map[
                region
            ] = i + 13

        # Segment 17
        center_region = (
            radius <
            max_radius * 0.08
        )

        segment_map[
            center_region
        ] = 17

        return segment_map


# ==========================================================
# COMPLETE REFINEMENT PIPELINE
# ==========================================================

class MyocardialRefinementPipeline:

    def __init__(self):

        self.active = ActiveContourRefinement()

        self.morphology = MorphologicalRefinement()

        self.graphcut = GraphCutOptimization()

        self.largest = LargestComponent()

        self.aha = AHA17SegmentMapper()

    def process(
            self,
            image,
            segmentation_mask):

        mask = self.active.refine(
            image,
            segmentation_mask
        )

        mask = self.morphology.process(
            mask
        )

        mask = self.graphcut.optimize(
            image,
            mask
        )

        mask = self.largest.keep_largest(
            mask
        )

        segments = self.aha.create_segments(
            mask
        )

        return mask, segments


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    image = cv2.imread(
        "image.png",
        0
    )

    mask = cv2.imread(
        "mask.png",
        0
    )

    mask = (
        mask > 0
    ).astype(np.uint8)

    pipeline = MyocardialRefinementPipeline()

    refined_mask, aha_segments = pipeline.process(
        image,
        mask
    )

    cv2.imwrite(
        "refined_mask.png",
        refined_mask * 255
    )

    cv2.imwrite(
        "aha17_segments.png",
        aha_segments
    )

    print("Step 3 Completed")
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from radiomics import featureextractor


# ==========================================================
# MRI STRUCTURAL FEATURE EXTRACTION
# DenseNet121
# ==========================================================

class MRI_DenseNet_Features(nn.Module):

    def __init__(self):

        super().__init__()

        densenet = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        self.features = densenet.features

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):

        x = self.features(x)

        x = self.pool(x)

        x = x.view(
            x.size(0),
            -1
        )

        return x


# ==========================================================
# MRI RADIOMICS FEATURES
# ==========================================================

class RadiomicsFeatures:

    def __init__(self):

        self.extractor = (
            featureextractor.RadiomicsFeatureExtractor()
        )

    def extract(
            self,
            image_path,
            mask_path):

        result = self.extractor.execute(
            image_path,
            mask_path
        )

        features = []

        for key,val in result.items():

            if isinstance(
                    val,
                    (int,float)
            ):

                features.append(float(val))

        return np.array(
            features,
            dtype=np.float32
        )


# ==========================================================
# MRI FEATURE FUSION
# ==========================================================

class MRIFeatureFusion(nn.Module):

    def __init__(
            self,
            radiomics_dim=100):

        super().__init__()

        self.fc = nn.Linear(
            1024 + radiomics_dim,
            512
        )

    def forward(
            self,
            dense_feat,
            radio_feat):

        x = torch.cat(
            [
                dense_feat,
                radio_feat
            ],
            dim=1
        )

        return self.fc(x)


# ==========================================================
# CNN SPATIAL ENCODER
# ==========================================================

class SpatialCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self,x):

        x = self.cnn(x)

        x = x.view(
            x.size(0),
            -1
        )

        return x


# ==========================================================
# CNN-LSTM
# ==========================================================

class EchoCNNLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = SpatialCNN()

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self,video):

        B,T,C,H,W = video.shape

        feats = []

        for t in range(T):

            f = self.cnn(
                video[:,t]
            )

            feats.append(f)

        feats = torch.stack(
            feats,
            dim=1
        )

        out,_ = self.lstm(feats)

        return out[:,-1]
        

# ==========================================================
# MOTION DESCRIPTORS
# ==========================================================

class MotionFeatures:

    def compute(
            self,
            frames):

        motion = []

        for i in range(
                len(frames)-1):

            flow = cv2.calcOpticalFlowFarneback(
                frames[i],
                frames[i+1],
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0
            )

            mag,_ = cv2.cartToPolar(
                flow[...,0],
                flow[...,1]
            )

            motion.append([
                np.mean(mag),
                np.std(mag),
                np.max(mag)
            ])

        motion = np.array(
            motion
        )

        return np.mean(
            motion,
            axis=0
        )


# ==========================================================
# ECHO FEATURE FUSION
# ==========================================================

class EchoFeatureFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Linear(
            512 + 3,
            512
        )

    def forward(
            self,
            temporal_feat,
            motion_feat):

        x = torch.cat(
            [
                temporal_feat,
                motion_feat
            ],
            dim=1
        )

        return self.fc(x)


# ==========================================================
# MODALITY PROJECTION
# ==========================================================

class ModalityProjection(nn.Module):

    def __init__(self):

        super().__init__()

        self.proj = nn.Sequential(

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                256
            )
        )

    def forward(self,x):

        return self.proj(x)


# ==========================================================
# COMPLETE MRI PIPELINE
# ==========================================================

class MRIFeatureExtractor(nn.Module):

    def __init__(
            self,
            radiomics_dim=100):

        super().__init__()

        self.densenet = (
            MRI_DenseNet_Features()
        )

        self.fusion = (
            MRIFeatureFusion(
                radiomics_dim
            )
        )

        self.project = (
            ModalityProjection()
        )

    def forward(
            self,
            image,
            radiomics):

        dense = self.densenet(
            image
        )

        fused = self.fusion(
            dense,
            radiomics
        )

        emb = self.project(
            fused
        )

        return emb


# ==========================================================
# COMPLETE ECHO PIPELINE
# ==========================================================

class EchoFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        self.temporal = (
            EchoCNNLSTM()
        )

        self.fusion = (
            EchoFeatureFusion()
        )

        self.project = (
            ModalityProjection()
        )

    def forward(
            self,
            video,
            motion):

        temporal = (
            self.temporal(
                video
            )
        )

        fused = self.fusion(
            temporal,
            motion
        )

        emb = self.project(
            fused
        )

        return emb


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    mri_model = (
        MRIFeatureExtractor()
    )

    echo_model = (
        EchoFeatureExtractor()
    )

    mri_img = torch.randn(
        2,3,224,224
    )

    radio = torch.randn(
        2,100
    )

    echo_video = torch.randn(
        2,20,1,224,224
    )

    motion = torch.randn(
        2,3
    )

    mri_emb = mri_model(
        mri_img,
        radio
    )

    echo_emb = echo_model(
        echo_video,
        motion
    )

    print(
        mri_emb.shape
    )

    print(
        echo_emb.shape
    )
# cdcg_kan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================================
# GRAPH CONSTRUCTION
# ==========================================================

class CardiacGraphBuilder:

    def __init__(self,
                 num_nodes=17):

        self.num_nodes = num_nodes

    def spatial_adjacency(self):

        A = np.zeros(
            (
                self.num_nodes,
                self.num_nodes
            ),
            dtype=np.float32
        )

        for i in range(
                self.num_nodes):

            if i > 0:
                A[i,i-1] = 1

            if i < self.num_nodes-1:
                A[i,i+1] = 1

        np.fill_diagonal(
            A,
            1
        )

        return A

    def functional_adjacency(
            self,
            features):

        features = (
            features /
            (
                np.linalg.norm(
                    features,
                    axis=1,
                    keepdims=True
                ) + 1e-8
            )
        )

        A = np.dot(
            features,
            features.T
        )

        return A.astype(
            np.float32
        )

    def coupled_graph(
            self,
            structural,
            functional,
            alpha=0.5):

        return (
            alpha*structural +
            (1-alpha)*functional
        )


# ==========================================================
# KAN LAYER
# ==========================================================

class KANLayer(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim):

        super().__init__()

        self.linear = nn.Linear(
            in_dim,
            out_dim
        )

        self.spline = nn.Sequential(

            nn.Linear(
                out_dim,
                out_dim
            ),

            nn.Tanh(),

            nn.Linear(
                out_dim,
                out_dim
            )
        )

    def forward(self,x):

        x = self.linear(x)

        x = self.spline(x)

        return x


# ==========================================================
# GRAPH KAN CONV
# ==========================================================

class GraphKANConv(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim):

        super().__init__()

        self.kan = KANLayer(
            in_dim,
            out_dim
        )

    def forward(
            self,
            X,
            A):

        A = A + torch.eye(
            A.size(0),
            device=A.device
        )

        D = torch.diag(
            torch.pow(
                A.sum(1),
                -0.5
            )
        )

        A_hat = D @ A @ D

        X = A_hat @ X

        X = self.kan(X)

        return X


# ==========================================================
# MULTI-LAYER CDCG GRAPH
# ==========================================================

class CDCGGraphEncoder(nn.Module):

    def __init__(
            self,
            in_dim=256,
            hidden=256):

        super().__init__()

        self.g1 = GraphKANConv(
            in_dim,
            hidden
        )

        self.g2 = GraphKANConv(
            hidden,
            hidden
        )

        self.g3 = GraphKANConv(
            hidden,
            hidden
        )

    def forward(
            self,
            X,
            A):

        X = F.relu(
            self.g1(
                X,
                A
            )
        )

        X = F.relu(
            self.g2(
                X,
                A
            )
        )

        X = self.g3(
            X,
            A
        )

        return X


# ==========================================================
# MRI → ECHO ATTENTION
# ==========================================================

class MRIToEchoAttention(nn.Module):

    def __init__(
            self,
            dim=256):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=8,
            batch_first=True
        )

    def forward(
            self,
            mri,
            echo):

        out,_ = self.attn(
            query=echo,
            key=mri,
            value=mri
        )

        return out


# ==========================================================
# ECHO → MRI ATTENTION
# ==========================================================

class EchoToMRIAttention(nn.Module):

    def __init__(
            self,
            dim=256):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=8,
            batch_first=True
        )

    def forward(
            self,
            echo,
            mri):

        out,_ = self.attn(
            query=mri,
            key=echo,
            value=echo
        )

        return out


# ==========================================================
# CROSS DOMAIN FUSION
# ==========================================================

class CrossDomainFusion(nn.Module):

    def __init__(
            self,
            dim=256):

        super().__init__()

        self.mri2echo = (
            MRIToEchoAttention(
                dim
            )
        )

        self.echo2mri = (
            EchoToMRIAttention(
                dim
            )
        )

        self.proj = nn.Sequential(

            nn.Linear(
                dim*2,
                dim
            ),

            nn.ReLU(),

            nn.Linear(
                dim,
                dim
            )
        )

    def forward(
            self,
            mri,
            echo):

        att1 = self.mri2echo(
            mri,
            echo
        )

        att2 = self.echo2mri(
            echo,
            mri
        )

        fused = torch.cat(
            [
                att1.mean(1),
                att2.mean(1)
            ],
            dim=-1
        )

        return self.proj(
            fused
        )


# ==========================================================
# CONTRASTIVE ALIGNMENT
# ==========================================================

class ContrastiveLoss(nn.Module):

    def __init__(
            self,
            temperature=0.07):

        super().__init__()

        self.temperature = temperature

    def forward(
            self,
            z1,
            z2):

        z1 = F.normalize(
            z1,
            dim=1
        )

        z2 = F.normalize(
            z2,
            dim=1
        )

        sim = torch.mm(
            z1,
            z2.t()
        )

        sim = (
            sim /
            self.temperature
        )

        labels = torch.arange(
            z1.size(0),
            device=z1.device
        )

        loss = F.cross_entropy(
            sim,
            labels
        )

        return loss


# ==========================================================
# UNIFIED CARDIAC EMBEDDING
# ==========================================================

class UnifiedProjection(nn.Module):

    def __init__(
            self,
            dim=256):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                dim,
                dim
            ),

            nn.ReLU(),

            nn.Linear(
                dim,
                dim
            )
        )

    def forward(self,x):

        return self.net(x)


# ==========================================================
# COMPLETE CDCG-KAN
# ==========================================================

class CDCGKAN(nn.Module):

    def __init__(
            self,
            dim=256):

        super().__init__()

        self.graph_encoder = (
            CDCGGraphEncoder(
                dim,
                dim
            )
        )

        self.fusion = (
            CrossDomainFusion(
                dim
            )
        )

        self.project = (
            UnifiedProjection(
                dim
            )
        )

    def forward(
            self,
            mri_nodes,
            echo_nodes,
            A_mri,
            A_echo):

        Hm = self.graph_encoder(
            mri_nodes,
            A_mri
        )

        He = self.graph_encoder(
            echo_nodes,
            A_echo
        )

        fused = self.fusion(
            Hm.unsqueeze(0),
            He.unsqueeze(0)
        )

        cardiac_embedding = (
            self.project(
                fused
            )
        )

        return (
            cardiac_embedding,
            Hm,
            He
        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    model = CDCGKAN()

    mri_nodes = torch.randn(
        17,
        256
    )

    echo_nodes = torch.randn(
        17,
        256
    )

    A_mri = torch.eye(17)

    A_echo = torch.eye(17)

    zc, hm, he = model(
        mri_nodes,
        echo_nodes,
        A_mri,
        A_echo
    )

    print(
        "Cardiac Embedding:",
        zc.shape
    )

    print(
        "MRI Graph:",
        hm.shape
    )

    print(
        "Echo Graph:",
        he.shape
    )
# ==========================================================
# STEP 6 : HCRN-NET
# Hybrid Uncertainty-Aware Cardiac Risk Reasoning Network
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# GLOBAL RISK BRANCH (ViT)
# ==========================================================

class GlobalRiskBranch(nn.Module):

    def __init__(self,
                 input_dim=256):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            batch_first=True
        )

        self.vit = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        self.fc = nn.Linear(
            input_dim,
            256
        )

    def forward(self,x):

        x = x.unsqueeze(1)

        x = self.vit(x)

        x = x.mean(1)

        return self.fc(x)


# ==========================================================
# REGIONAL RISK BRANCH (GAT)
# ==========================================================

class GraphAttentionLayer(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim):

        super().__init__()

        self.W = nn.Linear(
            in_dim,
            out_dim
        )

        self.attn = nn.Linear(
            out_dim*2,
            1
        )

    def forward(self,
                X,
                A):

        H = self.W(X)

        N = H.size(0)

        scores = []

        for i in range(N):

            row = []

            for j in range(N):

                a = torch.cat(
                    [H[i],H[j]]
                )

                row.append(
                    self.attn(a)
                )

            scores.append(
                torch.stack(row)
            )

        scores = torch.stack(scores)

        scores = scores.squeeze(-1)

        scores = scores.masked_fill(
            A==0,
            -1e9
        )

        alpha = F.softmax(
            scores,
            dim=1
        )

        out = alpha @ H

        return out


class RegionalRiskBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.gat1 = GraphAttentionLayer(
            256,
            256
        )

        self.gat2 = GraphAttentionLayer(
            256,
            256
        )

    def forward(
            self,
            nodes,
            adjacency):

        x = F.relu(
            self.gat1(
                nodes,
                adjacency
            )
        )

        x = self.gat2(
            x,
            adjacency
        )

        return x.mean(0)


# ==========================================================
# TEMPORAL RISK BRANCH
# ==========================================================

class TemporalRiskBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            256,
            256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(
            512,
            256
        )

    def forward(self,x):

        x = x.unsqueeze(1)

        out,_ = self.lstm(x)

        out = out[:,-1]

        return self.fc(out)


# ==========================================================
# CRCM
# Cardiac Risk Coordination Module
# ==========================================================

class CRCM(nn.Module):

    def __init__(self):

        super().__init__()

        self.gr = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU()
        )

        self.rt = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU()
        )

        self.gt = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU()
        )

        self.final = nn.Sequential(

            nn.Linear(
                768,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                256
            )
        )

    def forward(
            self,
            global_feat,
            regional_feat,
            temporal_feat):

        gr = self.gr(
            torch.cat(
                [
                    global_feat,
                    regional_feat
                ],
                dim=-1
            )
        )

        rt = self.rt(
            torch.cat(
                [
                    regional_feat,
                    temporal_feat
                ],
                dim=-1
            )
        )

        gt = self.gt(
            torch.cat(
                [
                    global_feat,
                    temporal_feat
                ],
                dim=-1
            )
        )

        x = torch.cat(
            [gr,rt,gt],
            dim=-1
        )

        return self.final(x)


# ==========================================================
# DEEP EVIDENTIAL LEARNING
# ==========================================================

class EvidentialModule(nn.Module):

    def __init__(self):

        super().__init__()

        self.mean_head = nn.Linear(
            256,
            1
        )

        self.var_head = nn.Linear(
            256,
            1
        )

        self.conf_head = nn.Linear(
            256,
            1
        )

    def forward(self,x):

        mean = self.mean_head(x)

        variance = F.softplus(
            self.var_head(x)
        )

        confidence = torch.sigmoid(
            self.conf_head(x)
        )

        return (
            mean,
            variance,
            confidence
        )


# ==========================================================
# SURVIVAL PREDICTION
# ==========================================================

class SurvivalModule(nn.Module):

    def __init__(self):

        super().__init__()

        self.hazard = nn.Linear(
            256,
            1
        )

        self.time_event = nn.Linear(
            256,
            1
        )

    def forward(self,x):

        hazard = torch.exp(
            self.hazard(x)
        )

        survival = torch.exp(
            -hazard
        )

        tte = self.time_event(x)

        return (
            hazard,
            survival,
            tte
        )


# ==========================================================
# COX LOSS
# ==========================================================

class CoxLoss(nn.Module):

    def forward(
            self,
            risk,
            time,
            event):

        order = torch.argsort(
            time,
            descending=True
        )

        risk = risk[order]

        event = event[order]

        log_cumsum = torch.log(
            torch.cumsum(
                torch.exp(risk),
                dim=0
            )
        )

        loss = -torch.mean(
            event *
            (
                risk -
                log_cumsum
            )
        )

        return loss


# ==========================================================
# EVIDENTIAL LOSS
# ==========================================================

class EvidentialLoss(nn.Module):

    def forward(
            self,
            pred,
            target,
            variance):

        error = (
            pred-target
        )**2

        loss = (
            error/(variance+1e-6)
            +
            torch.log(
                variance+1e-6
            )
        )

        return loss.mean()


# ==========================================================
# COMPLETE HCRN-NET
# ==========================================================

class HCRNNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.global_branch = (
            GlobalRiskBranch()
        )

        self.regional_branch = (
            RegionalRiskBranch()
        )

        self.temporal_branch = (
            TemporalRiskBranch()
        )

        self.crcm = CRCM()

        self.evidential = (
            EvidentialModule()
        )

        self.survival = (
            SurvivalModule()
        )

    def forward(
            self,
            cardiac_embedding,
            graph_nodes,
            adjacency):

        global_feat = (
            self.global_branch(
                cardiac_embedding
            )
        )

        regional_feat = (
            self.regional_branch(
                graph_nodes,
                adjacency
            )
        )

        temporal_feat = (
            self.temporal_branch(
                cardiac_embedding
            )
        )

        coordinated = (
            self.crcm(
                global_feat,
                regional_feat,
                temporal_feat
            )
        )

        risk_mean, variance, confidence = (
            self.evidential(
                coordinated
            )
        )

        hazard, survival, tte = (
            self.survival(
                coordinated
            )
        )

        return {

            "risk_score":
                risk_mean,

            "variance":
                variance,

            "confidence":
                confidence,

            "hazard":
                hazard,

            "survival":
                survival,

            "time_to_event":
                tte,

            "global":
                global_feat,

            "regional":
                regional_feat,

            "temporal":
                temporal_feat,

            "coordinated":
                coordinated
        }


# ==========================================================
# COMPLETE PIPELINE
# MATS-NET + FEATURE EXTRACTION +
# CDCG-KAN + HCRN-NET
# ==========================================================

class FullCardiacFramework(nn.Module):

    def __init__(
            self,
            matsnet,
            mri_feature,
            echo_feature,
            cdcg_kan):

        super().__init__()

        self.matsnet = matsnet

        self.mri_feature = mri_feature

        self.echo_feature = echo_feature

        self.cdcg_kan = cdcg_kan

        self.hcrn = HCRNNet()

    def forward(
            self,
            image,
            mri_nodes,
            echo_nodes,
            A_mri,
            A_echo):

        segmentation = (
            self.matsnet(
                image
            )
        )

        cardiac_embedding, Hm, He = (
            self.cdcg_kan(
                mri_nodes,
                echo_nodes,
                A_mri,
                A_echo
            )
        )

        outputs = self.hcrn(
            cardiac_embedding,
            Hm,
            A_mri
        )

        outputs[
            "segmentation"
        ] = segmentation

        return outputs


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    cardiac_embedding = torch.randn(
        1,
        256
    )

    graph_nodes = torch.randn(
        17,
        256
    )

    adjacency = torch.eye(
        17
    )

    model = HCRNNet()

    outputs = model(
        cardiac_embedding,
        graph_nodes,
        adjacency
    )

    for k,v in outputs.items():

        if torch.is_tensor(v):
            print(
                k,
                v.shape
            )
# ==========================================================
# JOINT LOSS
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def forward(self,pred,target):

        smooth = 1.0

        pred = pred.view(-1)
        target = target.view(-1)

        inter = (pred*target).sum()

        dice = (
            2*inter + smooth
        ) / (
            pred.sum() +
            target.sum() +
            smooth
        )

        return 1-dice


class FocalTverskyLoss(nn.Module):

    def __init__(
            self,
            alpha=0.7,
            beta=0.3,
            gamma=0.75):

        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
            self,
            pred,
            target):

        pred = pred.view(-1)
        target = target.view(-1)

        TP = (
            pred*target
        ).sum()

        FP = (
            (1-target)*pred
        ).sum()

        FN = (
            target*(1-pred)
        ).sum()

        tversky = (
            TP + 1
        ) / (
            TP +
            self.alpha*FP +
            self.beta*FN +
            1
        )

        return (
            1-tversky
        ) ** self.gamma


class ShapeConsistencyLoss(nn.Module):

    def forward(
            self,
            pred):

        dx = torch.abs(
            pred[:,:,1:,:] -
            pred[:,:,:-1,:]
        )

        dy = torch.abs(
            pred[:,:,:,1:] -
            pred[:,:,:,:-1]
        )

        return (
            dx.mean() +
            dy.mean()
        )


class TotalSegmentationLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.dice = DiceLoss()

        self.focal = FocalTverskyLoss()

        self.bce = nn.BCELoss()

        self.shape = ShapeConsistencyLoss()

    def forward(
            self,
            pred,
            target):

        return (

            self.dice(
                pred,target
            )

            +

            self.focal(
                pred,target
            )

            +

            self.bce(
                pred,target
            )

            +

            0.1*self.shape(
                pred
            )

        )
# ==========================================================
# GRADCAM++
# ==========================================================

class GradCAMPlusPlus:

    def __init__(
            self,
            model,
            target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(
            self.forward_hook
        )

        target_layer.register_backward_hook(
            self.backward_hook
        )

    def forward_hook(
            self,
            module,
            input,
            output):

        self.activations = output

    def backward_hook(
            self,
            module,
            grad_in,
            grad_out):

        self.gradients = grad_out[0]

    def generate(
            self,
            image):

        output = self.model(
            image
        )

        score = output.max()

        self.model.zero_grad()

        score.backward()

        weights = (
            self.gradients.mean(
                dim=(2,3),
                keepdim=True
            )
        )

        cam = (
            weights *
            self.activations
        ).sum(1)

        cam = F.relu(cam)

        cam = cam.detach()

        return cam
# ==========================================================
# GRADCAM++
# ==========================================================

class GradCAMPlusPlus:

    def __init__(
            self,
            model,
            target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(
            self.forward_hook
        )

        target_layer.register_backward_hook(
            self.backward_hook
        )

    def forward_hook(
            self,
            module,
            input,
            output):

        self.activations = output

    def backward_hook(
            self,
            module,
            grad_in,
            grad_out):

        self.gradients = grad_out[0]

    def generate(
            self,
            image):

        output = self.model(
            image
        )

        score = output.max()

        self.model.zero_grad()

        score.backward()

        weights = (
            self.gradients.mean(
                dim=(2,3),
                keepdim=True
            )
        )

        cam = (
            weights *
            self.activations
        ).sum(1)

        cam = F.relu(cam)

        cam = cam.detach()

        return cam
# ==========================================================
# MC DROPOUT
# ==========================================================

def mc_dropout_predict(
        model,
        image,
        runs=30):

    model.train()

    preds = []

    with torch.no_grad():

        for _ in range(runs):

            pred = model(
                image
            )

            preds.append(
                pred
            )

    preds = torch.stack(
        preds
    )

    mean = preds.mean(0)

    variance = preds.var(0)

    return mean, variance
# ==========================================================
# SHAP
# ==========================================================

import shap

class SHAPExplainer:

    def __init__(
            self,
            model):

        self.model = model

    def explain(
            self,
            background,
            samples):

        explainer = shap.DeepExplainer(
            self.model,
            background
        )

        values = explainer.shap_values(
            samples
        )

        return values
# ==========================================================
# TRAINING LOOP
# ==========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

criterion = TotalSegmentationLoss()

for epoch in range(200):

    model.train()

    total_loss = 0

    for images,masks in train_loader:

        images = images.cuda()

        masks = masks.cuda()

        pred = model(images)

        loss = criterion(
            pred,
            masks
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    print(
        epoch,
        total_loss
    )