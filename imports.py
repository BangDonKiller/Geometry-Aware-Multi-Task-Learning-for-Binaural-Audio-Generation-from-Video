import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import os
import librosa
import random
from PIL import Image, ImageEnhance
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import torch.optim.lr_scheduler as lr_scheduler
import moviepy.editor as mpy
# from moviepy.video.VideoClip import ImageClip
# from moviepy.video.io.VideoFileClip import VideoFileClip
# from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
# from moviepy.audio.io.AudioFileClip import AudioFileClip
from PIL import Image
import tempfile
from scipy.io import wavfile
import ffmpeg
import subprocess
from pydub import AudioSegment
import tqdm