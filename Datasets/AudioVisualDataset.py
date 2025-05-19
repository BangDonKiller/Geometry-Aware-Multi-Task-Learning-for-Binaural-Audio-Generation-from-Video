import sys
import os
import random

random.seed(42)

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from imports import *
from params import *

def generate_spectrogram(audio):
    """
    將一維音訊訊號轉換為包含實部與虛部的頻譜圖（兩通道）

    - 參數:
        - audio (np.ndarray): 單聲道或多聲道的一維音訊訊號
    
    - 回傳:
        - np.ndarray: 形狀為 (2, 頻率bins, 時間frames) 的二通道頻譜圖，第一通道為實部、第二通道為虛部
    """
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel


def process_image(image, augment):
    """
    對輸入圖像進行裁剪與資料增強處理（亮度與色彩調整）

    - 參數:
        - image (PIL.Image): 輸入圖像
        - augment (bool): 是否啟用資料增強

    - 回傳:
        - PIL.Image: 處理後的圖像
    """
    image = image.resize((480, 240))
    w, h = image.size
    w_offset = w - 448
    h_offset = h - 224
    left = random.randrange(0, w_offset + 1)
    upper = random.randrange(0, h_offset + 1)
    image = image.crop((left, upper, left+448, upper+224))
    if augment:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random() * 0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random() * 0.6 + 0.7)
    return image


def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    """
    將音訊樣本正規化，使其均方根值 (RMS) 接近指定目標

    - 參數:
        - samples (np.ndarray): 音訊樣本
        - desired_rms (float): 期望的 RMS 值，預設為 0.1
        - eps (float): 為避免除以 0，加入的最小常數
    
    - 回傳:
        - np.ndarray: 正規化後的音訊樣本
    """
    rms = np.maximum(eps, np.sqrt(np.mean(samples ** 2)))
    samples = samples * (desired_rms / rms)
    return samples


class AudioVisualDataset(Dataset):
    def __init__(self, audio_dir, frame_dir, gpu_available, mode = 'train'):
        super(AudioVisualDataset, self).__init__()
        self.audio_dir = audio_dir
        self.frame_dir = frame_dir
        self.audio_length = audio_length           # the audio for each length is 0.63 sec
        self.audio_sampling_rate = audio_sampling_rate    # sampling rate for each audio
        self.enable_data_augmentation = True
        self.nThreads = 0
        self.audios = []
        self.device = device
        self.mode = mode

        for file_name in os.listdir(audio_dir):
            file_path = os.path.join(audio_dir, file_name)
            # Check if the path points to a file (as opposed to a directory)
            if os.path.isfile(file_path):
                self.audios.append(file_path)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.vision_output_transform = transforms.Compose([transforms.PILToTensor()])
        
        # 切分資料集為訓練、驗證和測試集
        total_samples = len([f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))])
        train_ratio = 0.7  # 70% for training
        val_ratio = 0.15  # 15% for validation
        test_ratio = 0.15  # 15% for testing
        
        train_samples = int(total_samples * train_ratio)
        val_samples = int(total_samples * val_ratio)
        test_samples = total_samples - train_samples - val_samples      
        
        self.train_indices = random.sample(range(total_samples), train_samples)
        self.val_indices = random.sample(list(set(range(total_samples)) - set(self.train_indices)), val_samples)
        self.test_indices = list(set(range(total_samples)) - set(self.train_indices) - set(self.val_indices))

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index):
        if self.mode =='train':
            if simbinaural:
                #load audio
                audio, audio_rate = librosa.load(self.audios[index], sr=self.audio_sampling_rate, mono=False)

                # randomly get a start time for the audio segment from the clip
                ch_segment = np.random.choice(int(audio.shape[1]/(audio_rate * 5))) # Choose a 5 second interval to work with
                audio_start_time = random.uniform(0, 4.3 - self.audio_length) + (ch_segment*5.0)
                audio_end_time = audio_start_time + self.audio_length
                audio_start = int(audio_start_time * self.audio_sampling_rate)
                audio_end = audio_start + int(self.audio_length * self.audio_sampling_rate)
                audio = audio[:, audio_start:audio_end]
                audio = normalize(audio)
                audio_channel1 = audio[0,:]
                audio_channel2 = audio[1,:]

                #get the frame dir path based on audio path
                path_parts = self.audios[index].strip().split('/')
                video_num = path_parts[-1][6:-4]
                
                    
                # 獲取與音訊片段最接近的影像幀檔案
                frame_index = int(
                    round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 5))  # 5 frames extracted per second
                
                # 讀取影像幀檔案，並做資料強化
                frame = process_image(Image.open(os.path.join(self.frame_dir, video_num, video_num + "_" + str(frame_index).zfill(4) + '.jpg')).convert('RGB'),
                                    self.enable_data_augmentation)
                # 將影像轉換為張量並進行正規化
                frame = self.vision_transform(frame)

                # 獲取與原始影像幀檔案相差1秒的影像幀檔案，做的資料處理與上面相同
                second_frame_index = int(round((random.uniform(0, 4.9) + (ch_segment*5.0)) * 5))
                while second_frame_index == frame_index:
                    second_frame_index = int(round((random.uniform(0, 4.9) + (ch_segment*5.0)) * 5))
                second_frame = process_image(Image.open(os.path.join(self.frame_dir, video_num, video_num + "_" + str(frame_index).zfill(4) + '.jpg')).convert('RGB'),
                                    self.enable_data_augmentation)
                second_frame = self.vision_transform(second_frame)
                
                # Load RIR
                binaural_rir_file = os.path.join(rir_path, video_num, f'{video_num}_{ch_segment}.wav')
                rir, sr = librosa.load(binaural_rir_file, sr=16000, mono=False)
                rir = np.pad(rir, ((0,0), (0, max(0, 3568 - rir.shape[1]))), 'constant', constant_values=0)
                spec1 = np.abs(librosa.stft(rir[0, :], n_fft=512, hop_length=16, win_length=64, center=True))[:224, :224]
                spec2 = np.abs(librosa.stft(rir[1, :], n_fft=512, hop_length=16, win_length=64, center=True))[:224, :224]
                spectro = torch.FloatTensor(np.stack((spec1, spec2)))
                
            else:
                # 載入音訊  
                audio, audio_rate = librosa.load(self.audios[index], sr=self.audio_sampling_rate, mono=False)
                # 在此音樂檔案中(10s)隨機選擇一個開始時間，此片段0.63秒
                audio_start_time = random.uniform(0, 9.9 - self.audio_length)
                audio_end_time = audio_start_time + self.audio_length
                audio_start = int(audio_start_time * self.audio_sampling_rate)
                audio_end = audio_start + int(self.audio_length * self.audio_sampling_rate)
                audio = audio[:, audio_start:audio_end]
                # 正規化音訊樣本
                audio = normalize(audio)
                # 區分左右聲道
                audio_channel1 = audio[0, :]
                audio_channel2 = audio[1, :]

                # 找到音訊片段的對應影像幀檔案路徑
                path_parts = self.audios[index].strip().split('/')                
                video_num = path_parts[-1][-10:-4]

                # 獲取與音訊片段最接近的影像幀檔案
                frame_index = int(
                    round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  # 10 frames extracted per second
                # 讀取影像幀檔案，並做資料強化
                frame = process_image(Image.open(os.path.join(self.frame_dir, video_num, str(frame_index) + '.jpg')).convert('RGB'),
                                    self.enable_data_augmentation)
                # 將影像轉換為張量並進行正規化
                frame = self.vision_transform(frame)

                # 獲取與原始影像幀檔案相差1秒的影像幀檔案，做的資料處理與上面相同
                # 這裡的 delta 是隨機生成的，範圍在 -1 到 1 之間
                delta = random.uniform(-1, 1)
                second_frame_index = int(np.round(frame_index + 10*delta)) 
                if second_frame_index <= 0:
                    second_frame_index = int(np.round(frame_index + 10*abs(delta)))
                second_frame = process_image(Image.open(os.path.join(self.frame_dir, video_num, str(frame_index) + '.jpg')).convert('RGB'),
                                    self.enable_data_augmentation)
                second_frame = self.vision_transform(second_frame)
                spectro = False
            
            # 將音訊片段轉換為頻譜圖，並將其轉換為浮點數張量
            # 這邊的處理是作為Ground Truth的資料
            audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
            audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 + audio_channel2))
            channel1_spec = torch.FloatTensor(generate_spectrogram(audio_channel1))
            channel2_spec = torch.FloatTensor(generate_spectrogram(audio_channel2))
            
            left_spec = torch.FloatTensor(generate_spectrogram(audio_channel1)[:, :256, :])
            right_spec = torch.FloatTensor(generate_spectrogram(audio_channel2)[:, :256, :])
            
            # 有50%機率左右聲道顛倒
            if np.random.random() < 0.5:
                coherence_spec = torch.cat((left_spec, right_spec), dim=0)
                label = torch.FloatTensor([0])
            else:
                coherence_spec = torch.cat((right_spec, left_spec), dim=0)
                label = torch.FloatTensor([1])
            
            return {'frame': frame, 'second_frame': second_frame, 'audio_diff_spec': audio_diff_spec, 'audio_mix_spec': audio_mix_spec, 'channel1_spec': channel1_spec , 'channel2_spec': channel2_spec, 'cl_spec': coherence_spec, 'label': label, 'rir_spec': spectro}

        elif self.mode =='val':
            # val做的事情與train一樣
            if simbinaural:
                #load audio
                audio, audio_rate = librosa.load(self.audios[index], sr=self.audio_sampling_rate, mono=False)

                # randomly get a start time for the audio segment from the clip
                ch_segment = np.random.choice(int(audio.shape[1]/(audio_rate * 5))) # Choose a 5 second interval to work with
                audio_start_time = random.uniform(0, 4.3 - self.audio_length) + (ch_segment*5.0)
                audio_end_time = audio_start_time + self.audio_length
                audio_start = int(audio_start_time * self.audio_sampling_rate)
                audio_end = audio_start + int(self.audio_length * self.audio_sampling_rate)
                audio = audio[:, audio_start:audio_end]
                audio = normalize(audio)
                audio_channel1 = audio[0,:]
                audio_channel2 = audio[1,:]

                #get the frame dir path based on audio path
                path_parts = self.audios[index].strip().split('/')
                video_num = path_parts[-1][6:-4]
                
                # 獲取與音訊片段最接近的影像幀檔案
                    
                frame_index = int(
                    round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 5))  # 5 frames extracted per second
                # 讀取影像幀檔案，並做資料強化
                
                frame = process_image(Image.open(os.path.join(self.frame_dir, video_num, video_num + "_" + str(frame_index).zfill(4) + '.jpg')).convert('RGB'),
                                    self.enable_data_augmentation)
                # 將影像轉換為張量並進行正規化
                frame = self.vision_transform(frame)

                # 獲取與原始影像幀檔案相差1秒的影像幀檔案，做的資料處理與上面相同
                second_frame_index = int(round((random.uniform(0, 4.9) + (ch_segment*5.0)) * 5))
                while second_frame_index == frame_index:
                    second_frame_index = int(round((random.uniform(0, 4.9) + (ch_segment*5.0)) * 5))
                second_frame = process_image(Image.open(os.path.join(self.frame_dir, video_num, video_num + "_" + str(frame_index).zfill(4) + '.jpg')).convert('RGB'),
                                    self.enable_data_augmentation)
                second_frame = self.vision_transform(second_frame)
                
                # Load RIR
                binaural_rir_file = os.path.join(rir_path, video_num, f'{video_num}_{ch_segment}.wav')
                rir, sr = librosa.load(binaural_rir_file, sr=16000, mono=False)
                rir = np.pad(rir, ((0,0), (0, max(0, 3568 - rir.shape[1]))), 'constant', constant_values=0)
                spec1 = np.abs(librosa.stft(rir[0, :], n_fft=512, hop_length=16, win_length=64, center=True))[:224, :224]
                spec2 = np.abs(librosa.stft(rir[1, :], n_fft=512, hop_length=16, win_length=64, center=True))[:224, :224]
                spectro = torch.FloatTensor(np.stack((spec1, spec2)))
                
            else:
                # 載入音訊  
                audio, audio_rate = librosa.load(self.audios[index], sr=self.audio_sampling_rate, mono=False)
                # 在此音樂檔案中(10s)隨機選擇一個開始時間，此片段0.63秒
                audio_start_time = random.uniform(0, 9.9 - self.audio_length)
                audio_end_time = audio_start_time + self.audio_length
                audio_start = int(audio_start_time * self.audio_sampling_rate)
                audio_end = audio_start + int(self.audio_length * self.audio_sampling_rate)
                audio = audio[:, audio_start:audio_end]
                # 正規化音訊樣本
                audio = normalize(audio)
                # 區分左右聲道
                audio_channel1 = audio[0, :]
                audio_channel2 = audio[1, :]

                # 找到音訊片段的對應影像幀檔案路徑
                path_parts = self.audios[index].strip().split('/')                
                video_num = path_parts[-1][-10:-4]
                
                # 獲取與音訊片段最接近的影像幀檔案
                frame_index = int(
                    round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  # 10 frames extracted per second
                # 讀取影像幀檔案，並做資料強化
                frame = process_image(Image.open(os.path.join(self.frame_dir, video_num, str(frame_index) + '.jpg')).convert('RGB'),
                                    self.enable_data_augmentation)
                # 將影像轉換為張量並進行正規化
                frame = self.vision_transform(frame)

                # 獲取與原始影像幀檔案相差1秒的影像幀檔案，做的資料處理與上面相同
                # 這裡的 delta 是隨機生成的，範圍在 -1 到 1 之間
                delta = random.uniform(-1, 1)
                second_frame_index = int(np.round(frame_index + 10*delta)) 
                if second_frame_index <= 0:
                    second_frame_index = int(np.round(frame_index + 10*abs(delta)))
                second_frame = process_image(Image.open(os.path.join(self.frame_dir, video_num, str(frame_index) + '.jpg')).convert('RGB'),
                                    self.enable_data_augmentation)
                second_frame = self.vision_transform(second_frame)
                spectro = False
            
            # 將音訊片段轉換為頻譜圖，並將其轉換為浮點數張量
            # 這邊的處理是作為Ground Truth的資料
            audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
            audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 + audio_channel2))
            channel1_spec = torch.FloatTensor(generate_spectrogram(audio_channel1))
            channel2_spec = torch.FloatTensor(generate_spectrogram(audio_channel2))
            
            left_spec = torch.FloatTensor(generate_spectrogram(audio_channel1)[:, :256, :])
            right_spec = torch.FloatTensor(generate_spectrogram(audio_channel2)[:, :256, :])
            
            # 有50%機率左右聲道顛倒
            if np.random.random() < 0.5:
                coherence_spec = torch.cat((left_spec, right_spec), dim=0)
                label = torch.FloatTensor([0])
            else:
                coherence_spec = torch.cat((right_spec, left_spec), dim=0)
                label = torch.FloatTensor([1])
            
            return {'frame': frame, 'second_frame': second_frame, 'audio_diff_spec': audio_diff_spec, 'audio_mix_spec': audio_mix_spec, 'channel1_spec': channel1_spec , 'channel2_spec': channel2_spec, 'cl_spec': coherence_spec, 'label': label, 'rir_spec': spectro}
        else:
            # 載入音訊
            audio, audio_rate = librosa.load(self.audios[index], sr=self.audio_sampling_rate, mono=False)
            # 將音訊正規化後合併為單聲道
            audio = normalize(audio)
            audio_channel1 = audio[0, :]
            audio_channel2 = audio[1, :]
            audio_mix = audio_channel1 + audio_channel2
            
            # 找到音訊片段的對應影像幀檔案路徑
            path_parts = self.audios[index].strip().split('/')
            if simbinaural:
                ch_segment = np.random.choice(int(audio.shape[1]/(audio_rate * 5))) # Choose a 5 second interval to work with
                video_num = path_parts[-1][6:-4]
                
                # Load RIR
                binaural_rir_file = os.path.join(rir_path, video_num, f'{video_num}_{ch_segment}.wav')
                rir, sr = librosa.load(binaural_rir_file, sr=16000, mono=False)
                rir = np.pad(rir, ((0,0), (0, max(0, 3568 - rir.shape[1]))), 'constant', constant_values=0)
                spec1 = np.abs(librosa.stft(rir[0, :], n_fft=512, hop_length=16, win_length=64, center=True))[:224, :224]
                spec2 = np.abs(librosa.stft(rir[1, :], n_fft=512, hop_length=16, win_length=64, center=True))[:224, :224]
                spectro = torch.FloatTensor(np.stack((spec1, spec2)))
            else:
                video_num = path_parts[-1][-10:-4]
                spectro = False
            
            frames_dir = os.path.join(self.frame_dir, video_num)
            frame_files = sorted(os.listdir(frames_dir))
            
            print("Frame files: ", frames_dir)

            # 遍歷該音訊所有的影像幀檔案，並將其轉換為張量
            frames = []
            frames_to_video = []
            for frame_file in sorted(frame_files, key=lambda x: int(x.split('.')[0])):
                frame_path = os.path.join(frames_dir, frame_file)
                frame = process_image(Image.open(frame_path).convert('RGB'), augment=False).convert('RGB')
                
                frame_to_video = Image.open(frame_path)
                frame_to_video = frame_to_video.convert('RGB')
                frame_to_video = frame_to_video.resize((480, 240))
                frame_to_video = frame_to_video.convert('RGB')
                frame_to_video = self.vision_output_transform(frame_to_video)
                frames_to_video.append(frame_to_video)
                
                frame = self.vision_transform(frame)
                frames.append(frame)
                
            return {'frames': frames, 'audio_mix': audio_mix, 'audio_channel1': audio_channel1 , 'audio_channel2': audio_channel2, "frames_to_video":frames_to_video, 'rir_spec': spectro}
        
if __name__ == "__main__":
    fake_audio_size = int(audio_length * audio_sampling_rate)
    fake_audio = np.random.rand(fake_audio_size)
    print(type(fake_audio))
    fake_audio_spec = generate_spectrogram(fake_audio)
    print(fake_audio.shape)
    print(fake_audio_spec.shape)
    