from Models.backbone_model import *
from Models.geometry_model import *
from Models.spatial_model import *
from Models.rir_model import *
from Models.model import *
from imports import * 
from Datasets.AudioVisualDataset import *
from networks.Networks import *
from params import *


def show_spec(spec_left_true,spec_right_true, spec_left_pred,spec_right_pred, idx):
    """
    顯示並儲存真實與預測的左右聲道頻譜圖。

    參數:
    - spec_left_true (Tensor): 真實左聲道頻譜 (複數形式，實部與虛部為不同通道)
    - spec_right_true (Tensor): 真實右聲道頻譜 (複數形式)
    - spec_left_pred (Tensor): 預測左聲道頻譜 (複數形式)
    - spec_right_pred (Tensor): 預測右聲道頻譜 (複數形式)
    - idx (int): 此組資料的索引，用於儲存圖檔命名
    """
    spec_left_pred = torch.clone(spec_left_pred).to('cpu').detach().numpy()
    spec_right_pred = torch.clone(spec_right_pred).to('cpu').detach().numpy()
    spec_left_true = torch.sqrt(spec_left_true[0, :, :]**2 + spec_left_true[1, :, :]**2).cpu().detach().numpy()
    spec_right_true = torch.sqrt(spec_right_true[0, :, :]**2 + spec_right_true[1, :, :]**2).cpu().detach().numpy()
    spec_left_pred = np.sqrt(spec_left_pred[0, :, :]**2 + spec_left_pred[1, :, :]**2)
    spec_right_pred = np.sqrt(spec_right_pred[0, :, :]**2 + spec_right_pred[1, :, :]**2)

    
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(spec_left_true), ref=np.max), hop_length=160,
                             y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Left True Spectrogram (dB)')
    
    plt.subplot(2, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(spec_right_true), ref=np.max), hop_length=160,
                             y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Right True Spectrogram (dB)')
    
    plt.subplot(2, 2, 3)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(spec_left_pred), ref=np.max), hop_length=160,
                             y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Left Predicted Spectrogram (dB)')
    
    plt.subplot(2, 2, 4)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(spec_right_pred), ref=np.max), hop_length=160,
                             y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Right Predicted Spectrogram (dB)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(test_spec_path,'spectrogram_subplot_' + str(idx) + '.jpg'), format='jpg')


def save_audio(audio_data, path):
    """
    儲存單聲道或雙聲道音訊至 WAV 檔案。

    參數:
    - audio_data (Tensor 或 np.ndarray): 單聲道或雙聲道音訊資料 (格式為 [2, N] 或 [N])
    - path (str): 儲存的目標路徑
    """
    if len(audio_data) == 1:
        audio_data = audio_data/1.1/torch.max(torch.abs(audio_data))
        wavfile.write(path, audio_sampling_rate , audio_data.numpy().astype(np.float32))
    elif len(audio_data) == 2:
        audio_data[0] = audio_data[0]/1.1/torch.max(torch.abs(audio_data[0]))
        audio_data[1] = audio_data[1]/1.1/torch.max(torch.abs(audio_data[1]))
        if type(audio_data[0]).__module__ == np.__name__ and type(audio_data[1]).__module__ == np.__name__:
            audio_channel_left = audio_data[0]
            audio_channel_right = audio_data[1]
        if type(audio_data[0]).__module__ != np.__name__:
            audio_channel_left = audio_data[0].numpy().astype(np.float32)
        if type(audio_data[1]).__module__ != np.__name__:
            audio_channel_right = audio_data[1].numpy().astype(np.float32)
        
        audio =  np.hstack((audio_channel_left.reshape(-1, 1), audio_channel_right.reshape(-1, 1)))
        wavfile.write(path,audio_sampling_rate, audio)


def video_from_frames(frames, path):
    """
    將連續影格合成為一部影片，設為 30fps、長度 10 秒。

    參數:
    - frames (list of Tensor): 每個 frame 為 4D tensor，形狀為 [1, C, H, W]
    - path (str): 合成影片的儲存路徑
    """
    frame_duration = 1 / 10  # 每個畫面的持續時間
    desired_duration = 10  # 目標視頻持續時間


    # 計算達成目標持續時間所需的影格數量
    num_frames = int(desired_duration / frame_duration)

    # 轉置 frames 並將它們轉換為 NumPy 陣列
    transposed_frames = [np.transpose(frame[0, :, :, :], (1, 2, 0)) for frame in frames]

    # 創建影像剪輯列表，這裡需要將 Tensor 轉換為 NumPy 陣列
    frame_clips = [mpy.ImageClip(frame_file, duration=frame_duration) for frame_file in transposed_frames[:num_frames]]

    # 合併視頻剪輯為最終的視頻
    video_clip = mpy.concatenate_videoclips(frame_clips, method="compose")

    # 計算視頻的總時長
    video_duration = len(frame_clips) * frame_duration
    video_clip = video_clip.set_duration(video_duration)

    # 設置視頻的幀率（fps）
    # frame_rate = 30
    frame_rate = 10
    video_clip = video_clip.set_fps(frame_rate)

    # 輸出視頻文件到指定路徑
    video_clip.write_videofile(path, codec='libx264')

    
def video_from_files(idx, paths):
    """
    將視訊與音訊合併為一個影片檔案。

    參數:
    - idx (int): 該影片的索引編號，用於命名
    - paths (dict): 包含 "frames_video"（視訊路徑）與 "combine_audio"（音訊路徑）之字典
    """
    output_video = 'video_' + str(idx) + '.mp4'
    output_path = os.path.join(video_path, output_video)
    
    if os.path.exists(output_path):
            os.remove(output_path)
            
    combine_audio_path = paths["combine_audio"]

    # loading video gfg
    video_clip = mpy.VideoFileClip(paths["frames_video"]) 
    
    # create stereo audio
    audio_clip = mpy.AudioFileClip(combine_audio_path)

    video_clip = video_clip.set_audio(audio_clip)
    
    video_clip.write_videofile(output_path, codec='libx264')

    
    
def remove_temps_media(idx, paths):
    """
    刪除指定的暫存媒體檔案（影片、音訊）。

    參數:
    - idx (int): 索引（僅供語意參考）
    - paths (dict): 暫存檔案路徑字典
    """
    paths = list(paths.values())
    
    for file_path in paths:
            # Check if the file exists before removing it
        if os.path.exists(file_path):
            os.remove(file_path)


def create_video(frames, audio_left, audio_right, idx):
    """
    從影格與左右聲道音訊生成一個帶聲音的影片並儲存。

    參數:
    - frames (list of Tensor): 影格序列，每個為形狀 [1, C, H, W] 的 tensor
    - audio_left (Tensor): 左聲道音訊資料
    - audio_right (Tensor): 右聲道音訊資料
    - idx (int): 索引，用於命名與儲存
    """
    frames_video_file = 'frame_video_' + str(idx) + '.mp4'
    combine_audio_file = 'combine_audio' + str(idx) + '.wav'
    
    frames_video_path = os.path.join(video_path, frames_video_file)
    combine_audio = os.path.join(video_path, combine_audio_file)
    
    paths = {"frames_video":frames_video_path, "combine_audio":combine_audio}
    
    # Create temp audio files and video separately
    save_audio([audio_left, audio_right], paths["combine_audio"])
    video_from_frames(frames, paths["frames_video"])
    
    # Combine audio files and video
    video_from_files(idx, paths)
    
    # Remove temp files
    remove_temps_media(idx, paths)
    


def inverse_spectrogram(audio_spec, length,  type='tensor'):
    """
    將頻譜轉換回時域音訊（Inverse STFT）。

    參數:
    - audio_spec (Tensor or np.ndarray): 頻譜資料，形狀為 [2, F, T]，表示實部與虛部
    - length (int): 要回復的音訊長度（樣本數）
    - type (str): 指定輸入資料型別為 'tensor' 或 'numpy'

    回傳:
    audio_reconstructed (np.ndarray): 還原後的音訊訊號
    """
    if type == 'tensor':
        audio_spectogram = audio_spec[0,0,:,:] + 1j * audio_spec[0,1,:,:]
    else:
        audio_spectogram = audio_spec[0] + 1j * audio_spec[1]
        
    audio_spectogram = audio_spectogram.detach().cpu().numpy()
    
    # Compute the inverse STFT to get the audio signal
    audio_reconstructed = librosa.core.istft(audio_spectogram, hop_length=160, win_length=400, length=length, center=True)
    
    return audio_reconstructed


def data_test_handle(data, idx, num_loops):
    """
    擷取測試資料中對應的影格與音訊段，並產生其頻譜資訊。

    參數:
    - data (dict): 包含 'frames', 'audio_mix', 'audio_channel1', 'audio_channel2' 的測試資料
    - idx (int): 當前測試資料索引
    - num_loops (int): 總測試循環次數（用於判斷是否為最後一段）

    回傳:
    dict: 包含 frame, second_frame, 以及四種頻譜資訊（diff, mix, ch1, ch2）
    """
    frames = data['frames']
    audio_mix = data['audio_mix']
    audio_channel1 = data['audio_channel1']
    audio_channel2 = data['audio_channel2']
    audio_full_time = np.floor(len(audio_channel1[0,:]) / audio_sampling_rate)
    
    if idx < num_loops - 1:
        audio_start_time = idx * test_overlap * audio_length
        audio_end_time = idx * test_overlap * audio_length + audio_length
        audio_start = int(round(audio_start_time * audio_sampling_rate))
        audio_end = int(round(audio_end_time * audio_sampling_rate))
    else:
        audio_start_time = audio_full_time - audio_length
        audio_end_time = audio_full_time
        audio_start = int(round(audio_start_time * audio_sampling_rate))
        audio_end = int(round(audio_end_time * audio_sampling_rate))
        
    
    # get the closest frame to the audio segment
    frame_index = int(round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  # 10 frames extracted per second
    frame = frames[frame_index]


    # get a frame 1 secend befor/after the original frame
    delta = random.uniform(-1, 1)
    second_frame_index = int(np.round(frame_index + 10*delta)) 
    if second_frame_index <= 0:
        second_frame_index = int(np.round(frame_index + 10*abs(delta)))
    second_frame = frames[second_frame_index]
    
    
    audio_mix = audio_mix[:, audio_start : audio_end]
    audio_channel1 = audio_channel1[:, audio_start : audio_end]
    audio_channel2 = audio_channel2[:, audio_start : audio_end]
    
    # passing the spectrogram of the difference
    audio_diff_spec = batch_spec(audio_channel1 - audio_channel2)
    audio_mix_spec = batch_spec(audio_mix)
    channel1_spec = batch_spec(audio_channel1)
    channel2_spec = batch_spec(audio_channel2)
    
    return {'frame': frame, 'second_frame': second_frame, 'audio_diff_spec': audio_diff_spec, 'audio_mix_spec': audio_mix_spec, 'channel1_spec': channel1_spec , 'channel2_spec': channel2_spec}


def handle_output(data, outputs, idx, num_loops):
    """
    將每一段產出的頻譜進行疊加與平均處理，形成完整音訊頻譜。

    參數:
    - data (dict): 包含 ground truth 的 slide 覆蓋區段頻譜與音訊資料
    - outputs (dict): 模型產出結果，包含左右聲道與雙聲道頻譜，以及重建音訊
    - idx (int): 當前段落索引
    - num_loops (int): 全部段落數量，用於判定是否為最後一段
    """
    left_spectrogram_slide = data['left_spectrogram']
    right_spectrogram_slide = data['right_spectrogram']
    binaural_spectrogram_slide = data['binaural_spectrogram']
    audio_gt_slide = data['audio_gt']
    time_frame = audio_gt_slide.shape[3]
    
    
    left_spectrogram = outputs["left_spectrogram"]
    right_spectrogram = outputs["right_spectrogram"]
    binaural_spectrogram = outputs["binaural_spectrogram"]
    audio_gt = outputs["audio_gt"]
    
    if idx == num_loops - 1:
        left_spectrogram[:,:,:,int(left_spectrogram.shape[3] - time_frame) : ] += left_spectrogram_slide
        left_spectrogram[:,:,:, int(time_frame * test_overlap) :int((idx - 1) * time_frame * test_overlap + time_frame) ] *= 0.5
        
        right_spectrogram[:,:,:,int(right_spectrogram.shape[3] - time_frame) : ] += right_spectrogram_slide
        right_spectrogram[:,:,:, int(time_frame * test_overlap) :int((idx - 1) * time_frame * test_overlap + time_frame) ] *= 0.5
        
        binaural_spectrogram[:,:,:,int(binaural_spectrogram.shape[3] - time_frame) : ] += binaural_spectrogram_slide
        binaural_spectrogram[:,:,:, int(time_frame * test_overlap) :int((idx - 1) * time_frame * test_overlap + time_frame) ] *= 0.5
        
        audio_gt[:,:,:,int(audio_gt.shape[3] - time_frame) : ] += audio_gt_slide
        audio_gt[:,:,:, int(time_frame * test_overlap) :int((idx - 1) * time_frame * test_overlap + time_frame) ] *= 0.5
        
        if left_spectrogram.shape[3] - time_frame < (idx - 2) * time_frame * test_overlap + time_frame:
            left_spectrogram[:,:,:, int(left_spectrogram.shape[3] - time_frame) :int((idx - 2) * time_frame * test_overlap + time_frame) ] *= 2/3
            right_spectrogram[:,:,:, int(right_spectrogram.shape[3] - time_frame) :int((idx - 2) * time_frame * test_overlap + time_frame) ] *= 2/3
            binaural_spectrogram[:,:,:, int(binaural_spectrogram.shape[3] - time_frame) :int((idx - 2) * time_frame * test_overlap + time_frame) ] *= 2/3
            audio_gt[:,:,:, int(audio_gt.shape[3] - time_frame) :int((idx - 2) * time_frame * test_overlap + time_frame) ] *= 2/3

        
    else:
        left_spectrogram[:,:,:,int(idx * time_frame * test_overlap) :int(idx * time_frame * test_overlap + time_frame) ] += left_spectrogram_slide
        right_spectrogram[:,:,:,int(idx * time_frame * test_overlap) :int(idx * time_frame * test_overlap + time_frame) ] += right_spectrogram_slide
        binaural_spectrogram[:,:,:,int(idx * time_frame * test_overlap) :int(idx * time_frame * test_overlap + time_frame) ] += binaural_spectrogram_slide
        audio_gt[:,:,:,int(idx * time_frame * test_overlap) :int(idx * time_frame * test_overlap + time_frame) ] += audio_gt_slide
    
    return (left_spectrogram, right_spectrogram, binaural_spectrogram, audio_gt)


def build_test_model():
    """
    載入個別模型權重且組建整體模型(test_model)
    """
    resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    visual_net = VisualNet(resnet18).to(device)
    audio_net = AudioNet().to(device)
    fusion_net = APNet().to(device)
    spatial_net = AudioNet(input_nc=4).to(device)
    generator = Generator().to(device)

    # Load the saved model parameters
    visual_net.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'visual_best.pth')))
    audio_net.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'audio_best.pth')))
    fusion_net.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'fusion_best.pth')))
    spatial_net.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'classifier_best.pth')))
    generator.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'generator_best.pth')))

    nets = (visual_net, spatial_net, audio_net, fusion_net, generator)

    # construct our models
    test_model = model(nets)
    
    return test_model


def batch_spec(batch_audio):
    """
    將輸入的批次音訊中的第一筆資料轉換為對應的頻譜張量。

    此函式用於從 batch_audio 中提取第一筆音訊資料，
    並透過 generate_spectrogram 函式產生其頻譜，再轉為 PyTorch tensor，
    並增加 batch 維度以符合模型輸入需求。

    參數:
    - batch_audio (Tensor): 一個形狀為 [batch_size, samples] 的 PyTorch tensor，包含多筆音訊資料。

    回傳:
    - Tensor: 一個形狀為 [1, 頻率bins, 時間frames] 的 PyTorch tensor，為該音訊對應的頻譜。
    """
    audio = batch_audio[0,:].numpy()
    spec = torch.FloatTensor(generate_spectrogram(audio))
    batch_spec = torch.unsqueeze(spec, dim=0)
    return batch_spec


if __name__=='__main__':
    
    # 載入資料集並切割測試資料集
    test_dataset = AudioVisualDataset(audios_dir, frames_dir, gpu_available, 'test')
    subset_test_dataset = Subset(test_dataset, test_dataset.test_indices)
    data_loader_test = DataLoader(
                subset_test_dataset,
                batch_size=batch_size_test,
                shuffle=True,
                num_workers=int(test_dataset.nThreads))
    
    
    # run the test
    # 載入權重
    test_model = build_test_model()
    test_model.eval()
    loss_criterion = torch.nn.MSELoss()
    if(len(gpu_ids) > 0 and gpu_available):
        loss_criterion.cuda(gpu_ids[0])
    losses_stft, losses_env = [],[]
    batch_idx = 0
    
    for idx, data in enumerate(data_loader_test):
        sliding_window_size = audio_length * audio_sampling_rate  # 計算每個滑動視窗的音訊樣本數
        audio_length_test = data['audio_mix'].shape[1]  # 取得混合音訊的總長度（以樣本數表示）

        num_loops = int((np.floor(audio_length_test / sliding_window_size)) / test_overlap 
                        - 1 + np.ceil(audio_length_test / sliding_window_size 
                        - audio_length_test // sliding_window_size))  # 計算滑動視窗要跑幾次以涵蓋整段音訊

        loss_geometry = []  # 用於儲存每一段的 geometric consistency loss

        channel1_spec = batch_spec(data['audio_channel1']).to(device)  # 將 channel1 的音訊轉為頻譜並搬移到運算裝置
        channel2_spec = batch_spec(data['audio_channel2']).to(device)  # 將 channel2 的音訊轉為頻譜並搬移到運算裝置

        # 預先建立空的張量來儲存模型預測的各種頻譜
        left_spectrogram = torch.zeros_like(channel1_spec[:, :, :-1, :]).to(device)
        right_spectrogram = torch.zeros_like(channel1_spec[:, :, :-1, :]).to(device)
        binaural_spectrogram = torch.zeros_like(channel1_spec[:, :, :-1, :]).to(device)
        audio_gt = torch.zeros_like(channel1_spec[:, :, :-1, :]).to(device)

        # 將所有預測結果組合成一個 dict
        outputs = {
            "left_spectrogram": left_spectrogram,
            "right_spectrogram": right_spectrogram,
            "binaural_spectrogram": binaural_spectrogram,
            "audio_gt": audio_gt
        }

        frames = []  # 儲存 frame 用於產生影片

        for j in range(num_loops):  # 對每個滑動視窗段進行推論
            test_data = data_test_handle(data, j, num_loops)  # 擷取當前段的 frame 與頻譜資料等

            output = test_model(test_data, mode='test')  # 呼叫模型的 forward pass，取得預測結果

            # 處理模型輸出，整合目前段的頻譜到 outputs 中
            (left_spectrogram, right_spectrogram, binaural_spectrogram, audio_gt) = handle_output(
                output, outputs, j, num_loops
            )

            if j == debug_test_idx and idx < 10:  # 如果目前段為 debug 指定段，顯示頻譜圖以方便除錯
                show_spec(
                    test_data["channel1_spec"][0, :, :, :],
                    test_data["channel2_spec"][0, :, :, :],
                    output["left_spectrogram"][0, :, :, :],
                    output["right_spectrogram"][0, :, :, :],
                    idx
                )

            # 計算 geometric consistency loss，強化視覺特徵一致性
            mse_geometry = loss_criterion(output['visual_feature'], output['second_visual_feature'])
            loss_geometry_slide = torch.maximum(mse_geometry - alpha, torch.tensor(0))  # 使用 margin 的方式壓制過小誤差

            loss_geometry.append(loss_geometry_slide.item())  # 將目前段的 geometric loss 存入列表

        # 預測的頻譜轉回時域音訊（左聲道）
        re_left_spectrogram = inverse_spectrogram(left_spectrogram, audio_length_test)
        re_left_spectrogram = torch.FloatTensor(re_left_spectrogram)

        # 預測的頻譜轉回時域音訊（右聲道，疑似用 left_spectrogram 錯植）
        re_right_spectrogram = inverse_spectrogram(left_spectrogram, audio_length_test)
        re_right_spectrogram = torch.FloatTensor(re_right_spectrogram)

        # 計算 STFT 領域的 loss（含 binaural 差異、兩聲道一致性等）
        loss_geometry = sum(loss_geometry) / len(loss_geometry)  # 計算平均 geometric loss
        difference_loss = loss_criterion(binaural_spectrogram, audio_gt)  # 預測雙耳與真實雙耳頻譜的差異
        channel1_loss = loss_criterion(left_spectrogram, channel1_spec[:, :, :-1, :])  # 預測左聲道與真實左聲道頻譜差異
        channel2_loss = loss_criterion(right_spectrogram, channel2_spec[:, :, :-1, :])  # 預測右聲道與真實右聲道頻譜差異
        fusion_loss = 0.5 * (channel1_loss + channel2_loss)  # 左右聲道損失的平均
        loss_backbone = lambda_binarual * difference_loss + lambda_f * fusion_loss  # 結合 binaural 差異與聲道融合損失

        # 將 STFT loss 與幾何一致性 loss 結合為最終 loss
        loss = lambda_b * loss_backbone + lambda_g * loss_geometry
        losses_stft.append(loss_backbone.item())  # 儲存 STFT 領域的 loss

        # 計算 ENV（音訊波形）領域的 loss（重建音訊與真實音訊差異）
        loss_left = loss_criterion(re_left_spectrogram, data['audio_channel1'][0, :])  # 左聲道重建誤差
        loss_right = loss_criterion(re_right_spectrogram, data['audio_channel2'][0, :])  # 右聲道重建誤差
        env_loss = 0.5 * (loss_left + loss_right)  # 左右平均重建誤差
        losses_env.append(env_loss.item())  # 儲存 ENV loss

        # 若是前幾筆測試資料，則輸出合成影片
        frames = data["frames_to_video"]
        if idx < 10:
            create_video(frames, re_left_spectrogram, re_right_spectrogram, batch_idx)

        batch_idx += 1  # 計數累加

    # 計算所有測試資料的平均 STFT loss
    loss_avg_stft = (sum(losses_stft) / len(losses_stft))
    print("test average loss (stft) is:", loss_avg_stft)

    # 計算所有測試資料的平均 ENV loss
    loss_avg_env = (sum(losses_env) / len(losses_env))
    print("test average loss(env) is:", loss_avg_env)

    