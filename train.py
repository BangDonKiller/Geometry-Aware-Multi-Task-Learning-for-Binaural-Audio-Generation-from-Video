# this code handles the train
from Models.backbone_model import *
from Models.geometry_model import *
from Models.spatial_model import *
from Models.rir_model import *
from Models.model import *
from imports import * 
from Datasets.AudioVisualDataset import *
from networks.Networks import *
from params import *


def lr_decrease(optimizer, decay_param=0.9):
    """
    將優化器 (optimizer) 中的學習率 (learning rate) 按照 decay_param 進行衰減。

    Args:
        optimizer (torch.optim.Optimizer): 優化器物件，例如 Adam、SGD 等。
        decay_param (float, optional): 學習率衰減係數，預設為 0.9，代表每次調整後學習率變為原本的 90%。
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_param

    

def debug_dataset(dataset, epoch, idx=15, flag='input'):
    """
    針對資料集 (dataset) 中指定索引 (idx) 的資料進行除錯，將頻譜圖與影像存到本地資料夾。

    Args:
        dataset (dict): 包含聲音頻譜與影像資料的字典。
        epoch (int): 當前訓練的 epoch 數，用來標記存檔名稱。
        idx (int, optional): 欲查看的資料索引，預設為 15。
        flag (str, optional): 指定是 input (輸入資料) 還是 output (模型輸出結果)，預設為 'input'。
    """
    if flag == 'input':
        frame = dataset['frame']
        frame_idx = frame[idx]
        
        data_idx = dataset['channel1_spec'][:,:,:-1,:]
        audio_spec_idx = data_idx[idx]
    elif flag == 'output':
        data_idx = dataset['left_spectrogram']
        audio_spec_idx = data_idx[idx]
        
        cpu_tensor = audio_spec_idx.clone().cpu()
        audio_spec_idx = cpu_tensor.detach()

    audio_spec_idx = torch.sqrt(audio_spec_idx[0,:,:]**2 + audio_spec_idx[1,:,:]**2) 
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audio_spec_idx), ref=np.max), hop_length=160,
                            y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')
    if flag == 'input':
        plt.savefig('pic_for_debug/audio_spec_input_' + str(epoch) + '.jpg', format='jpg')
        
        plt.imshow(frame_idx.permute(1,2,0).numpy())
        plt.savefig('pic_for_debug/frame.jpg', format='jpg')
    elif flag == 'output':
        plt.savefig('pic_for_debug/audio_spec_output_' + str(epoch) + '.jpg', format='jpg')



    
def display_val(model, loss_criterion, writer, index, dataset_val):
    """
    在驗證資料集 (validation dataset) 上評估模型表現，計算平均損失 (loss)，並將結果寫入 TensorBoard。

    Args:
        model (torch.nn.Module): 要驗證的模型。
        loss_criterion (torch.nn.Module): 損失函數，例如 MSELoss。
        writer (SummaryWriter): TensorBoardX 的寫入器，用來記錄 loss。
        index (int): 當前的迭代次數 (通常是 epoch 數或 step 數)。
        dataset_val (DataLoader): 驗證資料集的 DataLoader。
    
    Returns:
        avg_loss (float): 本次驗證的平均損失值。
    """
    losses = []
    with torch.no_grad():
        for i, val_data in enumerate(tqdm.tqdm(dataset_val, desc="Validation", leave=False)):
            output = model(val_data, mode='val')
            channel1_spec = val_data['channel1_spec'].to(device)
            channel2_spec = val_data['channel2_spec'].to(device)
            
            channel1_loss = loss_criterion(output['left_spectrogram'], val_data["channel1_spec"][:,:,:-1,:].to(device))
            channel2_loss = loss_criterion(output['right_spectrogram'], val_data["channel2_spec"][:,:,:-1,:].to(device))
            fus_loss = (channel1_loss / 2 + channel2_loss / 2)
            loss = loss_criterion(output['binaural_spectrogram'], Variable(output['audio_gt'])) + fus_loss

            losses.append(loss.item()) 
    avg_loss = sum(losses) / len(losses)
    writer.add_scalar('data/val_loss', avg_loss, index)
    print('val loss: %.3f' % avg_loss)
    return avg_loss



def clear_folder(folder_path):
    """
    清除指定資料夾 (folder_path) 中的所有檔案與子資料夾。

    Args:
        folder_path (str): 要清除的資料夾路徑。
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        elif os.path.isdir(file_path):
            clear_folder(file_path)
            os.rmdir(file_path)
            print(f"Removed directory: {file_path}")

    

if __name__ == '__main__':
    
    clear_folder(debug_dir)
    
    dataset = AudioVisualDataset(audios_dir, frames_dir, gpu_available)
    subset_dataset = Subset(dataset, dataset.train_indices)
    data_loader = DataLoader(
                subset_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=int(dataset.nThreads))


        # validation dataset
    dataset.mode = 'val'
    val_dataset = AudioVisualDataset(audios_dir, frames_dir, gpu_available,  'val')
    subset_val_dataset = Subset(val_dataset, val_dataset.val_indices)
    data_loader_val = DataLoader(
                subset_val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=int(val_dataset.nThreads))
    dataset.mode = 'train'
    
    # test dataset
    test_dataset = AudioVisualDataset(audios_dir, frames_dir, gpu_available, 'test')
    subset_test_dataset = Subset(test_dataset, test_dataset.test_indices)
    data_loader_test = DataLoader(
                subset_test_dataset,
                batch_size=batch_size_test,
                shuffle=True,
                num_workers=int(test_dataset.nThreads))
    

    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

    ## build nets
    # resnet18 main net in our code
    resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    visual_net = VisualNet(resnet18)

    # spatial coherence net 音訊翻轉模型
    spatial_net = AudioNet(input_nc=4)
    spatial_net.apply(weights_init)

    # audio network for backbone
    audio_net = AudioNet()
    audio_net.apply(weights_init)

    # fusion network for backbone
    fusion_net = APNet()
    fusion_net.apply(weights_init)

    # generator net for rir (Not used for FairPlay dataset)
    generator = Generator()
    generator.apply(weights_init)

    nets = (visual_net, spatial_net, audio_net, fusion_net, generator)

    # construct our models
    model = model(nets)

    # use models with gpu
    if gpu_available:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        model.to(dataset.device)
    else:
        model.to('cpu')
        
    param_sum = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("the number of parametrs is:",param_sum)
        
        
    #define Adam optimzer
    param_backbone = [{'params': visual_net.parameters(), 'lr': lr},
                    {'params': audio_net.parameters(), 'lr': lr_big},
                    {'params': fusion_net.parameters(), 'lr': lr_big},
                    {'params': spatial_net.parameters(), 'lr': lr}]
    
    optimizer = torch.optim.Adam(param_backbone, betas=(beta1,0.999), weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # set up loss function
    loss_criterion = torch.nn.MSELoss()
    spatial_loss_criterion = torch.nn.BCEWithLogitsLoss()
    if(len(gpu_ids) > 0 and gpu_available):
        loss_criterion.cuda(gpu_ids[0])
        spatial_loss_criterion.cuda(gpu_ids[0])

    batch_loss, batch_loss1, batch_fusion_loss, batch_rir_loss, batch_spat_const_loss, batch_geom_const_loss = [], [], [], [], [], []
    total_steps = 0
    best_err = float("inf")

    for epoch in range(epochs):
        if gpu_available:
            torch.cuda.synchronize(device=device)
        pbar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)", leave=False)
        for i, data in enumerate(pbar):


            total_steps += batch_size

            ## forward pass
            # zero grad
            optimizer.zero_grad()

            output = model(data, mode='train')
            

            ## compute loss for each model
            # backbone loss
            channel1_spec = data['channel1_spec'].to(device)
            channel2_spec = data['channel2_spec'].to(device)
            difference_loss = loss_criterion(output['binaural_spectrogram'], output['audio_gt'])
            # channel1_loss = loss_criterion(2*output['left_spectrogram']-output['binaural_spectrogram'], output['audio_gt'].detach())
            # channel2_loss = loss_criterion(output['binaural_spectrogram']-2*output['right_spectrogram'], output['audio_gt'].detach())
            
            # 左右聲道與原資料的loss
            channel1_loss = loss_criterion(output['left_spectrogram'], data["channel1_spec"][:,:,:-1,:].to(device))
            channel2_loss = loss_criterion(output['right_spectrogram'], data["channel2_spec"][:,:,:-1,:].to(device))
            
            # 左右聲道loss的平均值
            fusion_loss = (channel1_loss / 2 + channel2_loss / 2)
            
            # backbone的loss = 頻譜差異的loss + 左右聲道loss的平均值
            loss_backbone = lambda_binarual * difference_loss + lambda_f * fusion_loss
            
            # geometric consistency loss 上下一幀的視覺向量特徵差異
            mse_geometry = loss_criterion(output['visual_feature'], output['second_visual_feature']) 
            loss_geometry = torch.maximum(mse_geometry - alpha, torch.tensor(0))
            
            # spatial coherence loss 音訊翻轉任務的loss
            c = output['cl_pred']
            c_pred = output['label']
            loss_spatial = spatial_loss_criterion(c, c_pred)
            
            # combine loss
            # 現在沒有RIR loss
            loss = lambda_b * loss_backbone + lambda_g * loss_geometry + lambda_s * loss_spatial
            
            # batch_loss = 總損失
            batch_loss.append(loss.item())
            # batch_loss1 = 頻譜差異的loss
            batch_loss1.append(difference_loss.item())
            # batch_fusion_loss = 左右聲道loss的平均值
            batch_fusion_loss.append(fusion_loss.item())
            # batch_spat_const_loss = 音訊翻轉任務的loss
            batch_spat_const_loss.append(loss_spatial.item())
            # batch_geom_const_loss = 上下一幀的視覺向量特徵差異
            batch_geom_const_loss.append(loss_geometry.item())
            
            # update optimizer
            #optimizer_resnet.zero_grad()
            optimizer.zero_grad()
            
            loss.backward()
            
            #optimizer_resnet.step()
            optimizer.step()
            
            # update pbar
            avg_loss = sum(batch_loss) / len(batch_loss)
            pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})


            # 顯示當前batch的Loss並且寫入tensorboard
            if(i % display_freq == 0):
                if spec_debug:
                    debug_dataset(data, epoch)
                    debug_dataset(output, epoch, flag='output')
                # print('Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
                avg_loss = sum(batch_loss) / len(batch_loss)
                avg_loss1 = sum(batch_loss1) / len(batch_loss1)
                avg_fusion_loss = sum(batch_fusion_loss) / len(batch_fusion_loss)
                avg_spat_const_loss = sum(batch_spat_const_loss) / len(batch_spat_const_loss)
                avg_geom_const_loss = sum(batch_geom_const_loss) / len(batch_geom_const_loss)
                # print('Average loss: %.3f' % (avg_loss))
                batch_loss, batch_loss1, batch_fusion_loss, batch_rir_loss, batch_spat_const_loss, batch_geom_const_loss = [], [], [], [], [], []
                writer.add_scalar('data/loss', avg_loss, total_steps)
                writer.add_scalar('data/loss1', avg_loss1, total_steps)
                writer.add_scalar('data/fusion_loss', avg_fusion_loss, total_steps)
                writer.add_scalar('data/spat_const_loss', avg_spat_const_loss, total_steps)
                writer.add_scalar('data/geom_const_loss', avg_geom_const_loss, total_steps)
                    
        # 存取模型權重至checkpoints資料夾
        if(epoch % save_latest_freq == 0):
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            torch.save(visual_net.state_dict(), os.path.join('.', checkpoints_dir2, 'visual_latest.pth'))
            torch.save(audio_net.state_dict(), os.path.join('.', checkpoints_dir2, 'audio_latest.pth'))
            torch.save(fusion_net.state_dict(), os.path.join('.', checkpoints_dir2, 'fusion_latest.pth'))
            torch.save(spatial_net.state_dict(), os.path.join('.', checkpoints_dir2, 'classifier_latest.pth'))
            torch.save(generator.state_dict(), os.path.join('.', checkpoints_dir2, 'generator_latest.pth'))

        if(epoch % validation_freq == 0):
            model.eval()
            dataset.mode = 'val'
            print('Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
            val_err = display_val(model, loss_criterion, writer, total_steps, data_loader_val)
            print('end of display \n')
            model.train()
            dataset.mode = 'train'
            #save the model that achieves the smallest validation error
            if val_err < best_err:
                best_err = val_err
                print('saving the best model (epoch %d, total_steps %d) with validation error %.3f\n' % (epoch, total_steps, val_err))
                torch.save(visual_net.state_dict(), os.path.join('.', checkpoints_dir2, 'visual_best.pth'))
                torch.save(audio_net.state_dict(), os.path.join('.', checkpoints_dir2, 'audio_best.pth'))
                torch.save(fusion_net.state_dict(), os.path.join('.', checkpoints_dir2, 'fusion_best.pth'))
                torch.save(spatial_net.state_dict(), os.path.join('.', checkpoints_dir2, 'classifier_best.pth'))
                torch.save(generator.state_dict(), os.path.join('.', checkpoints_dir2, 'generator_best.pth'))
        if (epochs * lr_decrese_fq) > 0 and epochs % lr_decrese_fq:
            lr_decrease(optimizer)
        
    