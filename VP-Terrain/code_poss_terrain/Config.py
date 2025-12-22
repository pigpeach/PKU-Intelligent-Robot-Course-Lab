# 定义了项目所需要用到的参数
class config():
    train_filename = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\VP-Terrain\data\_train_7_new.hdf5"
    # valid_filename = "../data/poss-terrain/test_7_normal.hdf5"
    valid_filename = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\VP-Terrain\data\_dark_7_new.hdf5"

    timestep = 100  # 时间步长
    pro_batch_size = 256  # 机体数据训练批次
    vision_batch_size = 32 # 视觉数据训练批次
    feature_size = 9  # 每个步长对应的特征数量
    out_channels = 64  # 卷积输出通道
    output_size = 1  # 最终输出层大小为1
    pro_epochs = 400  # 机体数据训练次数
    vision_epochs = 100  # 视觉数据训练次数

    best_loss = 100  # 损失
    learning_rate = 0.0005  # 学习率
    # ablation study
    vision_backbone = 'mobilenet' # 'mobilenet' or 'resnet' or 'vit'

    alpha = 0.05 #标签率
    


    save_folder = f'./SSL_poss/{vision_backbone}_{alpha}/'  # 保存文件夹
    pro_model_name = 'only_pro'  # 模型名称
    pro_save_path = save_folder + f'{pro_model_name}.pth'  # 最优模型保存路径
    vision_model_name = 'only_vision'  # 模型名称
    vision_save_path = save_folder + f'{vision_model_name}.pth'  # 最优模型保存路径
    best_all_pre = 0.0

    pro_cuda_device = 0
    vision_cuda_device = 0
