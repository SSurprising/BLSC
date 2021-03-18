0. setup.py
运行一遍所有脚本
所有输出目录，都填在绝对路径'/output/'下，勿填相对路径

各个文件运行需要的路径参数
1. prerpocessed.py
    对原始数据进行预处理，然后保存为新的数据集
    1) ct_path, 原始CT数据路径
    2) gt_path, 原始label数据路径
    3) new_ct_path, 新的CT数据路径
    2) new_gt_path, 新的label数据路径


2. train.py
    1) CT_dir, Seg_dir, 训练数据的CT路径和label路径
    2) loss_image_path, loss下降图像存储路径，默认为/output/loss_show文件中
    3）save_path, 存储模型的路径，默认在 /output/checkpoints下
    
    
3. val.py
    1) val_ct_dir, val_seg_dir, 验证数据的CT路径和label路径
    2) organ_pred_dir, 分割结果的保存路径
    3) excel_path, 分割指标的存储路径，目前分割指标的可视化也放在该目录下
    4) module_dir, 最优模型存储路径，同train.py中的save_path
    
4. visualization_ct.py
    可视化CT图像
    1) ct_path. 原始CT的路径
    2) raw_gray_save_path. CT数据可视化后的存储路径
    
5. get_gt_edges.py
    根据ground truth获取器官的边缘坐标，存放在json文件中
    1) label_path, ground truth图像的路径
    2) save_path,  生成坐标json文件的存储路径

6. get_pred_edges.py
    根据预测结果获取器官的边缘坐标，存放在json文件中
    1) pred_path, 网络分割结果的路径
    2) save_path,  生成坐标json文件的存储路径
    

7. Todo
    1) test.py，使用没有label的数据进行测试，只给出分割结果，没有指标
    
    
输出文件的存放结构(都在/output文件夹下)：
1) processed_dataset 存放预处理后生成的数据集
        --/train/CT
        --/train/GT
        ...

2) checkpoints 存放训练的模型
    训练得到的最优模型记为best.pth
    我们自己预存放的默认最优模型为standard_best.pth
    当best.pth不存在时，则使用我们默认给出的standard_best.pth进行测试
    
3)loss_show 存放损失变化图片的位置
    
4)pred_result 存放网络预测的结果,如生成的nii文件

5)dice_score （之后改为叫matrices或许更好。。。。）关于评价指标相关的输出，例如存放指标的文件和指标可视化的文件

6)coordinates 存放相关坐标的文件
    如在分割任务中，存放器官轮廓的路径
    
7)visualization 存放各种可视化图片的位置
    如，将CT图像可视化出来存在visualization/CT/文件夹下

