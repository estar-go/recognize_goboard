# recognize goboard

## 项目说明
* 识别传入棋盘并输出成19x19的矩阵
* 判断输入的图片是否是纯棋盘

## 结构
* `configs` 配置文件夹 包含训练和测试配置
* `models` 包含网络`backbone`等神经网络代码
* `dataloaders` 网络训练用的`dataloader`类
* `utils` 杂七杂八的工具类
* `process_data` 处理数据格式的工具集合
* `train.py` 训练入口
* `loss` 网络训练各种`loss function`

## 使用说明
* 1. 在`configs/train_cfg.py`中修改训练参数
    * `train_file_path`: 训练数据目录(目前仅支持lmdb格式)
    * `test_file_path`: 测试数据目录(目前仅支持lmdb格式)
    * `ngpu`: 显卡数量
    * `workers`: 每个dataloader多线程加载数量(根据CPU和内存大小决定)
    * `goboard_size`: 盘面大小 默认19
    * `in_feature_dim`: 输入feature维度，目前3,4,23可选，默认23，修改需要更换trainer里的dataloader
    * `cnn_type`: backbone类型
        * convneXt: 目前效果SOTA的视觉backbone，但是在go上效果并不好，需要再改
    * `lr`: 训练learning rate
    * `batch_size`: 训练总batch size大小（实际每张卡上batch size 则为 batch_size//ngpu）
    * `adam`: 是否选用adam做优化器(暂时只用adam，所以默认True)
    * `epoch_num`: 总训练轮数
    * `show_iter`: 多少步显示一次盘面可视化
    * `print_iter`: 多少步print一次loss到log里
    * `model_output_base_path`: 模型保存位置
    * `resume_from_path`: 从之前的模型继续训练
* 2.在环境中运行`python -m visdom.server &` 启动visdom进行可视化服务
* 3.修改`run_ddp_train.sh`将用的显卡`CUDA_VISIBLE_DEVICES=0,...,n-1`和对应的显卡数量`--nproc_per_node=n`。修改后运行该脚本启动训练`sh run_ddp_train.sh`
    
## TODO
* 判断输入的图片是否是纯棋盘

## DONE
* init project
* 了解数据格式，下载并清洗数据
    * 下载各种开源棋盘数据转换为`SGF`
* 识别传入棋盘并输出成19x19的矩阵label_smooth_ce_loss.py`
* 训练代码`ddp_trainer.py`
* 完善可视化结果工具 使用visdom进行可视化