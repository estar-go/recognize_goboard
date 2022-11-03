import time

class Config(object):

    train_file_path = '/data/datasets/godatasets/goboard_datasets/train/'
    test_file_path = '/data/datasets/godatasets/goboard_datasets/test/'
    mode = 'train'
    cuda = True
    ngpu = 2
    workers = 4
    image_size = 304
    # image_size = 608

    class_num = 3
    dataloader_type = 'file'
    cnn_type = 'goCNN'
    # cnn_type = 'conveXt_tiny_96x192x384'
    # cnn_type = 'conveXt_base_128x256x512'
    # cnn_type = 'conveXt_tiny_192x384x768'
    cnn_type = 'pplcx10'
    # cnn_type = 'pplcx05'
    # cnn_type = 'conveXt_tiny_48x96x192x384'
    # cnn_type = 'ndconveXt_tiny_48x96x192'
    # cnn_type = 'ndconveXt_tiny_48x96'
    # cnn_type = 'ndconveXt_tiny_32x64x128x256'

    alpha = 1e-3    
    batch_scale = 8
    batch_size = int(4 * batch_scale)

    lr = 1e-3
    beta = 0.5
    optims_type = 'Radam' # [adam, sgd, adadelta, adamW, Radam, Nadam]
    momentum = 0.9
    manualSeed = 1234

    epoch_num = 20
    show_iter = 2 # batch_size // (ngpu * 20)
    print_iter = 2
    resume_epoch = 0
    model_output_base_path = f'/data/projects/weights/training/goboard_recogn/work_dir/{cnn_type}_{optims_type}_' + time.strftime('%Y%m%d-%H%M', time.localtime())[:-1]  
    log_name = '/train.log'
    vis_log = f'{model_output_base_path}/vis.log'
    pretrained = ''
    resume_from_path = '/data/projects/weights/training/goboard_recogn/work_dir/conveXt_tiny_48x96x192x384_Radam_20220721-024/best_models.pth'
    resume_from_path = '/data/projects/weights/training/goboard_recogn/work_dir/pplcx10_Radam_20220721-094/best_models.pth'