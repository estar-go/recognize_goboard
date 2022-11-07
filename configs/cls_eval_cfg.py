class Config(object):
    cuda = True
    image_size = 224

    class_num = 2
    cnn_type = 'origin_pplcx10'
    # cnn_type = 'conveXt_tiny_48x96x192x384'
    # cnn_type = 'ndconveXt_tiny_32x64x128x256'
    weight_path = '/data/projects/weights/training/goboard_recogn/work_dir/conveXt_tiny_48x96x192x384_Radam_20220721-024/best_models.pth'
    weight_path = '/mnt/data/disk0/projects/weights/training/goboard_recogn/work_dir/classifier_origin_pplcx10_Radam_20221104-064/best_models.pth'
    device = 'cuda:0'
    device = 'cpu'
    
    log_base_path = 'logs/'
    log_name = 'server.log'

    server_port = 8002