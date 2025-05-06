from easydict import EasyDict as Edict

cfg = Edict()

# ############################################
#                  model
# ############################################
cfg.model = Edict()
cfg.model.search_size = [256, 256] #[224, 224]  # h, w
cfg.model.template_size = [128, 128] #[112, 112]

# backbone
cfg.model.backbone = Edict()
cfg.model.backbone.search_size = cfg.model.search_size
cfg.model.backbone.template_size = cfg.model.template_size
cfg.model.backbone.type = 'MAEEncode'
#
cfg.model.backbone.arch = 'base'
cfg.model.backbone.weights = './checkpoints/translate_template_common_pretrain/translate_template_common_pretrain_E500.pth'  # absolute path to pretrain checkpoint

# cfg.model.backbone.arch = 'small'
# cfg.model.backbone.weights = '/home/space/Documents/Experiments/BaseT/pretrain/vit-s16-moco3.pth'  # absolute path to pretrain checkpoint

#
cfg.model.backbone.lr_mult = 0.1
cfg.model.backbone.train_layers = [11, 10, 9, 8, 7, 6]# [11, 10, 9] [11, 10, 9, 8, 7, 6] []
cfg.model.backbone.train_all = (cfg.model.backbone.lr_mult > 0) & (len(cfg.model.backbone.train_layers) == 0)


# # backbone
# cfg.model.backbone = Edict()
# cfg.model.backbone.search_size = cfg.model.search_size
# cfg.model.backbone.template_size = cfg.model.template_size
# cfg.model.backbone.type = 'ResNet'
# #
# cfg.model.backbone.arch = 'resnet50'
# cfg.model.backbone.norm_layer = None  # None for frozenBN
# cfg.model.backbone.use_pretrain = True
# cfg.model.backbone.dilation_list = [False, False, False]  # layer2 layer3 layer4, in increasing depth order
# #
# cfg.model.backbone.top_layer = 'layer3'
# cfg.model.backbone.use_inter_layer = False
# #
# cfg.model.backbone.lr_mult = 0.1
# cfg.model.backbone.train_layers = []
# cfg.model.backbone.train_all = (cfg.model.backbone.lr_mult > 0) & (len(cfg.model.backbone.train_layers) == 0)



# nlp
cfg.model.use_language = False

# neck
cfg.model.neck = Edict()
cfg.model.neck.search_size = []
cfg.model.neck.template_size = []
cfg.model.neck.in_channels_list = []
cfg.model.neck.inter_channels = 256
cfg.model.neck.type = 'DWCorr'
#
cfg.model.neck.transformer = Edict()
cfg.model.neck.transformer.in_channels = 256
cfg.model.neck.transformer.num_heads = 8
cfg.model.neck.transformer.dim_feed = 2048
cfg.model.neck.transformer.dropout = 0.1
cfg.model.neck.transformer.activation = 'relu'
cfg.model.neck.transformer.norm_before = False
cfg.model.neck.transformer.return_inter_decode = False
cfg.model.neck.transformer.num_encoders = 0
cfg.model.neck.transformer.num_decoders = 4

# head
cfg.model.head = Edict()
cfg.model.head.search_size = []
cfg.model.head.stride = -1
cfg.model.head.in_channels = cfg.model.neck.inter_channels
cfg.model.head.inter_channels = 256
cfg.model.head.type = 'Corner' #'Center', 'Corner'

# criterion
cfg.model.criterion = Edict()
cfg.model.criterion.type = 'DETR' #, 'TBSI'
#
cfg.model.criterion.alpha_giou = 2
cfg.model.criterion.alpha_l1 = 5
cfg.model.criterion.alpha_conf = 1

# ############################################
#                  data
# ############################################
cfg.data = Edict()
cfg.data.num_works = 6 #8
cfg.data.batch_size = 8 #32
cfg.data.sample_range = 200 #200
#
cfg.data.datasets_train = []
cfg.data.datasets_val = []

cfg.data.path = '/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/LasHeR/LasHeR_dp/crop0/'
cfg.data.annotation = '/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/LasHeR/LasHeR_dp/train.json'
cfg.data.use = 64000#19000
cfg.data.VIDEO_QUALITY = 0.40
cfg.data.MEMORY_NUM = 2
cfg.data.FAR_SAMPLE = 3
#
cfg.data.num_samples_train = 64000
cfg.data.num_samples_val = 1000
#
cfg.data.search_size = cfg.model.search_size
cfg.data.search_scale_f = 4.0#4.0
cfg.data.search_jitter_f = [0.5, 3]#[0.0, 0.0]#[0.5, 3]
#
cfg.data.template_size = cfg.model.template_size
cfg.data.template_scale_f = 2
cfg.data.template_jitter_f = [0.0, 0.0]#[0.0, 0.0]
cfg.data.padding = 2

# ############################################
#                  trainer
# ############################################
cfg.trainer = Edict()
cfg.trainer.deterministic = False
cfg.trainer.seed = 123
cfg.trainer.print_interval = None
cfg.trainer.start_epoch = 61
cfg.trainer.end_epoch = 100
cfg.trainer.UNFIX_EPOCH = 10
cfg.trainer.UNFIX_EPOCH2 = 21#21
cfg.trainer.UNFIX_EPOCH3 = 121#51
cfg.trainer.stage = 'stage2' #stage1, stage2
cfg.trainer.sync_bn = False
cfg.trainer.amp = True
#
cfg.trainer.resume = None
cfg.trainer.pretrain = None
cfg.trainer.pretrain_lr_mult = 0.1
#
cfg.trainer.val_interval = 5
cfg.trainer.save_interval = 5

# distributed train
cfg.trainer.dist = Edict()
cfg.trainer.dist.distributed = False
cfg.trainer.dist.master_addr = None
cfg.trainer.dist.master_port = None
#
cfg.trainer.dist.device = 'cuda'
cfg.trainer.dist.world_size = None
cfg.trainer.dist.local_rank = None
cfg.trainer.dist.rank = None

# optimizer
cfg.trainer.optim = Edict()
cfg.trainer.optim.type = 'AdamW'
#
cfg.trainer.optim.base_lr = 3e-4#3e-4#7.5e-4 7.5e-4 #1e-4 ,from 25 to 1e-5
cfg.trainer.optim.momentum = 0.9
cfg.trainer.optim.weight_decay = 1e-4#1e-4
#
cfg.trainer.optim.grad_clip_norm = None
cfg.trainer.optim.grad_acc_steps = 2
cfg.trainer.print_interval = cfg.trainer.optim.grad_acc_steps

# lr_scheduler
cfg.trainer.lr_scheduler = Edict()
cfg.trainer.lr_scheduler.type = 'multi_step'  # lr_scheduler list | 'cosine' 'multi_step'
cfg.trainer.lr_scheduler.warmup_epoch = 0 #0
cfg.trainer.lr_scheduler.milestones = [0] #240

# tracker
cfg.tracker = Edict()
cfg.tracker.score_threshold = 0.0
cfg.tracker.name = f'translate_track'

if __name__ == '__main__':
    import pprint

    print('\n' + pprint.pformat(cfg))
