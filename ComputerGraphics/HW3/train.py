import argparse
import h5py
import metric
import numpy as np
import os
import torch
import wandb

from config import DATASET_DIR
from datasets.lidarcap_dataset import collate, TemporalDataset
from modules.geometry import rotation_matrix_to_axis_angle
from modules.regressor import *
from modules.loss import Loss
from tools import common, crafter, multiprocess
from tools.util import save_smpl_ply
from tqdm import tqdm
torch.set_num_threads(1)

class MyTrainer(crafter.Trainer):
    def forward_backward(self, inputs):
        output = self.net(inputs)
        loss, details = self.loss_func(**output)
        loss.backward()
        return details

    def forward_val(self, inputs):
        output = self.net(inputs)
        loss, details = self.loss_func(**output)
        return details

    def forward_net(self, inputs):
        output = self.net(inputs)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # bs
    parser.add_argument('--bs', type=int, default=16,
                        help='input batch size for training (default: 24)')
    parser.add_argument('--eval_bs', type=int, default=4,
                        help='input batch size for evaluation')
    # threads
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads (default: 4)')
    # gpu
    parser.add_argument('--gpu', type=int,
                        default=[0], help='-1 for CPU', nargs='+')
    # lr
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    # epochs
    parser.add_argument('--epochs', type=int, default=200,
                        help='Traning epochs (default: 200)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Traning epochs (default: 100)')
    # dataset
    parser.add_argument("--dataset", type=str, required=True)
    # debug
    parser.add_argument('--debug', action='store_true', help='For debug mode')
    # eval or visual
    parser.add_argument('--eval', default=False, action='store_true',
                        help='evaluation the trained model')

    parser.add_argument('--visual', default=False, action='store_true',
                        help='visualization the result ply')

    # extra things, ignored
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='the saved ckpt needed to be evaluated or visualized')

    # wandb
    parser.add_argument('--project', type=str, default='lidarcap')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--name', type=str, default='lidarcap-scy')
    
    # metric
    parser.add_argument('--metric-file-name', type=str, default='metric.txt')
    
    # model version
    parser.add_argument('--version', type=str, default='v3')
    parser.add_argument('--save-ckp-dir', type=str, default=None)

    args = parser.parse_args()

    if args.debug:
        os.environ['WANDB_MODE'] = 'dryrun'

    wandb.init(project=args.project, 
               entity=args.entity,
               name=args.name)
    
    wandb.config.update(args, allow_val_change=True)
    config = wandb.config

    iscuda = common.torch_set_gpu(config.gpu)
    common.make_reproducible(iscuda, 0)
    wandbid = [x for x in wandb.run.dir.split('/') if wandb.run.id in x][-1]
    
    # model save models in training
    if args.save_ckp_dir:
        model_dir = os.path.join('output', args.save_ckp_dir)
    else:
        model_dir = os.path.join('output', wandbid, 'model')
    
    os.makedirs(model_dir, exist_ok=True)

    dataset_name = args.dataset
    from yacs.config import CfgNode
    cfg = CfgNode.load_cfg(open('base.yaml'))
    # Load training and validation data
    if args.eval:
        test_dataset = TemporalDataset(cfg.TestDataset)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.eval_bs, num_workers=config.threads, pin_memory=True, collate_fn=collate)
        loader = {'Test': test_loader}

    else:
        train_dataset = TemporalDataset(cfg.TrainDataset, dataset=dataset_name, train=True)
        train_loader = torch.utils.data.DataLoader(
train_dataset, batch_size=config.bs, shuffle=True, num_workers=config.threads, pin_memory=True, collate_fn=collate)
        valid_dataset = TemporalDataset(cfg.TrainDataset, dataset=dataset_name, train=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config.bs, shuffle=False, num_workers=config.threads, pin_memory=True, collate_fn=collate)
        loader = {'Train': train_loader, 'Valid': valid_loader}

    if args.version == 'v1':
        net = Regressor()
    elif args.version == 'v2':
        net = RegressorV2()
    elif args.version == 'v3':
        net = RegressorV3()
    elif args.version == 'v4':
        net = RegressorV4()
    else:
        raise Exception('Wrong argument: --version')
    
    loss = Loss()

    if args.ckpt_path is not None:
        save_model = torch.load(args.ckpt_path)['state_dict']
        model_dict = net.state_dict()
        state_dict = {k: v for k, v in save_model.items()
                      if k in model_dict.keys()}
        model_dict.update(state_dict)
        net.load_state_dict(model_dict)

    # Define optimizer
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                 lr=config.lr, weight_decay=1e-4)
    sc = {'factor': 0.9, 'patience': 1, 'threshold': 0.01, 'min_lr': 0.00000003}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=sc['factor'], patience=sc['patience'],
        threshold_mode='rel', threshold=sc['threshold'], min_lr=sc['min_lr'],
        verbose=True)

    # Instance tr
    wandbid = args.ckpt_path.split('.')[0] if args.eval else None
    train = MyTrainer(net, loader, loss, optimizer)
    if iscuda:
        train = train.cuda()

    if args.eval:
        if args.visual:

            visual_dir = os.path.join('visual', wandbid, dataset_name)
            os.makedirs(visual_dir, exist_ok=True)
            final_loss, pred_rotmats, pred_vertices = train(
                epoch=1, train=False, test=True, visual=True)
            n = len(pred_vertices)
            filenames = [os.path.join(
                visual_dir, '{}.ply'.format(i + 1)) for i in range(n)]
            multiprocess.multi_func(save_smpl_ply, 32, len(
                pred_vertices), 'saving ply', False, pred_vertices, filenames)

        else:
            final_loss, pred_rotmats = train(
                epoch=1, train=False, visual=False, test=True)
        print('EVAL LOSS', final_loss['loss'])

        pred_poses = []
        for pred_rotmat in tqdm(pred_rotmats):
            pred_poses.append(rotation_matrix_to_axis_angle(torch.from_numpy(pred_rotmat.reshape(-1, 3, 3))).numpy().reshape((-1, 72)))
        pred_poses = np.stack(pred_poses)

        # test_dataset_filename = os.path.join(DATASET_DIR, '{}_test.hdf5'.format(dataset_name))
        test_dataset_filename = os.path.join(DATASET_DIR, '{}_test.hdf5'.format(dataset_name))
        
        test_data = h5py.File(test_dataset_filename, 'r')
        gt_poses = test_data['pose'][:]
        metric.output_metric(pred_poses.reshape(-1, 72), gt_poses.reshape(-1, 72), args.metric_file_name)

    else:
        # Training loop
        mintloss = float('inf')
        minvloss = float('inf')
        for epoch in range(1, config.epochs + 1):
            print('')

            train_loss_dict = train(epoch)
            val_loss_dict = train(epoch, train=False)
            train_loss_log = {'train' + k: v for k,
                              v in train_loss_dict.items()}
            val_loss_log = {'val' + k: v for k, v in val_loss_dict.items()}
            epoch_logs = train_loss_log
            epoch_logs.update(val_loss_log)
            wandb.log(epoch_logs)

            # save model in this epoch
            # if this model is better, then save it as best
            if train_loss_dict['loss'] <= mintloss:
                mintloss = train_loss_dict['loss']
                best_save = os.path.join(model_dir, 'best-train-loss.pth')
                torch.save({'state_dict': net.state_dict()}, best_save)
                common.hint(f"Saving best train loss model at epoch {epoch}")
            if val_loss_dict['loss'] <= minvloss:
                minvloss = val_loss_dict['loss']
                best_save = os.path.join(model_dir, 'best-valid-loss.pth')
                torch.save({'state_dict': net.state_dict()}, best_save)
                common.hint(f"Saving best valid loss model at epoch {epoch}")

            common.clean_summary(wandb.run.summary)
            wandb.run.summary["best_train_loss"] = mintloss
            wandb.run.summary["best_valid_loss"] = minvloss

            # scheduler
            scheduler.step(train_loss_dict['loss'])


"""
    
split-1, epoch=79:
5,6,8,25,26,27,28,30,31,32,33,34,35,36,37,38,39,40,42
7,24,29,41
    EVAL LOSS 0.02669398682786429
    accel_error: 42.42551326751709
    mpjpe: 66.80289655923843
    pa_mpjpe: 55.75178936123848
    pve: 85.02522855997086
    pck_30: 0.9008229166666667
    pck_50: 0.966013888888889

split-2, epoch=75:
5,7,8,24,26,27,29,30,31,32,33,34,35,36,37,38,39,41,42
6,25,28,40
    EVAL LOSS 0.03671799302443953
    accel_error: 45.26100307703018
    mpjpe: 77.90874689817429
    pa_mpjpe: 65.43153524398804
    pve: 99.36566650867462
    pck_30: 0.862182866820041
    pck_50: 0.948595667177914

split-3, epoch=80:
5,6,7,8,25,26,27,28,29,30,31,33,34,35,36,37,39,40,41
24,32,38,42
    EVAL LOSS 0.0318373475814226
    accel_error: 42.48834401369095
    mpjpe: 78.49521189928055
    pa_mpjpe: 65.39293378591537
    pve: 98.52030128240585
    pck_30: 0.8657561129273942
    pck_50: 0.9558323968124234

-----------------------------------

以下改进都是基于 split-3 划分方式：

Replace GRU with transformer encoder:
LiDARCapv2, epoch=89
n_heads = 8
n_layers = 3
Parameters:
    before:  136429.1171875 KB
    after:  144413.1171875 KB
    EVAL LOSS 0.03421852965702248
    accel_error: 44.30478438735008
    mpjpe: 84.74975824356079
    pa_mpjpe: 67.39835441112518
    pve: 105.08088767528534
    pck_30: 0.8432977029696227
    pck_50: 0.949647101893475
    
Add SENet to fuse the feature:
LiDARCapV3, epoch=28
Parameters:
    before:  136429.1171875 KB
    after:  136942.6171875 KB
    EVAL LOSS 0.036288382399684134
    accel_error: 42.37852618098259
    mpjpe: 88.2103443145752
    pa_mpjpe: 71.74468785524368
    pve: 109.26418006420135
    pck_30: 0.8308419748671843
    pck_50: 0.9437959746628524


Add both SENet and TransformerEncoder:
LiDARCapV4, epoch=61
Parameters:
    before:  136429.1171875 KB
    after:  144926.6171875 KB
    accel_error: 41.864559054374695
    mpjpe: 82.06760138273239
    pa_mpjpe: 66.7903870344162
    pve: 102.69425809383392
    pck_30: 0.8535973045225446
    pck_50: 0.9532229090042229

-----------------------------------
All in epoch=10

V1:
EVAL LOSS 0.04190502362934495
accel_error: 45.770421624183655
mpjpe: 102.88787633180618
pa_mpjpe: 85.00783890485764
pve: 127.91843712329865
pck_30:0.7861177802751668
pck_50:0.9139401137447214

V2:
accel_error: 44.692616909742355
mpjpe: 108.02476853132248
pa_mpjpe: 91.12952649593353
pve: 133.68940353393555
pck_30:0.7650311606048223
pck_50:0.8996049584525269

V3:
EVAL LOSS 0.042920006539526524
accel_error: 44.996317476034164
mpjpe: 106.59985989332199
pa_mpjpe: 90.23744612932205
pve: 132.37610459327698
pck_30:0.7717709184715978
pck_50:0.9068640597330063

V4:
EVAL LOSS 0.04026160759087857
accel_error: 43.90769079327583
mpjpe: 99.21376407146454
pa_mpjpe: 81.35108649730682
pve: 123.92357736825943
pck_30:0.7982350837760522
pck_50:0.9195156058438906
"""