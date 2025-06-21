import argparse
import logging
import shutil
import sys
import time
import warnings
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

warnings.filterwarnings('ignore')

from dataloader.dataset import *
from dataloader.aug import *
from utils.util_f import *
from utils.losses import BinaryFocalLoss

from validation import crossval, crossval_last
from networks.hvan import HVAN as model1

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default='HVAF')
# parser.add_argument("--exp_name", type=str, default='debug')
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--k_num", type=int, default=1)
parser.add_argument("--train_bs", type=int, default=8)
parser.add_argument("--val_bs", type=int, default=8)
parser.add_argument("--write_image", type=bool, default=False)
parser.add_argument("--save_root_path", type=str, default='./work_dir')  # comparative method

parser.add_argument("--max_epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--transverse_path", type=str, default='/datasets/prostate_trus/transverse')
parser.add_argument("--sagittal_path", type=str, default='/datasets/prostate_trus/sagittal')
parser.add_argument("--fold_neg", type=str, default='/dataloader/negative_cases.txt')
parser.add_argument("--fold_pos", type=str, default='/dataloader/positive_cases.txt')
parser.add_argument("--inter_log", type=int, default=20)
parser.add_argument("--seed1", type=int, default=2000)
parser.add_argument("--per_val_epoch", type=int, default=1)
parser.add_argument("--rm_exp", type=bool, default=False)
args = parser.parse_args()


def create_model():
    model = model1(1, 1)
    model = nn.DataParallel(model.cuda())
    return model


def save_parameter(exp_save_path, d=True):
    delete = True if os.path.basename(exp_save_path) == 'debug' else d
    if os.path.exists(exp_save_path) is True:
        assert delete is True, args.exp_name
        shutil.rmtree(exp_save_path)
    os.makedirs(exp_save_path)
    os.makedirs(os.path.join(exp_save_path, 'ckp_model'))
    # save this .py
    py_path_old = sys.argv[0]
    py_path_new = os.path.join(exp_save_path, os.path.basename(py_path_old))
    shutil.copy(py_path_old, py_path_new)
    logging.basicConfig(filename=os.path.join(exp_save_path, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info('PID: {}'.format(os.getpid()))


def reproduce(seed1):
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(seed1)
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)


def train():
    reproduce(args.seed1)
    exp_time = time.localtime()
    exp_time_format = time.strftime("%m-%d-%H-%M", exp_time)
    exp_save_path = os.path.join(args.save_root_path, '{}'.format(args.exp_name))
    save_parameter(exp_save_path, args.rm_exp)
    writer = SummaryWriter(log_dir=exp_save_path)

    print('-------------------------------------- setting --------------------------------------')
    print("experiment name: {}".format(exp_save_path.split('/')[-1]))
    print("time: ", exp_time_format)
    print("gpu: {}".format(args.gpu))
    print("lr: {}".format(args.lr))
    print("Fold: {}".format(args.k_num))
    print('-------------------------------------- setting --------------------------------------')

    # load data
    pos_list_tvt, neg_list_tvt = get_train_val_test_balance(args.fold_neg, args.fold_pos)
    # pos_list, neg_list = list(chain(*pos_list_tvt)), list(chain(*neg_list_tvt))
    pos_list, neg_list = pos_list_tvt, neg_list_tvt

    box_list = ['381', '482', '428', '399']
    pos_nobox_list = np.array([item for item in pos_list if item not in box_list])
    neg_list = np.array(neg_list)

    kf = KFold(n_splits=args.k_num, shuffle=True, random_state=args.seed1)
    k_index = 0

    best_metrics = np.zeros((args.k_num, 5))
    last_metrics = np.zeros((args.k_num, 5))
    for (train_pos_index, val_pos_index), (train_neg_index, val_neg_index) in zip(kf.split(pos_nobox_list),
                                                                                  kf.split(neg_list)):
        # if k_index == 1:
        #     k_index += 2
        #     break
        train_pos_list, val_pos_list = list(pos_nobox_list[train_pos_index]), list(pos_nobox_list[val_pos_index])
        train_neg_list, val_neg_list = list(neg_list[train_neg_index]), list(neg_list[val_neg_index])

        train_dataset = PD3C_B_E(train_pos_list + train_neg_list,
                                 args.transverse_path, args.sagittal_path,
                                 transform=transforms.Compose([
                                     NormalizationFrame('volume1'),
                                     NormalizationFrame('volume2'),
                                     SynthesizeTransView('volume1'),
                                     RandomTranslation(p=0.2),
                                     RandomRotateTransform(angle_range=(-10, 10),
                                                           p_per_sample=0.20),
                                     ToTensor()]))

        val_datasets = PD3C_B_E(box_list + val_pos_list + val_neg_list,
                                args.transverse_path, args.sagittal_path,
                                transform=transforms.Compose([
                                    NormalizationFrame('volume1'),
                                    NormalizationFrame('volume2'),
                                    SynthesizeTransView('volume1'),
                                    ToTensor()]))

        def worker_init_fn(worker_id):
            random.seed(args.seed1 + worker_id)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.train_bs,
                                      shuffle=True,
                                      num_workers=2,
                                      pin_memory=False,
                                      worker_init_fn=worker_init_fn)

        val_dataloader = DataLoader(val_datasets,
                                    batch_size=args.val_bs,
                                    shuffle=False)

        save_train_val_test_txt([train_pos_list + train_neg_list, val_pos_list + val_neg_list],
                                exp_save_path, k_idx=k_index)
        model = create_model()
        model_py_path = model.module.get_model_py_path()
        model_py_save_path = os.path.join(exp_save_path, os.path.basename(model_py_path))
        if os.path.exists(model_py_save_path) is False:
            shutil.copy(model_py_path, model_py_save_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        celoss = BinaryFocalLoss(alpha=0.32, gamma=2.0)

        n_total_iter = 0
        loss_log_list = []

        for epoch in range(args.max_epoch):
            lr = poly_lr(epoch, args.max_epoch, args.lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss_epoch = 0.0
            i_batch = 0

            train_prefetcher = data_prefetcher(train_dataloader)
            batch_nobox = train_prefetcher.next()

            y_true = []
            y_pred_sm = []

            model.train()
            while batch_nobox is not None:
                start_time = time.perf_counter()
                train_transverse = batch_nobox['volume1'].cuda().float()
                train_sagittal = batch_nobox['volume2'].cuda().float()
                train_label = batch_nobox['cspca'].cuda()

                optimizer.zero_grad()
                out = model(train_transverse, train_sagittal)
                loss = celoss(out.view(-1), train_label.float())

                y_true.extend(train_label)
                y_pred_sm.extend(torch.sigmoid(out))

                loss_epoch += loss
                # loss_ce_epoch += loss_ce
                loss.backward()

                optimizer.step()
                batch_nobox = train_prefetcher.next()

                n_total_iter += 1
                i_batch += 1

                loss_log_list.append(loss.item())
                if n_total_iter % args.inter_log == 0:
                    end_time = time.perf_counter()
                    used_time = (end_time - start_time) / 60
                    logging.info("[Epoch: %4d/%d] [Train index: %2d/%d] [loss: %.4f] [used time: %.2fmin]" % (
                        epoch, args.max_epoch, i_batch, len(train_dataloader),
                        np.mean(loss_log_list), used_time))
                    loss_log_list = []

            y_true = torch.stack(y_true, dim=0)
            y_pred_sm = torch.stack(y_pred_sm, dim=0)
            fpr, tpr, _ = roc_curve(y_true.cpu().data.numpy(), y_pred_sm.cpu().data.numpy(), pos_label=1)
            train_auc = auc(fpr, tpr)

            writer.add_scalar(f"LR/lr{k_index}", lr, global_step=epoch)
            writer.add_scalar(f"train/auc{k_index}", train_auc, global_step=epoch)
            writer.add_scalar(f"Loss{k_index}/loss", loss_epoch / (i_batch), global_step=epoch)
            logging.info("fold: {}  epoch: {}  train auc: {:.4f}".format(k_index, epoch, train_auc))

            if (epoch + 1) % args.per_val_epoch == 0:
                logging.info(f"-------------------fold{k_index} epoch{epoch} evaluating------------------")
                best_metrics = crossval(model, val_dataloader, writer, epoch,
                                              exp_save_path, best_metrics, k_index, 'both')
            if epoch + 1 == args.max_epoch:
                last_metrics = crossval_last(model, val_dataloader, last_metrics, k_index, 'both')
        torch.save(model.module.state_dict(), '{}/ckp_model/model_fold{}_last.pth'.format(exp_save_path, k_index))
        k_index += 1

    logging.info('Best result/fold')
    for tmp in range(k_index):
        logging.info('fold{} AUC:{:.4f} F1:{:.4f} ACC:{:.4f} Sen:{:.4f} Spe:{:.4f}'
                     .format(tmp, best_metrics[tmp, 0], best_metrics[tmp, 1], best_metrics[tmp, 2],
                             best_metrics[tmp, 3], best_metrics[tmp, 4]))

    logging.info('Mean Result')
    mean_metrics = np.nanmean(best_metrics, axis=0, keepdims=False)
    std_metrics = np.nanstd(best_metrics, axis=0, keepdims=False)
    logging.info('AUC:{:.4f}±{:.4f} F1:{:.4f}±{:.4f} ACC:{:.4f}±{:.4f}\n'
                 'Sen:{:.4f}±{:.4f} Spe:{:.4f}±{:.4f}\n'
                 .format(mean_metrics[0], std_metrics[0],
                         mean_metrics[1], std_metrics[1],
                         mean_metrics[2], std_metrics[2],
                         mean_metrics[3], std_metrics[3],
                         mean_metrics[4], std_metrics[4]))

    logging.info('result of final epoch/fold')
    for tmp in range(k_index):
        logging.info('fold{} AUC:{:.4f} F1:{:.4f} ACC:{:.4f} Sen:{:.4f} Spe:{:.4f}'
                     .format(tmp, last_metrics[tmp, 0], last_metrics[tmp, 1], last_metrics[tmp, 2],
                             last_metrics[tmp, 3], last_metrics[tmp, 4]))

    logging.info('Mean Result of final epoch')
    mean_metrics = np.nanmean(last_metrics, axis=0, keepdims=False)
    std_metrics = np.nanstd(last_metrics, axis=0, keepdims=False)
    logging.info('AUC:{:.4f}±{:.4f} F1:{:.4f}±{:.4f} ACC:{:.4f}±{:.4f}\n'
                 'Sen:{:.4f}±{:.4f} Spe:{:.4f}±{:.4f}\n'
                 .format(mean_metrics[0], std_metrics[0],
                         mean_metrics[1], std_metrics[1],
                         mean_metrics[2], std_metrics[2],
                         mean_metrics[3], std_metrics[3],
                         mean_metrics[4], std_metrics[4],))
    
    writer.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train()
