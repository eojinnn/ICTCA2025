import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torch.utils.data as data_utils
from tqdm import tqdm
import numpy as np
import random
from torchinfo import summary
import argparse
import os
import cfg

from loss import LabelSmoothing
from model import PGCCPHAT, GCC_freq, GCC_time, Probabilistic_PGCCPHAT, modified_PGCCPHAT
from iceic_model import mPGCCPHAT
from data import LibriSpeechLocations, DelaySimulator, one_random_delay
from cdr_dereverb import cdr_robust

# Librispeech dataset constants
DATA_LEN = 2620
VAL_IDS = [260, 672, 908]  # use these speaker ids for validation
TEST_IDS = [61, 121, 237]  # use these speaker ids for testing
NUM_TEST_WINS = 1
MIN_SIG_LEN = 2  # only use snippets longer than 2 seconds

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str,
                    default='tdoa_exp', help='Name of the experiment')
parser.add_argument('--evaluate', action='store_true',
                    help='Set to true in order to evaluate the model across a range of SNRs and T60s')
parser.add_argument('--data_loc', default = 'D:/datas/librispeech/LibriSpeech/', help = 'librispeech location')
parser.add_argument('--input', type=str, default='freq', help='use stft or not')
args = parser.parse_args()

if not os.path.exists('experiments'):
    os.makedirs('experiments')
if not os.path.exists('experiments/'+args.exp_name):
    os.makedirs('experiments/'+args.exp_name)

if not args.evaluate:
    LOG_DIR = os.path.join('experiments/'+args.exp_name+'/')
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log.txt'), 'w')
    os.system('cp cfg.py experiments/' + args.exp_name + '/cfg.py')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


# for reproducibility
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)

# calculate the max_delay for gcc
max_tau_gcc = int(np.floor(np.linalg.norm(
    cfg.mic_locs_train[:, 0] - cfg.mic_locs_train[:, 1]) * cfg.fs / 343))

# training parameters
max_tau = cfg.max_delay
snr = cfg.snr
t60 = cfg.t60
fs = cfg.fs
sig_len = cfg.sig_len
epochs = cfg.epochs
batch_size = cfg.batch_size
lr = cfg.lr
wd = cfg.wd
label_smooth = cfg.ls

source_locs_train = np.random.uniform(
    low=cfg.xyz_min_train, high=cfg.xyz_max_train, size=(DATA_LEN, 3))
source_locs_val = np.random.uniform(
    low=cfg.xyz_min_train, high=cfg.xyz_max_train, size=(DATA_LEN, 3))
source_locs_test = np.random.uniform(
    low=cfg.xyz_min_test, high=cfg.xyz_max_test, size=(DATA_LEN, 3)) ##3차원 음원 위치 무작위로 2520개 설정

# fetch audio snippets within the range of [0, 2] seconds during training
lower_bound = 0
upper_bound = fs * MIN_SIG_LEN

# create datasets   
train_set = LibriSpeechLocations(source_locs_train, data_root = args.data_loc, split="test-clean")
print('Total data set size: ' + str(len(train_set)))

# create val and test split based on speaker ids
val_set = LibriSpeechLocations(source_locs_val, data_root = args.data_loc, split="test-clean")
test_set = LibriSpeechLocations(source_locs_test, data_root = args.data_loc, split="test-clean")

indices_test = [i for i, ((waveform, sample_rate, speaker_id), pos, seed)
                in enumerate(train_set) if speaker_id in TEST_IDS]
indices_val = [i for i, ((waveform, sample_rate, speaker_id), pos, seed)
               in enumerate(train_set) if speaker_id in VAL_IDS]
indices_train = [i for i, ((waveform, sample_rate, speaker_id), pos, seed)
                 in enumerate(train_set) if speaker_id not in TEST_IDS and speaker_id not in VAL_IDS]

train_set = data_utils.Subset(train_set, indices_train)
val_set = data_utils.Subset(val_set, indices_val)
test_set = data_utils.Subset(test_set, indices_test)

train_len = len(train_set)
val_len = len(val_set)
test_len = len(test_set)

print('Training data size: ' + str(train_len))
print('Validation data size: ' + str(val_len))
print('Test data size: ' + str(test_len))

(waveform, sample_rate, speaker_id), pos, seed = train_set[0]

# get delay statistics for normalization when using regression loss(지연 통계 계산)
if cfg.loss == "mse":
    delays = []
    for i in range(100):
        _, x_, delay, _ = one_random_delay(room_dim=cfg.room_dim_train, fs=fs, t60=0.,
                                           mic_locs=cfg.mic_locs_train, signal=waveform,
                                           xyz_min=cfg.xyz_min_train, xyz_max=cfg.xyz_max_train,
                                           snr=0, anechoic=True)
        delays.append(delay)

    delay_mu = np.mean(delays)
    delay_sigma = np.std(delays) #지연 값의 평균과 표준편차 계산해서 정규화에 사용


# use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

# load model
if cfg.model == 'PGCCPHAT':
    model = PGCCPHAT(max_tau=max_tau_gcc, head=cfg.head, input_shape=args.input)
elif cfg.model == 'mPGCCPHAT':
    model = mPGCCPHAT(max_tau=max_tau_gcc, head=cfg.head)
elif cfg.model == 'Probabilistic_PGCCPHAT':
    model = Probabilistic_PGCCPHAT(max_tau=max_tau_gcc, head=cfg.head, input_shape=args.input)
elif cfg.model == 'modified_PGCCPHAT':
    model = modified_PGCCPHAT(max_tau=max_tau_gcc, head=cfg.head, input_shape=args.input)
else:
    raise Exception("Please specify a valid model")

model = model.to(device)
model.eval()
summary(model, args = [torch.randn(2, 1, 47), torch.randn(2, 1, 47)])


fgcc = GCC_freq(max_tau=max_tau_gcc)
tgcc = GCC_time(max_tau=max_tau_gcc)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

if cfg.loss == 'mse':
    loss_fn = nn.MSELoss()
else:
    raise Exception("Please specify a valid loss function")

delay_simulator_train = DelaySimulator(cfg.room_dim_train, fs, sig_len, t60, cfg.mic_locs_train, max_tau,
                                       cfg.anechoic, train=True, snr=snr, lower_bound=lower_bound, upper_bound=upper_bound)
delay_simulator_val = DelaySimulator(cfg.room_dim_train, fs, sig_len, t60, cfg.mic_locs_train, max_tau,
                                     cfg.anechoic, train=True, snr=snr, lower_bound=lower_bound, upper_bound=upper_bound)
delay_simulator_test = DelaySimulator(cfg.room_dim_test, fs, sig_len, t60, cfg.mic_locs_test, max_tau,
                                      cfg.anechoic, train=False, snr=snr, lower_bound=lower_bound, upper_bound=upper_bound)

print('Using loss function: ' + str(loss_fn))

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=delay_simulator_train,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=delay_simulator_val,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=delay_simulator_test,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


#학습 루프
for e in range(epochs):
    if args.evaluate:
        break
    mae = 0
    gcc_mae = 0
    acc = 0
    gcc_acc = 0
    train_loss = 0
    logs = {}
    model.train()
    pbar_update = batch_size
    with tqdm(total=len(train_set)) as pbar:
        for batch_idx, (x1, x2, delays) in enumerate(train_loader):
            bs = x1.shape[0]

            x1 = x1.to(device) #([32,1,2048])
            x2 = x2.to(device)
            print('x1shape:',x1.shape)

            cc = tgcc(x1, x2)
            shift_gcc = torch.argmax(cc, dim=-1) - max_tau_gcc
            
            delays = delays.to(device)
            if args.input == 'freq':
                X1, X2 = cdr_robust(x1.squeeze(1), x2.squeeze(1))
                y_hat = model(X1, X2)
                x1 = X1
                x2 = X2
            else:
                y_hat = model(x1, x2)
            
            delays_loss = (delays - delay_mu) / delay_sigma
            shift = y_hat * delay_sigma + delay_mu - max_tau
            print(delays.shape)
            print(shift.shape)
            print(max_tau)
            gt = delays - max_tau
            mae += torch.sum(torch.abs(shift-gt))
            gcc_mae += torch.sum(torch.abs(shift_gcc-gt))

            acc += torch.sum(torch.abs(shift-gt) < cfg.t)
            gcc_acc += torch.sum(torch.abs(shift_gcc-gt) < cfg.t)

            loss = loss_fn(y_hat, delays_loss.to(device))

            if torch.isnan(y_hat).any():
                print(" NaN detected in y_hat")
            if torch.isnan(loss):
                print(" NaN detected in loss!")
                print("delays_loss:", delays_loss)
                print("y_hat:", y_hat)
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            train_loss += loss.detach().item() * bs

            pbar.update(pbar_update)

    train_loss = train_loss / train_len
    mae = mae / train_len
    gcc_mae = gcc_mae / train_len
    acc = acc / train_len
    gcc_acc = gcc_acc / train_len

    outstr = 'Train epoch %d, loss: %.6f, MAE: %.6f, GCC-MAE: %.6f, ACC: %.6f, GCC-ACC: %.6f' % (e,
                                                                                                 train_loss,
                                                                                                 mae,
                                                                                                 gcc_mae,
                                                                                                 acc,
                                                                                                 gcc_acc)

    log_string(outstr+'\n')

    scheduler.step()

    torch.cuda.empty_cache()

    # Validation
    model.eval()
    mae = 0.
    gcc_mae = 0.
    acc = 0.
    gcc_acc = 0.
    val_loss = 0.
    with tqdm(total=len(val_set)) as pbar:
        for batch_idx, (x1, x2, delays) in enumerate(val_loader):
            with torch.no_grad():
                bs = x1.shape[0]
                x1 = x1.to(device)
                x2 = x2.to(device)

                cc = tgcc(x1, x2)
                shift_gcc = torch.argmax(cc, dim=-1) - max_tau_gcc
                delays = delays.to(device)

                if args.input == 'freq':
                    X1, X2 = cdr_robust(x1.squeeze(1), x2.squeeze(1))
                    y_hat = model(X1, X2)
                    x1 = X1
                    x2 = X2
                else:
                    y_hat = model(x1, x2)

                if cfg.loss == 'ce':
                    delays_loss = torch.round(delays).type(torch.LongTensor)
                    shift = torch.argmax(y_hat, dim=-1) - max_tau
                else:
                    delays_loss = (delays - delay_mu) / delay_sigma
                    shift = y_hat * delay_sigma + delay_mu - max_tau

                gt = delays - max_tau
                mae += torch.sum(torch.abs(shift-gt))
                gcc_mae += torch.sum(torch.abs(shift_gcc-gt))

                acc += torch.sum(torch.abs(shift-gt) < cfg.t)
                gcc_acc += torch.sum(torch.abs(shift_gcc-gt) < cfg.t)

                loss = loss_fn(y_hat, delays_loss.to(device))
                val_loss += loss.detach().item() * bs

                torch.save(model.state_dict(), 'experiments/'
                        + args.exp_name+'/'+'model.pth')

                pbar.update(pbar_update)

    mae = mae / val_len
    gcc_mae = gcc_mae / val_len
    acc = acc / val_len
    gcc_acc = gcc_acc / val_len
    val_loss = val_loss / val_len

    outstr = 'Val epoch %d, loss: %.6f, MAE: %.6f, GCC MAE: %.6f, ACC: %.6f, GCC ACC: %.6f' % (e,
                                                                                               val_loss,
                                                                                               mae,
                                                                                               gcc_mae,
                                                                                               acc,
                                                                                               gcc_acc)
    log_string(outstr+'\n')

    torch.cuda.empty_cache()


# Save the model
if not args.evaluate:
    torch.save(model.state_dict(), 'experiments/'
               + args.exp_name+'/'+'model.pth')
    LOG_FOUT.close()


#테스트 루프
if args.evaluate:
    # load pre-trained model andevaluate on each window in the test set, for
    # each SNR and t60 in the list

    model.load_state_dict(torch.load(
        "experiments/"+args.exp_name+"/model.pth", map_location=torch.device(device)))

    model.eval()

    LOG_DIR = os.path.join('experiments/'+args.exp_name+'/')
    if cfg.anechoic:
        name = 'eval_anechoic.txt'
    else:
        name = 'eval.txt'
    LOG_FOUT = open(os.path.join(LOG_DIR, name), 'w')
    LOG_FOUT.write(str(args)+'\n')

    if cfg.anechoic:
        t60_range = [0.0]
    else:
        t60_range = cfg.t60_range

    ground_truth = np.empty(
        (test_len * NUM_TEST_WINS, len(cfg.snr_range), len(t60_range)))
    preds = np.empty(
        (test_len * NUM_TEST_WINS, len(cfg.snr_range), len(t60_range)))
    preds_gcc = np.empty(
        (test_len * NUM_TEST_WINS, len(cfg.snr_range), len(t60_range)))

    for snr_index, this_snr in enumerate(cfg.snr_range):
        for t60_index, this_t60 in enumerate(t60_range):

            mse = 0.
            gcc_mse = 0.
            mae = 0.
            gcc_mae = 0.
            acc = 0
            gcc_acc = 0
            test_loss = 0.

            start_index = 0
            end_index = 0

            pbar_update = batch_size
            with tqdm(total=len(test_set)*NUM_TEST_WINS) as pbar:

                for win in range(NUM_TEST_WINS):
                    delay_simulator_test = DelaySimulator(cfg.room_dim_test, fs, sig_len, [this_t60, this_t60],
                                                          cfg.mic_locs_test, max_tau, cfg.anechoic, False, [this_snr, this_snr], lower_bound=lower_bound+win*sig_len)

                    test_loader = torch.utils.data.DataLoader(
                        test_set,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=delay_simulator_test,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                    )

                    for batch_idx, (x1, x2, delays) in enumerate(test_loader):
                        with torch.no_grad():
                            bs = x1.shape[0]
                            x1 = x1.to(device)
                            x2 = x2.to(device)

                            cc = tgcc(x1, x2)
                            shift_gcc = torch.argmax(cc, dim=-1) - max_tau_gcc

                            delays = delays.to(device)
                            if args.input == 'freq':
                                X1, X2 = cdr_robust(x1.squeeze(1), x2.squeeze(1))
                                y_hat = model(X1, X2)
                                x1 = X1
                                x2 = X2
                            else:
                                y_hat = model(x1, x2)
                            
                            if cfg.loss == 'ce':
                                delays_loss = torch.round(
                                    delays).type(torch.LongTensor)
                                shift = torch.argmax(y_hat, dim=-1) - max_tau
                            else:
                                delays_loss = (delays - delay_mu) / delay_sigma
                                shift = y_hat * delay_sigma + delay_mu - max_tau

                            gt = delays - max_tau
                            mse += torch.sum(torch.abs(shift-gt)**2)
                            gcc_mse += torch.sum(torch.abs(shift_gcc-gt)**2)
                            mae += torch.sum(torch.abs(shift-gt))
                            gcc_mae += torch.sum(torch.abs(shift_gcc-gt))
                            acc += torch.sum(torch.abs(shift-gt) < cfg.t)
                            gcc_acc += torch.sum(torch.abs(shift_gcc-gt)
                                                 < cfg.t)

                            end_index = end_index + bs
                            ground_truth[start_index:end_index,
                                         snr_index, t60_index] = gt.cpu().numpy()
                            preds[start_index:end_index, snr_index,
                                  t60_index] = shift.cpu().numpy()
                            preds_gcc[start_index:end_index, snr_index,
                                      t60_index] = shift_gcc.view(-1).cpu().numpy()
                            start_index = start_index + bs

                            loss = loss_fn(y_hat, delays_loss.to(device))
                            test_loss += loss.item() * bs

                            pbar.update(pbar_update)

            rmse = torch.sqrt(mse / (test_len * NUM_TEST_WINS))
            gcc_rmse = torch.sqrt(gcc_mse / (test_len * NUM_TEST_WINS))
            mae = mae / (test_len * NUM_TEST_WINS)
            gcc_mae = gcc_mae / (test_len * NUM_TEST_WINS)
            acc = acc / (test_len * NUM_TEST_WINS)
            gcc_acc = gcc_acc / (test_len * NUM_TEST_WINS)
            test_loss = test_loss / (test_len * NUM_TEST_WINS)

            outstr = 'SNR: % d, T60: % .6f, loss: % .6f, RMSE: % .6f, GCC RMSE: % .6f, MAE: % .6f, GCC MAE: % .6f, ACC: % .6f, GCC ACC: % .6f' % (this_snr,
                                                                                                                                                  this_t60,
                                                                                                                                                  test_loss,
                                                                                                                                                  rmse,
                                                                                                                                                  gcc_rmse,
                                                                                                                                                  mae,
                                                                                                                                                  gcc_mae,
                                                                                                                                                  acc,
                                                                                                                                                  gcc_acc)
            log_string(outstr+'\n')

        torch.cuda.empty_cache()

    # Store all the ground truth delays and predictions
    if cfg.anechoic:
        np.savez('experiments/'+args.exp_name+'/'
                 + 'evaluations_anechoic.npz', ground_truth, preds, preds_gcc)
    else:
        np.savez('experiments/'+args.exp_name+'/'+'evaluations.npz',
                 ground_truth, preds, preds_gcc)

    LOG_FOUT.close()
    LOG_FOUT.close()