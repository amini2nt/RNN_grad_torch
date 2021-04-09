import argparse
import logging
import os
import time
import numpy as np

import torch

from rnnGrad.core.utils import get_optimizer
from rnnGrad.core.losses import *
from rnn import Recurrent

from rnnGrad.myTorch.task.copy_task import CopyData
from rnnGrad.myTorch.task.repeat_copy_task import RepeatCopyData
from rnnGrad.myTorch.task.associative_recall_task import AssociativeRecallData
from rnnGrad.myTorch.task.copying_memory import CopyingMemoryData
from rnnGrad.myTorch.task.adding_task import AddingData
from rnnGrad.myTorch.task.denoising import DenoisingData
from rnnGrad.myTorch.task.reverse_copying_memory import ReverseCopyingMemoryData

from rnnGrad.myTorch import Experiment
from rnnGrad.myTorch.utils.logger import Logger
from rnnGrad.myTorch.utils import MyContainer, create_config, one_hot

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str, default="config/default_rnn.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")
parser.add_argument('--config_params', type=str, default="default", help="config params to change")

os.environ['MYTORCH_SAVEDIR'] = '.config'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_data_iterator(config):
    if config.task == "copy":
        data_iterator = CopyData(num_bits=config.num_bits, min_len=config.min_len,
                                 max_len=config.max_len, batch_size=config.batch_size)
    elif config.task == "reverse_copying_memory":
        data_iterator = ReverseCopyingMemoryData(seq_len=config.seq_len, time_lag_min=config.time_lag_min,
                                                 time_lag_max=config.time_lag_max, num_digits=config.num_digits,
                                                 num_noise_digits=config.num_noise_digits,
                                                 batch_size=config.batch_size, seed=config.seed)
    elif config.task == "repeat_copy":
        data_iterator = RepeatCopyData(num_bits=config.num_bits, min_len=config.min_len,
                                       max_len=config.max_len, min_repeat=config.min_repeat,
                                       max_repeat=config.max_repeat, batch_size=config.batch_size)
    elif config.task == "associative_recall":
        data_iterator = AssociativeRecallData(num_bits=config.num_bits, min_len=config.min_len,
                                              max_len=config.max_len, block_len=config.block_len,
                                              batch_size=config.batch_size)
    elif config.task == "copying_memory":
        data_iterator = CopyingMemoryData(seq_len=config.seq_len, time_lag_min=config.time_lag_min,
                                          time_lag_max=config.time_lag_max, num_digits=config.num_digits,
                                          num_noise_digits=config.num_noise_digits,
                                          batch_size=config.batch_size, seed=config.seed)
    elif config.task == "adding":
        data_iterator = AddingData(seq_len=config.seq_len, batch_size=config.batch_size, seed=config.seed)
    elif config.task == "denoising_copy":
        data_iterator = DenoisingData(seq_len=config.seq_len, time_lag_min=config.time_lag_min,
                                      time_lag_max=config.time_lag_max, batch_size=config.batch_size,
                                      num_noise_digits=config.num_noise_digits,
                                      num_digits=config.num_digits, seed=config.seed)

    return data_iterator


def train(experiment, model, config, data_iterator, tr, logger, device):
    """Training loop.

    Args:
        experiment: experiment object.
        model: model object.
        config: config dictionary.
        data_iterator: data iterator object
        tr: training statistics dictionary.
        logger: logger object.
        device: torch device.
    """

    init_time = time.time()
    data = data_iterator.next()
    for step in range(tr.updates_done, config.max_steps):

        if config.inter_saving is not False:
            if tr.updates_done in config.inter_saving:
                experiment.save(str(tr.updates_done))

        #data = data_iterator.next()
        #data['mask'] = torch.from_numpy(data['mask']).to(device)
        seqloss = 0
        losses = []

        model.reset_hidden(batch_size=config.batch_size)
        model.reset()

        numd = config.num_digits + config.num_noise_digits

        for i in range(0, data["datalen"]):
            x = torch.from_numpy(data['x'][i]).to(device)
            y = torch.from_numpy(one_hot(data['y'][i], numd)).float().to(device)
            mask = data["mask"][i]

            output = model.forward(x, i + 1)

            if mask == 1:
                if not config.separate_optimizers:
                    model.optimizer.zero_grad()
                else:
                    for optimizer in model.optimizers:
                        optimizer.zero_grad()
                loss = model.compute_loss(output, y, i + 1)
                if config.use_time_logger:
                    losses.append(loss)
                seqloss += loss
                model.backward(i + 1)
                if not config.separate_optimizers:
                    model.optimizer.update(i + 1)
                    #print(model.optimizer._params["g2"])
                else:
                    model.optimizers[i].update(i + 1)

        for i in range(0, len(losses)):
            logger.log_scalar("Loss_at_time_" + str(i + 1), loss[i]/seqloss, tr.updates_done)

        if config.cell == "rnn":
            W_grad = np.linalg.norm(model._layer_list[0]._updates['W'].numpy())
            W_grad_rel = W_grad / loss
            logger.log_scalar("W_grad", W_grad, tr.updates_done)
            logger.log_scalar("W_grad_rel", W_grad_rel, tr.updates_done)

            U_grad = np.linalg.norm(model._layer_list[0]._updates['U'].numpy())
            U_grad_rel = U_grad / loss
            logger.log_scalar("U_grad", U_grad, tr.updates_done)
            logger.log_scalar("U_grad_rel", U_grad_rel, tr.updates_done)

        seqloss /= sum(data["mask"])
        tr.average_bce.append(seqloss)
        running_average = sum(tr.average_bce) / len(tr.average_bce)

        if config.use_logger:
            logger.log_scalar("running_avg_loss", running_average, step + 1)
            logger.log_scalar("loss", tr.average_bce[-1], step + 1)

        if config.use_time_logger:
            if config.cell == "rnn":
                rnn_layer =  model._layer_list[0]
                for t in rnn_layer._grads.keys():
                    W_grad = np.linalg.norm(rnn_layer._grads[t]['W'].numpy())
                    logger.log_scalar("W_grad_time_" + str(t), W_grad, tr.updates_done)

                    U_grad = np.linalg.norm(rnn_layer._grads[t]['U'].numpy())
                    logger.log_scalar("U_grad_time_" + str(t), U_grad, tr.updates_done)

        tr.updates_done += 1
        if tr.updates_done % 1 == 0:
            logging.info("examples seen: {}, inst loss: {}".format(tr.updates_done * config.batch_size,
                                                                   tr.average_bce[-1]))
        if tr.updates_done % config.save_every_n == 0:
            experiment.save()
    print(time.time() - init_time)


def create_experiment(config):
    """Creates an experiment based on config.

    Args:
        config: config dictionary.

    returns:
        experiment, model, data_iterator, training_statitics, logger, device
    """
    device = torch.device(config.device)
    logging.info("using {}".format(config.device))

    experiment = Experiment(config.name, config.save_dir)

    logger = None
    if config.use_logger:
        logger = Logger(config.log_dir)
        logger.log_config(config)

    np.random.seed(config.rseed)
    torch.manual_seed(config.rseed)

    input_size = config.num_digits + config.num_noise_digits + 1
    output_size = input_size - 1

    model = Recurrent(input_size, output_size, config.hidden_size, config.cell,
                      config.activation, config.chrono_init, config.t_max).to(device)
    model.add_loss(bce_with_logits_loss(average='mean'))

    data_iterator = get_data_iterator(config)

    if not config.separate_optimizers:
        optimizer = get_optimizer(config)
        optimizer.register_model(model)
        model.register_optimizer(optimizer)
    else:
        optimizers = [get_optimizer(config)] * data_iterator._datalen
        model.register_optimizers(optimizers)
        for i in range(0, len(optimizers)):
            optimizers[i].register_model(model)

    tr = MyContainer()
    tr.updates_done = 0
    tr.average_bce = []

    experiment.register_experiment(model=model, config=config, logger=logger, train_statistics=tr,
                                   data_iterator=data_iterator)

    return experiment, model, data_iterator, tr, logger, device


def run_experiment(args):
    """Runs the experiment.


    Args:
        args: command line arguments.
    """
    config = create_config(args.config, args.config_params)

    logging.info(config.get())

    experiment, model, data_iterator, tr, logger, device = create_experiment(config)

    if not args.force_restart:
        if experiment.is_resumable():
            experiment.resume()
    else:
        experiment.force_restart()

    train(experiment, model, config, data_iterator, tr, logger, device)


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_experiment(args)
