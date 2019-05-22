#!/usr/bin/env python
import numpy
import argparse
import logging
import ipdb
import time
from rnnGrad.core.optimizers import *
from rnnGrad.core.utils import get_optimizer
import torch
from myTorch import Experiment
from rnn import Recurrent
from myTorch.task.copy_task import CopyData
from myTorch.task.repeat_copy_task import RepeatCopyData
from myTorch.task.associative_recall_task import AssociativeRecallData
from myTorch.task.copying_memory import CopyingMemoryData
from myTorch.task.adding_task import AddingData
from myTorch.task.denoising import DenoisingData
from myTorch.utils.logger import Logger
from myTorch.utils import MyContainer, create_config, one_hot

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str, default="config/default_rnn.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")


def get_data_iterator(config):

    if config.task == "copy":
        data_iterator = CopyData(num_bits=config.num_bits, min_len=config.min_len,
                                 max_len=config.max_len, batch_size=config.batch_size)
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


def train(experiment, model, config, data_iterator, tr, logger):
    """Training loop.

    Args:
        experiment: experiment object.
        model: model object.
        config: config dictionary.
        data_iterator: data iterator object
        tr: training statistics dictionary.
        logger: logger object.
    """

    init_time = time.time()
    for step in range(tr.updates_done, config.max_steps):

        if config.inter_saving is not False:
            if tr.updates_done in config.inter_saving:
                experiment.save(str(tr.updates_done))

        data = data_iterator.next()
        data['mask'] = torch.from_numpy(data['mask'])
        seqloss = 0

        model.reset_hidden(batch_size=config.batch_size)
        model.reset()

        numd = config.num_digits + config.num_noise_digits


        for i in range(0, data["datalen"]):
            x = torch.from_numpy(data['x'][i])
            y = torch.from_numpy(one_hot(data['y'][i], numd)).float()
            mask = data["mask"][i]



            output = model.forward(x, i+1)

            if mask == 1:
                #if i == (data["datalen"]-1):
                model.optimizer.zero_grad()
                loss = model.compute_loss(output, y, i+1)
                seqloss += loss
                model.backward(i+1)
                model.optimizer.update(i+1)

        W_grad = np.linalg.norm(model._layer_list[0]._updates['W'].numpy())
        W_grad_rel = W_grad/loss
        logger.log_scalar("W_grad", W_grad, tr.updates_done)
        logger.log_scalar("W_grad_rel", W_grad_rel, tr.updates_done)

        U_grad = np.linalg.norm(model._layer_list[0]._updates['U'].numpy())
        U_grad_rel = U_grad/loss
        logger.log_scalar("U_grad", U_grad, tr.updates_done)
        logger.log_scalar("U_grad_rel", U_grad_rel, tr.updates_done)

        seqloss /= sum(data["mask"])
        tr.average_bce.append(seqloss)
        running_average = sum(tr.average_bce) / len(tr.average_bce)

        if config.use_tflogger:
            logger.log_scalar("running_avg_loss", running_average, step + 1)
            logger.log_scalar("loss", tr.average_bce[-1], step + 1)

        tr.updates_done += 1
        if tr.updates_done % 1 == 0:
            logging.info("examples seen: {}, inst loss: {}".format(tr.updates_done*config.batch_size,
                                                                                tr.average_bce[-1]))
        if tr.updates_done % config.save_every_n == 0:
            experiment.save()
    print(time.time()-init_time)


def create_experiment(config):
    """Creates an experiment based on config."""

    experiment = Experiment(config.name, config.save_dir)

    logger = None
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)

    np.random.seed(config.rseed)
    torch.manual_seed(config.rseed)
    input_size = config.num_digits + config.num_noise_digits + 1
    output_size = input_size - 1

    model = Recurrent(input_size, output_size, config.hidden_size)

    data_iterator = get_data_iterator(config)

    optimizer = get_optimizer(config)
    optimizer.register_model(model)
    model.register_optimizer(optimizer)

    tr = MyContainer()
    tr.updates_done = 0
    tr.average_bce = []

    experiment.register_experiment(model=model, config=config, logger=logger, train_statistics=tr,
        data_iterator=data_iterator)

    return experiment, model, data_iterator, tr, logger


def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)

    logging.info(config.get())
    experiment, model, data_iterator, tr, logger = create_experiment(config)

    if not args.force_restart:
        if experiment.is_resumable():
            experiment.resume()
    else:
        experiment.force_restart()

    train(experiment, model, config, data_iterator, tr, logger)


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    run_experiment(args)