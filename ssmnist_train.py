import numpy
import argparse
import logging

import torch

from core.optimizers import *
from core.utils import get_optimizer

from myTorch import Experiment
from rnn import Recurrent
from myTorch.task.ssmnist_task import SSMNISTData
from myTorch.task.mnist_task import PMNISTData
from myTorch.utils.logger import Logger
from myTorch.utils import MyContainer, create_config

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str, default="config/default_rnn.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")


def get_data_iterator(config):
    if config.task == "ssmnist":
        data_iterator = SSMNISTData(config.data_folder, num_digits=config.num_digits,
                                    batch_size=config.batch_size, seed=config.seed)
    elif config.task == "pmnist":
        data_iterator = PMNISTData(batch_size=config.batch_size, seed=config.seed)

    return data_iterator


def evaluate(experiment, model, config, data_iterator, tr, logger, device, tag):

    logging.info("Doing {} evaluation".format(tag))

    correct = 0.0
    num_examples = 0.0

    while True:

        data = data_iterator.next(tag)

        if data is None:
            break

        model.reset()
        model.reset_hidden(batch_size=config.batch_size)

        for i in range(0, data["datalen"]):

            x = torch.from_numpy(numpy.asarray(data['x'][i])).to(device)
            y = torch.from_numpy(numpy.asarray(data['y'][i])).to(device)
            mask = torch.from_numpy(numpy.asarray(data['mask'][i])).to(device)

            output = model.forward(x, i+1)
            if mask[0] == 1:


                values, indices = torch.max(output, 1)
                correct += (indices == y).to(device, dtype=torch.float32).sum().item()
                num_examples += data['x'].shape[1]


    final_accuracy = correct/ num_examples
    logging.info(" epoch {}, {} accuracy: {}".format(tr.epochs_done, tag, final_accuracy))

    if config.use_tflogger:
        logger.log_scalar("{}_accuracy".format(tag), final_accuracy, tr.epochs_done)


def train(experiment, model, config, data_iterator, tr, logger, device):
    """Training loop.

    Args:
        experiment: experiment object.
        model: model object.
        config: config dictionary.
        data_iterator: data iterator object
        tr: training statistics dictionary.
        logger: logger object.
    """

    for step in range(tr.updates_done, config.max_steps):



        data = data_iterator.next("train")
        if data is None:

            tr.epochs_done += 1
            evaluate(experiment, model, config, data_iterator, tr, logger, device, "valid")
            evaluate(experiment, model, config, data_iterator, tr, logger, device, "test")
            data_iterator.reset_iterator()
            data = data_iterator.next("train")

        seqloss = 0

        model.reset_hidden(batch_size=config.batch_size)
        model.reset()

        for i in range(0, data["datalen"]):

            x = torch.from_numpy(numpy.asarray(data['x'][i])).to(device)
            y = torch.from_numpy(numpy.asarray(data['y'][i])).to(device)
            mask = torch.from_numpy(numpy.asarray(data['mask'][i])).to(device)

            model.optimizer.zero_grad()

            output = model.forward(x, i+1)

            if mask[0]==1:
                model.optimizer.zero_grad()
                loss = model.compute_loss(output, y, i+1)
                seqloss += loss
                model.backward(i+1)
                model.optimizer.update(i+1)

        seqloss /= float(data["mask"].shape[1])
        tr.ce["train"].append(seqloss.item())
        running_average = sum(tr.ce["train"]) / len(tr.ce["train"])

        if config.use_tflogger:
            logger.log_scalar("running_avg_loss", running_average, step + 1)
            logger.log_scalar("train loss", tr.ce["train"][-1], step + 1)

        tr.updates_done += 1

        if tr.updates_done % 1 == 0:
            logging.info("examples seen: {}, inst loss: {}".format(tr.updates_done * config.batch_size,
                                                                   tr.ce["train"][-1]))

        if tr.updates_done % config.save_every_n == 0:
            experiment.save()


def create_experiment(config):
    """Creates an experiment based on config."""

    device = torch.device(config.device)
    logging.info("using {}".format(config.device))

    experiment = Experiment(config.name, config.save_dir)
    logger = None
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)

    torch.manual_seed(config.rseed)

    model = Recurrent(config.input_size, config.output_size, config.hidden_size)
    if config.device != "cpu":
        model.cuda()

    data_iterator = get_data_iterator(config)

    optimizer = get_optimizer(config)
    optimizer.register_model(model)
    model.register_optimizer(optimizer)

    tr = MyContainer()
    tr.updates_done = 0
    tr.epochs_done = 0
    tr.ce = {}
    tr.ce["train"] = []
    tr.accuracy = {}
    tr.accuracy["valid"] = []
    tr.accuracy["test"] = []
    tr.grad_norm = []


    experiment.register_experiment(model=model, config=config, logger=logger, train_statistics=tr,
        data_iterator=data_iterator)

    return experiment, model, data_iterator, tr, logger, device


def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)

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