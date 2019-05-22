import argparse
import logging
import time
import numpy as np

from core.optimizers import *


from myTorch.utils import MyContainer, create_config
from core.utils import get_optimizer
from myTorch import Logger
from myTorch import Experiment
from MLP import MLP
from myTorch.task.mnist import MNISTData

parser = argparse.ArgumentParser(description="MNIST Classification Task")
parser.add_argument("--config", type=str, default="config/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--config_params', type=str, default="default", help="config params to change")



def compute_accuracy(model, data_iterator, data_tag):
    """Computes accuracy for the given data split.

    Args:
        model: model object.
        data_iterator: data iterator object.
        data_tag: str, could be "train" or "valid" or "test"
        device: torch device

    returns:
        accuracy: float, average accuracy.
    """

    accuracy = 0.0
    total = 0.0

    while True:
        data = data_iterator.next(data_tag)
        if data is None:
            break
        x = data['x'].astype("float32")
        y = data['y'].astype("float32")
        output = model.forward(x)
        pred = np.argmax(output, 1)
        target = np.argmax(y, 1)
        accuracy += (pred==target).sum()
        total += len(pred)

    accuracy /= total
    return accuracy

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

    for i in range(tr.epochs_done, config.num_epochs):

        start_time = time.time()
        data_iterator.reset_iterator()
        avg_loss = 0
        while True:
            data = data_iterator.next("train")

            if data is None:
                break

            x = data['x'].astype("float32")
            y = data['y'].astype("float32")

            model.reset()
            model.optimizer.zero_grad()
            output = model.forward(x)
            loss = model.compute_loss(output, y)
            avg_loss += loss
            model.backward()
            model.optimizer.update()

            tr.updates_done += 1
            layer1_grad = np.linalg.norm(model._layer_list[0]._updates['W'])
            layer1_grad_rel = layer1_grad/loss
            tr.layer1_grad.append(layer1_grad)
            tr.layer1_grad_rel.append(layer1_grad_rel)
            logger.log_scalar("layer1_grad", layer1_grad, tr.updates_done)
            logger.log_scalar("layer1_grad_rel", layer1_grad_rel, tr.updates_done)

        avg_loss /= data_iterator._state.batches["train"]
        tr.train_loss.append(avg_loss)
        logger.log_scalar("training loss per epoch", avg_loss, i + 1)
        logging.info("training loss in epoch {}: {}".format(i + 1, avg_loss))

        val_acc = compute_accuracy(model, data_iterator, "valid")
        test_acc = compute_accuracy(model, data_iterator, "test")
        tr.valid_acc.append(val_acc)
        tr.test_acc.append(test_acc)
        logger.log_scalar("valid acc per epoch", val_acc, i + 1)
        logger.log_scalar("test acc per epoch", test_acc, i + 1)
        logging.info("valid acc in epoch {}: {}".format(i + 1, val_acc))
        logging.info("test acc in epoch {}: {}".format(i + 1, test_acc))

        tr.epochs_done += 1

        experiment.save()
        logging.info("iteration took {} seconds.".format(time.time()-start_time))

        


def create_experiment(config):
    """Creates the experiment.

    Args:
        config: config dictionary.

    returns:
        experiment, model, data_iterator, training_statitics, logger, device
    """

    experiment = Experiment(config.name, config.save_dir)

    logger = None
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)

    np.random.seed(config.rseed)


    hidden_layer_size = []
    for i in range(config.num_hidden_layers):
        hidden_layer_size.append(config.hidden_layer_size)
    model = MLP(num_hidden_layers=config.num_hidden_layers, hidden_layer_size=hidden_layer_size,
        activation=config.activation, input_dim=config.input_dim, output_dim=config.output_dim)
    logging.info("Model has {} parameters.".format(model.num_parameters()))

    data_iterator = MNISTData(batch_size=config.batch_size, inference_batch_size=config.inference_batch_size,
        seed=config.data_iterator_seed, use_one_hot=config.use_one_hot)

    
    optimizer = get_optimizer(config)
    optimizer.register_model(model)
    model.register_optimizer(optimizer)

    training_statistics = MyContainer()
    training_statistics.train_loss = []
    training_statistics.valid_acc = []
    training_statistics.test_acc = []
    training_statistics.layer1_grad = []
    training_statistics.layer1_grad_rel = []
    training_statistics.updates_done = 0
    training_statistics.epochs_done = 0


    experiment.register_experiment(model, config, logger, training_statistics, data_iterator)

    return experiment, model, data_iterator, training_statistics, logger


def run_experiment(args):
    """Runs the experiment.

    Args:
        args: command line arguments.
    """

    config = create_config(args.config, args.config_params)

    logging.info(config.get())

    experiment, model, data_iterator, training_statistics, logger = create_experiment(config)

    if not args.force_restart:
        if experiment.is_resumable():
            experiment.resume()
    else:
        experiment.force_restart()

    train(experiment, model, config, data_iterator, training_statistics, logger)
    logging.info("Training done!")



if __name__ == '__main__':
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO)
  run_experiment(args)




