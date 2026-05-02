import os
import os.path
import sys
import logging
import copy
import time
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    device = device.split(',')

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

def _train(args):

    if args["ca"]:
        prefix_head = "ca"
    else:
        prefix_head = "standard_head"

    prefix_lora = args["prefix"]
    if prefix_lora != "":
        prefix_lora += "_"

    if args["use_flat"]:
        prefix_lora += ("rho=" + str(args["rho"]) + "_")

    if not args["use_slora"] and not args["use_plora"]:
        prefix_lora += 'seq_lora'
    else:
        if args["use_slora"] and not args["use_plora"]:
            prefix_lora += "slora"
        elif not args["use_slora"] and args["use_plora"]:
            prefix_lora += "plora"
        else:
            prefix_lora += "slora+plora"

    prefix_lora +=\
        "_mg=" + str(args["merge_gamma"]) +\
        "_sg=" + str(args["slora_gamma"]) +\
        "_pg=" + str(args["plora_gamma"])

    logdir = 'logs/{}/{}_tasks/{}/{}'.format(args['dataset'], args['total_sessions'], prefix_head, prefix_lora)
    args['logdir'] = logdir
    print(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir, '{}_slora:{}_plora:{}_rank:{}_{}_{}_{}-{}'.format(args['seed'], args["use_slora"], args["use_plora"], args['rank'], args["lora_type"], args['model_name'], args['optim'], args['lrate']))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    print(logfilename)
    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args['dataset'], 
        args['shuffle'], 
        args['seed'], 
        args['init_cls'], 
        args['increment'], 
        args
    )
    model = factory.get_model(args['model_name'], args)


    cnn_curve, cnn_curve_with_task, nme_curve, cnn_curve_task = {'top1': []}, {'top1': []}, {'top1': []}, {'top1': []}
    for task_id in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        time_start = time.time()
        model.incremental_train(data_manager)
        time_end = time.time()
        logging.info('Time:{}'.format(time_end - time_start))
        time_start = time.time()
        cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task = model.eval_task()
        time_end = time.time()
        logging.info('Time:{}'.format(time_end - time_start))
        # raise Exception
        model.after_task()

        logging.info('CNN: {}'.format(cnn_accy['grouped']))
        cnn_curve['top1'].append(cnn_accy['top1'])
        cnn_curve_with_task['top1'].append(cnn_accy_with_task['top1'])
        cnn_curve_task['top1'].append(cnn_accy_task)
        logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
        logging.info('CNN top1 with task curve: {}'.format(cnn_curve_with_task['top1']))
        logging.info('CNN top1 task curve: {}'.format(cnn_curve_task['top1']))

        if task_id > 0:
            diagonal = np.diag(model.acc_matrix)
            forgetting = np.mean((np.max(model.acc_matrix, axis=1) -
                                model.acc_matrix[:, task_id])[:task_id])
            backward = np.mean((model.acc_matrix[:, task_id] - diagonal)[:task_id])

            result_str = "Forgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
            logging.info(result_str)

    logging.info('Accuracy Matrix: \n {}'.format(model.acc_matrix.T.round(2)))
    logging.info('Average Accuracy: {}'.format(np.mean(cnn_curve['top1'])))
    logging.info('Last Accuracy: {}'.format(cnn_curve['top1'][-1]))

def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random(args):
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))

