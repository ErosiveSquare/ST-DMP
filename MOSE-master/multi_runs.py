import datetime
import os
import random

import numpy as np
import torch
from torch.optim import Adam

from agent import get_agent
from experiment.dataset import get_data
from models import get_model
from models.buffer import Buffer
from models.cka_utils import (
    build_loader_from_tensor_file,
    run_layerwise_cka_from_checkpoints,
    save_task_dataset,
)
from utils.util import Logger, compute_performance


def multiple_run(args):
    test_all_acc = torch.zeros(args.run_nums)
    last_test_all_acc = torch.zeros(args.run_nums)

    accuracy_list = []
    last_accuracy_list = []
    base_seed = getattr(args, "seed", 0)
    for run in range(args.run_nums):
        run_seed = base_seed + run
        args.seed = run_seed
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        tmp_acc = []
        last_tmp_acc = []

        buffer_tmp_acc = []
        buffer_last_tmp_acc = []

        train_tmp_acc = []
        train_last_tmp_acc = []

        test_interval = max(1, getattr(args, 'test_interval', 1))

        print('=' * 100)
        print(f"-----------------------------run {run} start--------------------------")
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('=' * 100)
        data, class_num, class_per_task, task_loader, input_size = get_data(
            dataset_name=args.dataset, batch_size=args.batch_size, n_workers=args.n_workers, n_tasks=args.n_tasks
        )
        args.n_classes = class_num

        setattr(args, 'run_name', f"{args.exp_name} run_{run:02d}")
        print(f"\nRun {run}: {args.run_name} {'*' * 50}\n")
        logger = Logger(args, base_dir=f"./outputs/{args.method}/{args.dataset}")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.join(base_dir, "checkpoints", f"run_{run:02d}")
        data_dir = os.path.join(base_dir, "cka_data", f"run_{run:02d}")
        plot_dir = os.path.join(base_dir, "plot")
        os.makedirs(ckpt_dir, exist_ok=True)
        save_cka_data = getattr(args, "save_cka_data", False)
        if save_cka_data:
            os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        buffer = Buffer(args, input_size).cuda()
        model = get_model(method_name=args.method, nclasses=class_num).cuda()
        optimizer = Adam(model.parameters(), args.lr, weight_decay=args.wd)
        agent = get_agent(
            method_name=args.method, model=model, 
            buffer=buffer, optimizer=optimizer, input_size=input_size, args=args
        )

        print(f"number of classifier parameters:\t {model.n_params/1e6:.2f}M", )
        print(f"buffer parameters (image size prod):\t {np.prod(buffer.bx.size())/1e6:.2f}M", )

        latest_acc_list = None
        latest_all_acc_list = None
        for i in range(len(task_loader)):
            print(f"\n-----------------------------run {run} task id:{i} start training-----------------------------")

            train_log_holder = agent.train(i, task_loader[i]['train'])

            logger.log_losses(train_log_holder)

            if getattr(args, "save_runner_ckpt", False):
                ckpt_task_path = os.path.join(ckpt_dir, f"ckpt_task_{i}.pth")
                agent.save_checkpoint(ckpt_task_path)

            if save_cka_data:
                task_eval_loader = task_loader[i].get('test', task_loader[i]['train'])
                data_path = os.path.join(data_dir, f"task_{i}.pt")
                save_task_dataset(
                    task_eval_loader,
                    data_path,
                    max_batches=getattr(args, 'cka_save_max_batches', None),
                )

            if (i + 1) % test_interval == 0 or i == len(task_loader) - 1:
                acc_list, all_acc_list = agent.test(i, task_loader)

                latest_acc_list = acc_list
                latest_all_acc_list = all_acc_list

                tmp_acc.append(acc_list)
                last_tmp_acc.append(all_acc_list[args.expert])

                logger.log_accs(all_acc_list)

                for feat_id, acc_list_id in all_acc_list.items():
                    if feat_id == 'step':
                        continue
                    test_accuracy_id = acc_list_id[:i+1].mean()
                    logger.log_scalars({
                        f"test/{feat_id}_avg_acc":       test_accuracy_id,
                    }, step=agent.total_step)
        
        if getattr(args, "save_runner_ckpt", False):
            ckpt_final_path = os.path.join(ckpt_dir, "ckpt_final.pth")
            agent.save_checkpoint(ckpt_final_path)

        acc_list = latest_acc_list
        all_acc_list = latest_all_acc_list

        test_accuracy = acc_list.mean()
        test_all_acc[run] = test_accuracy
        
        tmp_acc = np.array(tmp_acc)
        avg_fgt = (tmp_acc.max(0) - tmp_acc[-1, :]).mean()
        diag_len = min(tmp_acc.shape[0], tmp_acc.shape[1]) if tmp_acc.ndim == 2 else 0
        if diag_len > 0:
            avg_bwt = (tmp_acc[-1, :diag_len] - np.diagonal(tmp_acc)[:diag_len]).mean()
        else:
            avg_bwt = 0.0
        accuracy_list.append(tmp_acc)

        logger.log_scalars({
            'test/final_avg_acc':       test_accuracy,
            'test/final_avg_fgt':       avg_fgt,
            'test/final_avg_bwt':       avg_bwt,
            'metrics/buffer_n_bits':        agent.buffer.n_bits / 1e6,
            'metrics/model_n_params':       agent.model.n_params / 1e6
        }, step=agent.total_step+1, verbose=True)

        logger.log_accs_table(
            name='task_accs_table', accs_list=tmp_acc,
            step=agent.total_step+1, verbose=True
        )

        # record the last scalars
        last_acc_list = all_acc_list[args.expert]
        last_test_accuracy = last_acc_list.mean()
        last_test_all_acc[run] = last_test_accuracy
        last_tmp_acc = np.array(last_tmp_acc)
        last_avg_fgt = (last_tmp_acc.max(0) - last_tmp_acc[-1, :]).mean()
        last_diag_len = min(last_tmp_acc.shape[0], last_tmp_acc.shape[1]) if last_tmp_acc.ndim == 2 else 0
        if last_diag_len > 0:
            last_avg_bwt = (last_tmp_acc[-1, :last_diag_len] - np.diagonal(last_tmp_acc)[:last_diag_len]).mean()
        else:
            last_avg_bwt = 0.0
        last_accuracy_list.append(last_tmp_acc)

        logger.log_scalars({
            'test/last_final_avg_acc':       last_test_accuracy,
            'test/last_final_avg_fgt':       last_avg_fgt,
            'test/last_final_avg_bwt':       last_avg_bwt,
        }, step=agent.total_step+1, verbose=True)

        logger.log_accs_table(
            name='last_task_accs_table', accs_list=last_tmp_acc,
            step=agent.total_step+1, verbose=True
        )

        if getattr(args, 'cka_eval', False):
            cka_task_id = int(getattr(args, 'cka_task_id', 0))
            ckpt_task_path = os.path.join(ckpt_dir, f"ckpt_task_{cka_task_id}.pth")
            data_path = os.path.join(data_dir, f"task_{cka_task_id}.pt")
            if cka_task_id >= len(task_loader):
                print("[CKA] skip: task id out of range", cka_task_id)
            elif not (os.path.isfile(ckpt_task_path) and os.path.isfile(ckpt_final_path)):
                print("[CKA] skip: missing checkpoints", ckpt_task_path, ckpt_final_path)
            elif not os.path.isfile(data_path):
                print("[CKA] skip: missing task data", data_path)
            else:
                cka_loader = build_loader_from_tensor_file(
                    data_path,
                    batch_size=args.batch_size,
                    num_workers=args.n_workers,
                    shuffle=False,
                )
                device = torch.device('cuda') if args.cuda else torch.device('cpu')
                run_layerwise_cka_from_checkpoints(
                    model_factory=lambda: get_model(method_name=args.method, nclasses=class_num),
                    ckpt_task_path=ckpt_task_path,
                    ckpt_final_path=ckpt_final_path,
                    loader=cka_loader,
                    device=device,
                    max_batches=int(getattr(args, 'cka_max_batches', 10)),
                    out_dir=str(getattr(args, 'cka_out_dir', plot_dir)),
                    run_name=str(getattr(args, 'run_name', f"run_{run:02d}")),
                    baseline_init_path=getattr(args, 'cka_baseline_init', None),
                    baseline_final_path=getattr(args, 'cka_baseline_final', None),
                )


        # buffer_tmp_acc = np.array(buffer_tmp_acc)
        # buffer_last_tmp_acc = np.array(buffer_last_tmp_acc)

        # train_tmp_acc = np.array(train_tmp_acc)
        # train_last_tmp_acc = np.array(train_last_tmp_acc)

        # logger.log_accs_table(
        #     name='buffer_task_accs_table', accs_list=buffer_tmp_acc,
        #     step=agent.total_step+1, verbose=True
        # )
        # logger.log_accs_table(
        #     name='buffer_last_task_accs_table', accs_list=buffer_last_tmp_acc,
        #     step=agent.total_step+1, verbose=True
        # )

        # logger.log_accs_table(
        #     name='train_task_accs_table', accs_list=train_tmp_acc,
        #     step=agent.total_step+1, verbose=True
        # )
        # logger.log_accs_table(
        #     name='train_last_task_accs_table', accs_list=train_last_tmp_acc,
        #     step=agent.total_step+1, verbose=True
        # )

        print('=' * 100)
        print("{}th run's Test result: Accuracy: {:.2f}%".format(run, test_accuracy))
        print('=' * 100)

        logger.close()

    last_accuracy_array = np.array(last_accuracy_list)
    last_avg_end_acc, last_avg_end_fgt, last_avg_acc, last_avg_bwtp, last_avg_fwt = compute_performance(last_accuracy_array)
    print(f"\n{'=' * 100}")
    print(f"total {args.run_nums}runs last test acc results: {last_test_all_acc}")
    print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
          .format(last_avg_end_acc, last_avg_end_fgt, last_avg_acc, last_avg_bwtp, last_avg_fwt))
    print('=' * 100)

    accuracy_array = np.array(accuracy_list)
    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_array)
    print(f"\n{'=' * 100}")
    print(f"total {args.run_nums}runs test acc results: {test_all_acc}")
    print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
          .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
    print('=' * 100)
