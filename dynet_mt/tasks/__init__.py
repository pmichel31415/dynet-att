#!/usr/bin/env python3
from ..command_line import get_base_parser, parse_args_and_yaml

from .train_task import TrainTask
from .translate_task import TranslateTask
from .eval_ppl_task import EvalPPLTask
from .eval_bleu_task import EvalBLEUTask

task_types = {
    "train": TrainTask,
    "eval_ppl": EvalPPLTask,
    "eval_bleu": EvalBLEUTask,
    "translate": TranslateTask,
}
supported_tasks = list(task_types.keys())


def get_task():
    parser, task_subparsers = get_base_parser()
    task_parsers = {}
    for name, task in task_types.items():
        task_parser = task_subparsers.add_parser(name, help=task.desc)
        task.add_args(task_parser)
        task_parsers[name] = task_parser
    base_args = parse_args_and_yaml(parser, known_args_only=False)
    # This is a hack for config files
    task_name = base_args.task
    task_type = task_types[task_name]
    task_args = task_type.parse_args(parser, task_parsers[task_name])
    return task_type(task_args)


def add_tasks_args(task_subparsers):
    for name, task in task_types.items():
        task_parser = task_subparsers.add_parser(name, help=task.desc)
        task.add_args(task_parser)


def task_from_args(parser, args):
    task_type = task_types[args.task]
    task = task_type(task_type.parse_args(parser))
    return task
