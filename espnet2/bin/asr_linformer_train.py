#!/usr/bin/env python3
from espnet2.tasks.asr_linformer import ASRLINFORMERTask


def get_parser():
    parser = ASRLINFORMERTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    ASRLINFORMERTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
