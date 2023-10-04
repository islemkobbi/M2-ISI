import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        default=False,
        help="generate the dataset",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="generate the dataset",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="generate the dataset",
    )
    parser.add_argument(
        "--q",
        action="store_true",
        default=False,
        help="quiet verbose",
    )
    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=32,
        help="training hyper-parameter: batch-size",
    )
    parser.add_argument(
        "--steps",
        nargs="?",
        type=int,
        default=20000,
        help="training hyper-parameter: number of optimization steps",
    )
    parser.add_argument(
        "--lr",
        nargs="?",
        type=float,
        default=1.0e-3,
        help="training hyper-parameter: initial learning rate",
    )

    args = parser.parse_args()

    from .Utils.clear import remove_chache_folders
    from .Utils.seeds import set_seeds

    set_seeds()

    if args.generate:
        from .GeneratData.generate import SetGenerator

        SetGenerator(
            coco_path=os.path.join(os.getcwd(), "train2017"),
            annotations_path=os.path.join(
                os.getcwd(), "annotations", "instances_train2017.json"
            ),
        ).setup()
    if args.train:
        from .Baseline.instance_segmentation import BaselineTrainer

        BaselineTrainer(quiet_mode=args.q).train(
            num_opt_steps=args.steps, batch_size=args.batch_size, lr=args.lr
        )

    if args.evaluate:
        from .Evaluation.evaluate_test import Evaluator

        Evaluator().evaluate(
            all_students_folder=os.path.join(os.getcwd(), "submissions")
        )

    remove_chache_folders(current_repo=os.getcwd())
