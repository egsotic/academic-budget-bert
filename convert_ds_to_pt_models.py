import glob
import os
import time
from timeit import default_timer as get_now
from run_pretraining import parse_arguments, prepare_model_and_optimizer, prepare_resuming_checkpoint


def main():
    start_time = time.time()
    args = parse_arguments()
    args.exp_start_marker = get_now()
    model, optimizer, lr_scheduler = prepare_model_and_optimizer(args)


    for path in glob.glob(args.load_training_checkpoint):
        print('path:', path)
        
        load_training_checkpoint = os.path.dirname(path)
        ckpt_id = os.path.basename(path)
        if not os.path.exists(os.path.join(args.output_dir, ckpt_id)):
            args.load_training_checkpoint = load_training_checkpoint
            args.load_checkpoint_id = ckpt_id
            _ = prepare_resuming_checkpoint(args, model)
            
            model.save_weights(ckpt_id, args.output_dir, is_deepspeed=True)


if __name__ == "__main__":
    main()
