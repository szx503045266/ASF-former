"""
Generate file for TensorboardX visualization.
"""
import json
import argparse
from pathlib import Path
from tensorboardX import SummaryWriter
import time

def get_args():
    parser = argparse.ArgumentParser(description="Args")
    parser.add_argument("--lg", type=str, 
			default="output/log.txt")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    log_file = Path(args.lg)
    outpth = Path(str("./tensorboard/{}-{}/".format(log_file.parents[0], "D{}".format(time.strftime('%Y-%m-%d-%H',time.localtime(time.time()))))))
    outpth.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(outpth))
    print("Processing {}...for {}".format(log_file, outpth))
    with open(log_file, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        line = line.split(',')
        epoch = int(line[0])
        train_loss = float(line[1])
        eval_loss = float(line[2])
        eval_top1 = float(line[3])
        eval_top5 = float(line[4])
        writer.add_scalar('loss/train_loss', train_loss, epoch)
        writer.add_scalar('loss/eval_loss', eval_loss, epoch)
        writer.add_scalar('loss/eval_top1', eval_top1, epoch)
        writer.add_scalar('loss/eval_top5', eval_top5, epoch)
        print(line)
    writer.close()
    print("Process Finished")

if __name__ == "__main__":
    main()