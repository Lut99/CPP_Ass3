# TEST.py
#   by Group 44
#
# This file tests multiple configurations for simulation.cu.

import argparse
import subprocess
import sys

APS = [10**i for i in range(3, 8)]
TIME_STEPS = [10**i for i in range(8, 3, -1)]
BLOCK_SIZES = [2**i for i in range(3, 11)]

def main(command, result_path, baseline_path):
    print("*** test.py for CUDA Wave Simulation ***\n")
    
    for bsize in BLOCK_SIZES:
        for i, ap in enumerate(APS):
            timestep = TIME_STEPS[i]

            # Run the command
            to_run = command.split()
            to_run += [str(ap), str(timestep), str(bsize)]
            (stdtext, _) = subprocess.Popen(to_run, stdout=subprocess.PIPE).communicate()
            # Search the stdtext for a result
            # target: wave timer               : avg =  109 ms, total =  109 ms, count =         1
            time = None
            for line in stdtext.decode("utf-8").split("\n"):
                if line[:10] == "wave timer":
                    total_pos = line.find("total =")
                    ms_pos = line[total_pos:].find("ms")
                    try:
                        time = int(line[total_pos + 8:ms_pos - 1])
                    except ValueError:
                        # Invalid time format; quit
                        sys.stderr.write("Invalid format: \"{}\"".format(line[total_pos + 8:ms_pos - 1]))
                        break
            if time is None:
                sys.stderr.write("Time not found")
                continue

            # Compare the accuracy of this with the sequential file
            # TODO



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="Command to run the file. Note that after this commands, the arguments of the file will be appended in the order: i_max t_max block_size", default="prun -v -np 1 -native '-C GTX480 --gres=gpu:1' ./simulation")
    parser.add_argument("--results", help="Path to the file with results from the run program", default="result.txt")
    parser.add_argument("--baseline", help="Path to the file with results from the sequential implementation", default="sequential.txt")

    args = parser.parse_args()

    main(args.command, args.results, args.baseline)