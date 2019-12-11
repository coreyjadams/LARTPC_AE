#!/usr/bin/env python
import os,sys,signal
import time

import numpy

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)


# import the necessary
from src.utils import flags


def main():

    # If you extend the flags class, change this line! 
    FLAGS = flags.dimnet()
    FLAGS.parse_args()
    # FLAGS.dump_config()

    

    if FLAGS.MODE is None:
        raise Exception("You must select a mode for running.")

    if FLAGS.DISTRIBUTED:
        from src.utils import distributed_trainer

        trainer = distributed_trainer.distributed_trainer()
    else:
        from src.utils import trainercore
        trainer = trainercore.trainercore()
        
    if FLAGS.MODE == 'train' or FLAGS.MODE == 'inference':
        
        # On these lines, you would get the network class and pass it to FLAGS
        # which can share it with the trainers.  This lets you configure the network
        # without having to rewrite the training interface each time.
        # It would look like this:

        trainer.initialize()
        trainer.batch_process()

    if FLAGS.MODE == 'iotest':
        trainer.initialize(io_only=True)

        total_start_time = time.time()
        # time.sleep(0.1)
        start = time.time()
        force_pop=False
        for i in range(FLAGS.ITERATIONS):
            mb = trainer.fetch_next_batch(force_pop=force_pop)
            end = time.time()
            if not FLAGS.DISTRIBUTED:
                print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            else:
                if trainer._rank == 0:
                    print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            # time.sleep(0.5)
            start = time.time()
            force_pop=True

        total_time = time.time() - total_start_time
        print("Time to read {} batches of {} images each: {}".format(
            FLAGS.ITERATIONS, 
            FLAGS.MINIBATCH_SIZE,
            time.time() - total_start_time
            ))

    trainer.stop()

if __name__ == '__main__':
    main()