# allThreads=(1 2 3 4)
# for i in ${allThreads[@]};
# do
#     echo $i
#     /home/cc/FSL/env/bin/python3 /home/cc/split-learning/main_m2_20_equal_dataset.py pc cifar10 $i &
#     # /home/cc/FSL/env/bin/python3 /home/cc/split-learning/main_m2_20_equal_dataset.py pc cifar10 $i
# done

/home/cc/FSL/env/bin/python3 learner_script.py cifar10 0 CUTS 4 &
/home/cc/FSL/env/bin/python3 learner_script.py cifar10 1 CUTS 7 &
/home/cc/FSL/env/bin/python3 learner_script.py cifar10 2 CUTS 16 &
/home/cc/FSL/env/bin/python3 learner_script.py cifar10 3 CUTS 30 &

wait