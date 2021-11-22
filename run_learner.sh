# allThreads=(1 2 3 4)
# for i in ${allThreads[@]};
# do
#     echo $i
#     /home/cc/FSL/env/bin/python3 /home/cc/split-learning/main_m2_20_equal_dataset.py pc cifar10 $i &
#     # /home/cc/FSL/env/bin/python3 /home/cc/split-learning/main_m2_20_equal_dataset.py pc cifar10 $i
# done

/home/cc/FSL/env/bin/python3 learner_script_0.py cifar10 0 &
/home/cc/FSL/env/bin/python3 learner_script_1.py cifar10 1 &
/home/cc/FSL/env/bin/python3 learner_script_2.py cifar10 2 &
/home/cc/FSL/env/bin/python3 learner_script_3.py cifar10 3 &

wait