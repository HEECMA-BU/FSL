/home/cc/FSL/env/bin/python3 learner_script_0.py cifar10 0 &
/home/cc/FSL/env/bin/python3 learner_script_1.py cifar10 1 &
/home/cc/FSL/env/bin/python3 learner_script_2.py cifar10 2 &
/home/cc/FSL/env/bin/python3 learner_script_3.py cifar10 3 &

wait

/home/cc/FSL/env/bin/python3 attack_script.py 4 0 &
/home/cc/FSL/env/bin/python3 attack_script.py 7 1 &
/home/cc/FSL/env/bin/python3 attack_script.py 16 2 &
/home/cc/FSL/env/bin/python3 attack_script.py 30 3 &

wait