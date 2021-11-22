allThreads=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
for i in ${allThreads[@]};
do
    /home/cc/FSL/env/bin/python3 attack_script.py 4 0 $i &
    /home/cc/FSL/env/bin/python3 attack_script.py 7 1 $i &
    /home/cc/FSL/env/bin/python3 attack_script.py 16 2 $i &
    /home/cc/FSL/env/bin/python3 attack_script.py 30 3 $i &
    wait
done

# /home/cc/FSL/env/bin/python3 attack_script.py 4 0 &
# /home/cc/FSL/env/bin/python3 attack_script.py 7 1 &
# /home/cc/FSL/env/bin/python3 attack_script.py 16 2 &
# /home/cc/FSL/env/bin/python3 attack_script.py 30 3 &

# wait