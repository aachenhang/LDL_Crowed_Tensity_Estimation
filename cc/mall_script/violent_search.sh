for((i = 3; i < 10; i++))
do
    python `echo make_net_$i.py` 
    python run.py
done
