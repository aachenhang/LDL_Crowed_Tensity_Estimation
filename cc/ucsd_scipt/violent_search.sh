for((i = 0; i < 10; i++))
do
    python `echo make_net_$i.py` 
    python run_UCSD.py
done
