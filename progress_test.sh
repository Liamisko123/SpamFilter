rm neural_network.pickle
for i in `seq 30`
do
echo ~run \#$i~
python3 test_filter.py
done
