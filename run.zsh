best=0.4
for rep in `seq 100`
    do
    echo - - attempt \#$rep - -
    rm neural_network.pickle
    for i in `seq 10`
        do
        echo ~run \#$i~
        python3 test_filter.py
        result1=`python3 test_quality.py "1"`
        result2=`python3 test_quality.py "2"`
        echo $result1
        echo $result2
        target=0.8
        if (( $(echo "$result2 > $best" | bc -l) )); then
            best=$result2
            echo new: $best
            cp neural_network.pickle nn_best.pickle
        fi
    done
done
echo best: $best