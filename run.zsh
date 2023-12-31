stop=0
for rep in `seq 50`
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
        if (( $(echo "$result2 > $target" | bc -l) )); then
            stop=1
            break
        fi
    done
    if (( $stop == 1 )); then
        echo "Found satisfactory network, saved to neural_network.pickle"
        break
    fi
done