model=quadcopter
net_file="$model/$model\_stepnet.nnet"
result_file=results.csv
echo "Network,Spec,Steps,Result,Time" > $result_file

for steps in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    for i in 0 1 2 3
    do
        for j in 0 1 2 3
        do
            echo "NONE" > bmc_output
            spec_file="$model/$model\_0$i\_0$j\_spec.nspec"
            echo "Running: make docker-bmc MODEL=$net_file SPEC=$spec_file STEPS=$steps"
            make docker-bmc MODEL=$net_file SPEC=$spec_file STEPS=$steps
            cat bmc_output >> $result_file
        done
    done
done
