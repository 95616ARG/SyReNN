rm -f *.eran *.nnet eran

make docker-image

models="pendulum pendulum_change_l pendulum_change_m pendulum_continuous satelite quadcopter"

for model in $models
do
    echo "Converting $model"
    make savenet MODEL=$model
    python3 to_eran.py $model.npz $model.eran
done

rm -rf *.npz
mkdir -p eran
mv *.eran eran
