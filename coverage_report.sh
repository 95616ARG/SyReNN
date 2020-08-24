COVTEMP=$PWD/coverage_tmp
rm -rf $COVTEMP
mkdir $COVTEMP

source bazel_python_venv_installed/bin/activate

# Go to the main workspace directory and run the coverage-report.
pushd $BUILD_WORKSPACE_DIRECTORY

# We find all .cov files, which should be generated by tests/helpers.py
cov_zips=$(ls bazel-out/*/testlogs/pysyrenn/*/tests/*/test.outputs/outputs.zip)
i=1
for cov_zip in $cov_zips
do
    echo $cov_zip
    unzip -p $cov_zip coverage.cov > $COVTEMP/$i.cov
    i=$((i+1))
done

# Remove old files
rm -rf .coverage htmlcov

# Then we build a new .coverage as well as export to HTML
python3 -m coverage combine $COVTEMP/*.cov
python3 -m coverage html pysyrenn/*/*.py

# Remove temporaries and go back to where Bazel started us.
rm -r $COVTEMP
popd
