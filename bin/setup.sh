source env/bin/activate
export CUDA_VISIBLE_DEVICES=$1
export MMS_ROOT=`pwd`
export PYTHONPATH=$MMS_ROOT:$PYTHONPATH
export PYTHON_EXEC=python
