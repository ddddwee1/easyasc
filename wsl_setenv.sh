export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/x86_64-linux/lib64:$ASCEND_HOME_PATH/x86_64-linux/devlib:$ASCEND_HOME_PATH/x86_64-linux/devlib/linux/x86_64:${LD_LIBRARY_PATH}"
export PYTHONPATH="$script_path/py_patch:$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}"
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/lib/gcc/x86_64-linux-gnu/11/include:${CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="/usr/include:/usr/lib/gcc/x86_64-linux-gnu/11/include:${C_INCLUDE_PATH}"