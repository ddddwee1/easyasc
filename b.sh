cd test_cust_op
bash build.sh
cd build_out
for f in custom_*.run; do bash "$f"; done
export LD_LIBRARY_PATH=/home/ddwe/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
cd ../../test_cust_op_aclnn_test
python setup_aclnn.py
