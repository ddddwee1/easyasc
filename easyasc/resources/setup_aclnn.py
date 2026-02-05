import os 

ascend_op_extension_name = 'test_aclnnop'
ascend_op_src_files = ['test.cpp']
ascend_toolkit_install_path='/home/ma-user/work/ascend-toolkit/latest'
custom_op_path = ascend_toolkit_install_path + '/opp'

ascend_include_paths=[
    ascend_toolkit_install_path + '/aarch64-linux/include',
    ascend_toolkit_install_path + '/acllib/include',
    custom_op_path + '/vendors/customize/op_api/include/',
    './'
]
ascend_libraries = ['runtime', 'ascendcl', 'stdc++', 'hccl', 'nnopbase', 'opapi', 'msprofiler', 'cust_opapi']
ascend_library_paths = [
    ascend_toolkit_install_path + '/runtime/lib64',
    ascend_toolkit_install_path + '/aarch64-linux/lib64',
    custom_op_path + '/vendors/customize/op_api/lib/',
]

cmd = 'g++ -O2 '
cmd += ' '.join(ascend_op_src_files) + ' '

cmd += ' '.join(['-L'+p for p in ascend_library_paths]) + ' '
cmd += ' '.join(['-I'+p for p in ascend_include_paths]) + ' '
cmd += ' '.join(['-l'+l for l in ascend_libraries]) + ' '
cmd += '-o ' + ascend_op_extension_name + ' -std=c++17 -pthread '
cmd += '-Wl,-rpath=' + ascend_toolkit_install_path + '/aarch64-linux/lib64 '

print('--> Compiling aclnn testcase...')
os.system(cmd)
