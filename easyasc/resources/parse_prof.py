import os 
import sys 
import re
import glob 
import shutil
import json 

from typing import Union, Iterator


class Path(os.PathLike[str]):
    path: str

    def __init__(self, path: str):
        self.path = path 

    def __str__(self) -> str:
        return self.path
    
    def __fspath__(self) -> str:
        return self.path 

    @staticmethod
    def _tostr(obj: Union[str, 'Path']) -> str:
        assert isinstance(obj, (str, Path))
        if isinstance(obj, str):
            return obj 
        else:
            return obj.path 

    def __add__(self, other: Union[str, 'Path']) -> 'Path':
        path_new = os.path.join(self.path, self._tostr(other))
        return Path(path_new)
    
    def __radd__(self, other: Union[str, 'Path']) -> 'Path':
        path_new = os.path.join(self._tostr(other), self.path)
        return Path(path_new)
    
    def __contains__(self, s: str):
        return s in self.path
    
    def __getitem__(self, idx: int) -> str:
        return self.path.split('/')[idx]

    def find_one(self, pattern: str, use_regex: bool = False) -> 'Path':
        if not use_regex:
            pattern = '^' + pattern.replace('*', '.*') + '$'
        new_path = None 
        for item in glob.glob(os.path.join(self.path, '*')):
            result = re.search(pattern, item.split('/')[-1])
            if result is not None:
                new_path = item 
        if new_path is not None:
            return Path(new_path)
        else:
            raise FileNotFoundError(f'Cannot find pattern {pattern} in folder {self.path}')
        
    def find(self, pattern: str, use_regex: bool = False) -> Iterator['Path']:
        if not use_regex:
            pattern = '^' + pattern.replace('*', '.*') + '$'
        for item in glob.glob(os.path.join(self.path, '*')):
            result = re.search(pattern, item.split('/')[-1])
            if result is not None:
                yield Path(item)

    def auto(self) -> 'Path':
        sub_paths = glob.glob(os.path.join(self.path, '*'))
        if len(sub_paths)==0:
            raise FileNotFoundError(f'Auto: No file under {self.path}')
        else:
            return Path(sub_paths[0])
    
    def parent(self) -> 'Path':
        hierarchy = self.path.split('/')
        if len(hierarchy)<=1:
            raise ValueError(f'{self.path} does not have parent')
        new_path = '/'.join(hierarchy[:-1])
        return Path(new_path)
        
    def copy_to(self, target: Union[str, 'Path']):
        if isinstance(target, str):
            target = Path(target)
        os.makedirs(os.path.dirname(target.path), exist_ok=True)
        shutil.copy(self.path, target.path)
    
    def copy_dir_to(self, target: Union[str, 'Path']):
        if isinstance(target, str):
            target = Path(target)
        os.makedirs(os.path.dirname(target.path), exist_ok=True)
        if os.path.exists(target.path):
            target.remove_dir()
        shutil.copytree(self.path, target.path)

    def remove(self, verbose: bool=True):
        try:
            os.remove(self.path)
        except Exception as e:
            if verbose:
                print(f'Cannot remove {self.path}. {e}')
    
    def remove_dir(self, verbose: bool=True):
        try:
            shutil.rmtree(self.path)
        except Exception as e:
            if verbose:
                print(f'Cannot remove {self.path}. {e}')

    def makedirs(self):
        os.makedirs(self.path, exist_ok=True)
    
    def replace(self, src: str, tgt: str) -> 'Path':
        path_new = self.path.replace(src, tgt)
        return Path(path_new)

    def exists(self) -> bool:
        return os.path.exists(self.path)
    
    def basename(self) -> str:
        return os.path.basename(self.path)
    
    def dirname(self) -> 'Path':
        return Path(os.path.dirname(self.path))


opname = ''
if len(sys.argv)>1:
    opname = sys.argv[1]
prof_path = Path('./').find_one('PROF_*')

print('>> Start parsing profiling data...')
os.system('msprof --parse=on --output=%s > /dev/null'%prof_path)
os.system('msprof --export=on --output=%s > /dev/null'%prof_path)

json_path = (prof_path + 'mindstudio_profiler_output').find_one('msprof_*.json')

def anyin(candidates, text):
    for c in candidates:
        if c in text:
            return True 
    return False

durations = []

data = json.load(open(json_path))
res = []
pid_dict = {}
for i in data:
    if i['name']==opname:
        durations.append(i['dur'])
    if 'args' in i:
        if 'name' in i['args']:
            if anyin(['CANN', 'Ascend Hardware', 'HCCL', 'Overlap'], i['args']['name']):
                pid_dict[i['pid']] = 1 

for i in data:
    if i['pid'] in pid_dict:
        res.append(i)

os.makedirs('../traces', exist_ok=True)
json.dump(res, open(os.path.join('../traces', json_path[-1]), 'w'))

prof_path.remove_dir()

if len(durations)>80:
    print("---- N_TESTS: %d   AVG: %.2fus  FirstRun: %.2fus  ----"%(len(durations), sum(durations[25:75])/50, durations[0]))

