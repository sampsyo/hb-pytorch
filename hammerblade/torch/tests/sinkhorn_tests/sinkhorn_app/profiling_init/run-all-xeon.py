import sys
import os
import json
import copy
import subprocess

TEST_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_sinkhorn_init.py')

def fancy_print(route):
  for waypoint in route:
    print(waypoint)

with open('sinkhorn_wmd.json',) as f:
  route = json.load(f)

fancy_print(route)
print()

print("total number of jobs: " + str(len(route)))

for i in range(len(route)):
  for j in range(5): # range(5) is default
    cmd = copy.deepcopy(route)
    cmd[i]['offload'] = True
    fancy_print(cmd)
    print()
    name = "sinkhorn_wmd_xeon_%d_%d" % (i, j)
    sh_cmd = "mkdir -p " + name
    print(sh_cmd)
    os.system(sh_cmd)
    with open(name + "/sinkhorn_wmd.json", 'w') as outfile:
      json.dump(cmd, outfile, indent=4, sort_keys=True)
    # sh_cmd = "cp -r data/ " + name + "/"
    # print(sh_cmd)
    # os.system(sh_cmd)

   # sh_cmd = "ln -s /work/global/lc873/work/sdh/playground/recsys_data/pytorch-apps/recsys/data " + name + "/data"
   # print(sh_cmd)
   # os.system(sh_cmd)
    script = "(cd " + name + "; python " + TEST_DIR + " > out.std 2>&1)"
    with open(name + "/run.sh", 'w') as outfile:
      outfile.write(script)
    print("starting cosim job ...")
    # run the job 3 times
    for runs in range(3):
      cosim_run = subprocess.Popen(["sh", name + "/run.sh"], env=os.environ)
    cosim_run.wait()
