"""
This file contains definitions of useful data structures and the paths
for the datasets and data files necessary to run the code.
"""
import os
ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATASET_DIR = "data/lidarhuman26M"
SMPL_FILE = os.path.join(DATA_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
# seems all zeros.
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(DATA_DIR, 'J_regressor_extra.npy')

"""
Each dataset uses different sets of joints.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are:
0 - Right Ankle
1 - Right Knee
2 - Right Hip
3 - Left Hip
4 - Left Knee
5 - Left Ankle
6 - Right Wrist
7 - Right Elbow
8 - Right Shoulder
9 - Left Shoulder
10 - Left Elbow
11 - Left Wrist
12 - Neck (LSP definition)
13 - Top of Head (LSP definition)
14 - Pelvis (MPII definition)
15 - Thorax (MPII definition)
16 - Spine (Human3.6M definition)
17 - Jaw (Human3.6M definition)
18 - Head (Human3.6M definition)
19 - Nose
20 - Left Eye
21 - Right Eye
22 - Left Ear
23 - Right Ear
"""

JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18,
              20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]
