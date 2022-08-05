from open3d import *
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def rotate_camera_around_object(ctrl,x,y,z,rot=[]):
    # print("angle x: {} Angle y: {}".format(x,y))
    camera_params = ctrl.convert_to_pinhole_camera_parameters()
    if len(rot)==0:
        rot = np.eye(4)
        rot[:3, :3] = R.from_euler('xyz', (x,y,z), degrees=True).as_matrix()
        rot = rot.dot(camera_params.extrinsic.copy())
        rot[:,3] = camera_params.extrinsic[:,3]
        camera_params.extrinsic = rot
        ctrl.convert_from_pinhole_camera_parameters(camera_params)
    else:
        camera_params.extrinsic = rot
        ctrl.convert_from_pinhole_camera_parameters(camera_params)

def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return o3d.utility.Vector3dVector(np_normalized)

def get_label_dict(inverse=False):
    label2int = {'bathtub': 0,
                 'bed': 1,
                 'chair': 2,
                 'desk': 3,
                 'dresser': 4,
                 'monitor': 5,
                 'night_stand': 6,
                 'sofa': 7,
                 'table': 8,
                 'toilet': 9}

    int2label = {0: 'bathtub',
                 1: 'bed',
                 2: 'chair',
                 3: 'desk',
                 4: 'dresser',
                 5: 'monitor',
                 6: 'night_stand',
                 7: 'sofa',
                 8: 'table',
                 9: 'toilet'}
    if inverse:
        return int2label
    else:
        return label2int


def obj_idn_to_string(obj_idn):
    obj_idn = int(obj_idn)
    if obj_idn < 10:
        return '000' + str(obj_idn)
    elif obj_idn < 100:
        return '00' + str(obj_idn)
    elif obj_idn <1000:
        return '0' + str(obj_idn)
    return str(obj_idn)

def random_down_sample(pcd,num_points=1024):
    point_inds = np.random.randint(0,len(pcd.points),num_points)
    downsampled_points = np.asarray(pcd.points)[point_inds]
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.cuda.pybind.utility.Vector3dVector(downsampled_points)
    return downsampled_pcd