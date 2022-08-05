'''
data_creation.py
1 Creates data from the 3D object:
    o 1 full point cloud sampled from the objects mesh
    o 2 augmented full point clouds sampled from the objects mesh
    - For the three point clouds, the following is extracted
        o 40 2D images from 40 views around the object
            o For each image the silhouette is extracted
            o For each silhouette the edges ad detected
            o For each edge the number of pixels is counted
        o 40 depth images from 40 views 
            o For each image the shannon entropy is calculated
        o 40 point clouds from the 40 views projected on the view from the point cloud
            o For each point cloud the number of point is counted

2 Creates a Saliency score for each viewpoint and a saliency map for each full and partial point cloud (saliency value per point)
    - The 2D images are used to train an MVCNN model
    - The MVCNN model evaluates the performance of all the possible pairs of 2D images from every viewpoint around an object
        o The scores are accumelated for each viewpoint in each training instance (pair), to create a score per viewpoint
        o The scores per viewpoint are related to the point in each partial point cloud, accumulated scores, creating a 3D saliency map
        o Partial 3D saliency maps are extracted from the full 3D saliency map
        
'''
import argparse
from open3d import *
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

import time
import os

from data_creation_util import printProgressBar, normalize3d, random_down_sample

parser = argparse.ArgumentParser(description="Generates views regularly positioned on a sphere around the object.")
parser.add_argument("--data", help="Specify root directory to the ModelNet10 dataset.")
parser.add_argument("--set", help="Subdirectory: 'train' or 'test'.", default='train')

parser.add_argument("--save_depth", help="Add to also save the correspondent depth-views dataset.", action='store_true')
parser.add_argument("--save_image", help="Add to save image views", action='store_true')

parser.add_argument("--save_pcd", help="Add to also save the correspondent pcd dataset.", action='store_true')
parser.add_argument("--save_part_pcd", help="Add to also save the correspondent partial pcd for every view dataset.", action='store_true')

parser.add_argument("--out", help="Select a desired output directory.", default="./")
parser.add_argument("-h_split", "--horizontal_split", help="Number of views from a single ring (around z). Each ring is divided in x "
                                                     "splits so each viewpoint is at an angle of multiple of 360/x. "
                                                     "Example: -x=12 --> phi=[0, 30, 60, 90, 120, ... , 330].",
                    default=8,
                    metavar='VALUE',
                    type=int
                    )
parser.add_argument("-v_split", "--vertical_split", help="Number of horizontal rings (sliced perpendicular to the z axis)."
                                                   "Each ring of viewpoints is an horizontal section"
                                                   " of a half sphere, looking at the center at an angle "
                                                   "90/(5), Example: -y=5 --> theta=[0,18, 36, 54, 72]",
                    default=3,
                    metavar='VALUE',
                    type=int
                    )

args = parser.parse_args()

OUT_DIR = args.out
DATA_PATH = args.data
SET_TYPE = args.set
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

if args.horizontal_split == 1:
    h_step_angle = 45
else:
    h_step_angle = int(90/args.horizontal_split)
v_step_angle = int(360/args.vertical_split )
# h_step_angle = 36
def rotate_camera_around_object(ctrl,x,y,z,rot=[]):
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


def generate_partial_pc(output_path,full_pcd,vis):
    vis.capture_depth_point_cloud('temp/part_pc.pcd',convert_to_world_coordinate=True,do_render=True)

    part_pc = o3d.io.read_point_cloud('temp/part_pc.pcd')
    part_pc.paint_uniform_color([0,0, 0])
    part_pc = part_pc.points

    full_pcd.paint_uniform_color([0, 0, 0])
    pcd_tree = o3d.geometry.KDTreeFlann(full_pcd)
    partial_pcd_idx = np.asarray([])
    for i in range(len(part_pc)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(part_pc[i], 1)
        if partial_pcd_idx.size == 0:
            partial_pcd_idx = np.asarray(idx)
        else:
            partial_pcd_idx = np.concatenate((partial_pcd_idx,np.asarray(idx)))
    partial_pcd_idx = np.unique(partial_pcd_idx)
    np.save(output_path,np.asarray(partial_pcd_idx))



def generate_views(output_path,object_path, h_split = 5, v_split = 8):
    file_name = os.path.basename(object_path)[0:-4]
    label = file_name[0:-5]
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, visible=False)
    try:
        mesh = o3d.io.read_triangle_mesh(object_path,enable_post_processing=True)
    except:
        print("failed to load mesh for {}").format(file_name)
    mesh.vertices = normalize3d(mesh.vertices)

    mesh.compute_vertex_normals()
    vis.add_geometry(mesh)

    max = mesh.get_axis_aligned_bounding_box().max_bound
    min = mesh.get_axis_aligned_bounding_box().min_bound
    width,height,depth = max - min

    vis.get_render_option().background_color = np.asarray([0,0,0])
    vis.get_render_option().mesh_show_back_face = True

    full_pcd = mesh.sample_points_poisson_disk(1024)

    vis.update_renderer()
    vis.poll_events()


    if args.save_pcd:
        o3d.io.write_point_cloud('{}/full_pc/{}/{}/{}.pcd'.format(output_path,SET_TYPE,label,file_name),full_pcd)

    for v in range(v_split):
        for h in range(h_split):
            x,y,z = -1*h_step_angle,0,0
            rot = R.from_euler('xyz', (x,y,z), degrees=True).as_matrix()

            mesh.rotate(rot,np.asarray(max-[width/2,height/2,depth/2]))
            
            vis.update_geometry(mesh)
            vis.update_renderer()
            vis.poll_events()
        
            if args.horizontal_split == 1:
                new_file_name = "{}_{}_{}".format(file_name,45,int(v*v_step_angle))
            else:
                new_file_name = "{}_{}_{}".format(file_name,int((h_step_angle*(args.horizontal_split-1))-h*h_step_angle),int(v*v_step_angle))
            
            if args.save_image:
                vis.capture_screen_image('{}/image/{}/{}/{}.png'.format(output_path,SET_TYPE,label,new_file_name),do_render=True)
            if args.save_depth:
                vis.capture_depth_image('{}/depth/{}/{}/{}.png'.format(output_path,SET_TYPE,label,new_file_name),do_render=True)
            if args.save_part_pcd:
                vis.capture_depth_point_cloud('temp/temp_pcd.pcd',convert_to_world_coordinate=True,do_render=True)
                part_pcd = o3d.io.read_point_cloud('temp/temp_pcd.pcd')
                part_pcd = random_down_sample(part_pcd,1024)
                o3d.io.write_point_cloud('{}/view_pc/{}/{}/{}.pcd'.format(output_path,SET_TYPE,label,new_file_name),part_pcd)
        
        if args.horizontal_split == 1:
            x,y,z = 45,0,v_step_angle
        else:    
            x,y,z = 90,0,v_step_angle
        rot = R.from_euler('xyz', (x,y,z), degrees=True).as_matrix()
        mesh.rotate(rot,np.asarray(max-[width/2,height/2,depth/2]))

        vis.update_geometry(mesh)
        vis.update_renderer()
        vis.poll_events()


# Create folder if non exist
if os.path.exists(OUT_DIR):
    print("[Error] Folder already exists.")
    if input("Do you want to continue [y/n]") != "y":
        exit(0)
else:
    os.mkdir(OUT_DIR)

def check_folder(label):
    if args.save_part_pcd:
        if not os.path.exists('temp'):
            os.mkdir('temp')
        if not os.path.exists(os.path.join(OUT_DIR,'view_pc')):
            os.mkdir(os.path.join(OUT_DIR,'view_pc'))
        if not os.path.exists(os.path.join(OUT_DIR,'view_pc',SET_TYPE)):
            os.mkdir(os.path.join(OUT_DIR,'view_pc',SET_TYPE))
        if not os.path.exists(os.path.join(OUT_DIR,'view_pc',SET_TYPE,label)):
            os.mkdir(os.path.join(OUT_DIR,'view_pc',SET_TYPE,label))
    if args.save_image:
        if not os.path.exists(os.path.join(OUT_DIR,'image')):
            os.mkdir(os.path.join(OUT_DIR,'image'))
        if not os.path.exists(os.path.join(OUT_DIR,'image',SET_TYPE)):
            os.mkdir(os.path.join(OUT_DIR,'image',SET_TYPE))
        if not os.path.exists(os.path.join(OUT_DIR,'image',SET_TYPE,label)):
            os.mkdir(os.path.join(OUT_DIR,'image',SET_TYPE,label))
    if args.save_depth:
        if not os.path.exists(os.path.join(OUT_DIR,'depth')):
            os.mkdir(os.path.join(OUT_DIR,'depth'))
        if not os.path.exists(os.path.join(OUT_DIR,'depth',SET_TYPE)):
            os.mkdir(os.path.join(OUT_DIR,'depth',SET_TYPE))
        if not os.path.exists(os.path.join(OUT_DIR,'depth',SET_TYPE,label)):
            os.mkdir(os.path.join(OUT_DIR,'depth',SET_TYPE,label))
    if args.save_pcd:
        if not os.path.exists(os.path.join(OUT_DIR,'full_pc')):
            os.mkdir(os.path.join(OUT_DIR,'full_pc'))
        if not os.path.exists(os.path.join(OUT_DIR,'full_pc',SET_TYPE)):
            os.mkdir(os.path.join(OUT_DIR,'full_pc',SET_TYPE))
        if not os.path.exists(os.path.join(OUT_DIR,'full_pc',SET_TYPE,label)):
            os.mkdir(os.path.join(OUT_DIR,'full_pc',SET_TYPE,label))

def check_files_exist(output_path,object_path,h_split,v_split):
    file_name = os.path.basename(object_path)[0:-4]


    if not os.path.isfile(os.path.join(OUT_DIR,'full_pc/{}/{}/{}.pcd'.format(SET_TYPE,label,file_name))):
        return False

    for v in range(v_split):
        for h in range(h_split):
            if args.horizontal_split == 1:
                new_file_name = "{}_{}_{}".format(file_name,45,int(v*v_step_angle))
            else:
                new_file_name = "{}_{}_{}".format(file_name,int((h_step_angle*(args.horizontal_split-1))-h*h_step_angle),int(v*v_step_angle))
            if not os.path.exists('{}/image/{}/{}/{}.png'.format(output_path,SET_TYPE,label,new_file_name)):
                return False
            if not os.path.exists('{}/depth/{}/{}/{}.png'.format(output_path,SET_TYPE,label,new_file_name)):
                return False
            if not os.path.exists('{}/view_pc/{}/{}/{}.npy'.format(output_path,SET_TYPE,label,new_file_name)):
                return False
    return True
# Loop over files
labels = []
for cur in os.listdir(DATA_PATH):
    if os.path.isdir(os.path.join(DATA_PATH, cur)):
        labels.append(cur)

for label in labels:

    check_folder(label)
    files = os.listdir(os.path.join(DATA_PATH, label,SET_TYPE ))
    files.sort()
    for filename in files:  # Removes file without .off extension
        if not filename.endswith('off'):
            files.remove(filename)
    number_of_files = len(files)
    file_num = 1
    start = time.time()
    for filename in files:
        obj_path = os.path.join(DATA_PATH, label, SET_TYPE, filename)


    
        if not check_files_exist(OUT_DIR,obj_path,h_split=args.horizontal_split,v_split=args.vertical_split):
            generate_views(OUT_DIR,obj_path,h_split=args.horizontal_split,v_split=args.vertical_split)


        average_time = (time.time()-start)/file_num
        predicted_time = (len(files)-file_num) * average_time
        printProgressBar(file_num,number_of_files,suffix = filename,prefix='Time left: {:.0f}'.format(predicted_time))
        file_num+=1
        # exit()

for set in os.listdir(path):
    if os.path.isdir(os.path.join(path,set)):
        df = {'label':[],'obj_ind':[],'pose':[],'view':[],'path':[]}
        for label in os.listdir(os.path.join(path,set)):
            if os.path.isdir(os.path.join(path,set,label)):
                for file in os.listdir(os.path.join(path,set,label)):
                    file_name = file.split('.')[0].split('_')
                    df['label'].append(label)
                    df['obj_ind'].append(file_name[-4])
                    df['pose'].append(file_name[-3])
                    if set_view_num == 40:
                        hor_split_n = int(file_name[-2])/18
                        ver_split_n = int(file_name[-1])/45
                        view_num = int(hor_split_n*8 + ver_split_n)
                    else:
                        view_num = int(file_name[-1])/30


                    df['view'].append(view_num)
                    df['path'].append(os.path.join(path,set,label,file))
 
        df = pd.DataFrame(df).sort_values(['label','obj_ind','pose','view'])
        df.to_csv(os.path.join(path,set,'dataset_{}.csv'.format(set)),index=False)
        df.to_pickle(os.path.join(path,set,'dataset_{}.pkl'.format(set)))