import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
import os
import torch
# Print iterations progress
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


def pickle_to_csv(input_file, output_file):
    input = pd.read_pickle(input_file)
    input.to_csv(output_file,index=None)

# pickle_to_csv('best_viewpoints_data/best_viewpoints_label_10_test.pkl','best_viewpoints_data/best_viewpoints_label_10_test.csv')

def csv_to_pickle(input,output='',column='sequence'):
    df = pd.read_csv(input)

    # df['sequence'] = df['sequence'].apply(literal_eval)
    # df = df.drop('Unnamed: 0',axis=1)
    df = df.drop('Unnamed: 0',axis=1)
    length = len(df)
    interval = int(length/100)
    printProgressBar(0, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i in range(len(df[column])):
        sequence = []
        
        txt_sequence = df[column][i].replace(']','')
        txt_sequence = txt_sequence.replace('[','')
        txt_sequence = txt_sequence.replace(',','')
        txt_sequence = txt_sequence.split()
        for char in txt_sequence:
            sequence.append(int(char))
        df[column][i] = np.asarray(sequence)

        if i % interval == 0:
            printProgressBar(i, length, prefix = 'Progress:', suffix = 'Complete', length = 50)




    print(df)
    df.to_pickle(output)

def string_to_tuple(string):
    string = string.replace('(',"")
    string = string.replace(')',"")
    string = string.split(',')
    string = [int(value) for value in string]
    # print(tuple(string))
    return tuple(string)

def string_to_array(string):
    string = string.replace('(',"")
    string = string.replace(')',"")
    string = string.replace('[',"")
    string = string.replace(']',"")
    string = string.split(',')
    string = [float(value) for value in string]
    # print(tuple(string))
    return string

def obj_idn_to_string(obj_idn):
    obj_idn = int(obj_idn)
    if obj_idn < 10:
        return '000' + str(obj_idn)
    elif obj_idn < 100:
        return '00' + str(obj_idn)
    elif obj_idn <1000:
        return '0' + str(obj_idn)
    return str(obj_idn)

def string_array_to_tuple(string_array):
    new_array = []
    string_array = string_array.replace('[ ','')
    string_array = string_array.replace('] ','')
    string_array = string_array.replace('[','')
    string_array = string_array.replace(']','')
    string_array = string_array.replace('  ',' ')
    new_array = string_array.split(' ')
    new_array = [int(vp) for vp in new_array]
    # print(tuple(new_array))
    # exit()
    return tuple(new_array)
# data = pd.read_csv('../best_viewpoints_data/best_sequence_pair_train_maxpool.csv')

# data['sequence'] = [string_array_to_tuple(sequence) for sequence in data['sequence'].values]
# print(data)
# data.to_csv('../best_viewpoints_data/best_sequence_pair_train_maxpool.csv',index=False)




def vis_saliency_map(pcd, label, obj_ind,color_map,epoch):
    # pcd = o3d.io.read_point_cloud('{}/full_pcd/{}/{}/pcd_full_{}_{}.pcd'.format(data_path,'test',label,label,obj_ind))
    points = np.asarray(pcd.points)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    marker_data = go.Scatter3d(
        x=x, 
        y=y, 
        z=z, 
        marker=dict(
            color=color_map,
            size=8,
            colorbar=dict(
                title="Colorbar"
            ),
            colorscale="Viridis"
        ), 
        mode='markers',
        
    )
    
    fig=go.Figure(data=marker_data)
    fig.update_layout(title="Epoch: {}".format(epoch))
    fig.show()

def feature_visualization(data_path,model,dset_train,test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on {}'.format(device))
    
    class_to_idx = dset_train.class_to_idx
    idx_to_class = {}

    for (label, idx) in class_to_idx.items():
        idx_to_class[idx]=label

    for i, (inputs, targets,obj_inds) in enumerate(test_loader):
        inputs= inputs.type(torch.FloatTensor)
        inputs= inputs.to(device)
        inputs = inputs.permute(0,2,1)

        outputs, matrix3x3, matrix64x64,xb = model(inputs)
        outputs = torch.argmax(outputs,axis=1)
        xb = xb.detach().cpu().numpy()

        for i in range(len(obj_inds)):
            target = targets.numpy()[i]
            label = idx_to_class[target]
            if target == outputs.cpu().numpy()[i]:
                print("correct prediction")
            filename_pcd = 'pcd_full_{}_{}.pcd'.format(label,obj_inds[i])
            pcd_path = os.path.join(data_path,'full_pcd','test',label,filename_pcd)
            full_pcd = o3d.io.read_point_cloud(pcd_path)
            vis_saliency_map(full_pcd,label,obj_inds[i],xb[i])

        exit()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def my_accuracy(output_, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    target = target[0:-1:nview]
    batch_size = target.size(0)

    num_classes = output_.size(2)
    output_ = output_.cpu().numpy()
    output_ = output_.transpose( 1, 2, 0 )
    scores = np.zeros( ( vcand.shape[ 0 ], num_classes, batch_size ) )
    output = torch.zeros( ( batch_size, num_classes ) )
    # compute scores for all the candidate poses (see Eq.(6))
    for j in range(vcand.shape[0]):
        for k in range(vcand.shape[1]):
            scores[ j ] = scores[ j ] + output_[ vcand[ j ][ k ] * nview + k ]
    # for each sample #n, determine the best pose that maximizes the score (for the top class)
    for n in range( batch_size ):
        j_max = int( np.argmax( scores[ :, :, n ] ) / scores.shape[ 1 ] )
        output[ n ] = torch.FloatTensor( scores[ j_max, :, n ] )
    output = output.to(device)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class view_matrix():
    def __init__(self,x,y,start_cell):
        self.x = x
        self.y = y
        self.stack = np.array(range(x*y))
        self.cells = self.stack.reshape(5,8)
        start = self.stack[start_cell:]
        end = self.stack[0:start_cell]
        start = np.append(start,end)
        self.stack = start

    def get_distances(self,c):
        distances = np.zeros((self.x,self.y))
        for x in range(self.x):
            for y in range(self.y):
                # Calculate euclidean distance from c
                distances[x,y] = abs(x-c[0])+abs(y-c[1])

        distances[c[0],c[1]] = 1000
        return np.asarray(distances)
    def del_from_stack(self,cell):
        self.stack = np.delete(self.stack,0)


def count_files(path):
    count = 0

    for set in os.listdir(path):
        if os.path.isdir(os.path.join('modelnet40_40/image',set)):
                labels = os.listdir(os.path.join('modelnet40_40/image',set))
                for label in labels:
                        files = os.listdir(os.path.join('modelnet40_40/image',set,label))
                        count += len(files)
    print(count)