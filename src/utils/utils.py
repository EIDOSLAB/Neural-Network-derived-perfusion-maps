import os, re, json, random, torch, wandb, math

import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from tqdm import tqdm

from utils.preproc import *


def get_data_list(imgs_path):
    '''
    This function creates the volumes-maps pair for each patient. We consider only registred
    and filtered data. It returns a list of lists.
    '''
    # change re.findall() to improve code
    data_list = []
    _path = os.listdir(imgs_path)
    _path.sort()
    for p in _path:
        #if re.findall("Registered", p) \
            #and not re.findall("Registered_Filtered_3mm_20HU_Maps", p) \
            #and not re.findall("Registered_Filtered_3mm_20HU", p):
        if re.findall("Registered_Filtered_3mm_20HU", p) \
            and not re.findall("Registered_Filtered_3mm_20HU_Maps", p):
            #not re.findall("Registered", p):
            data_list.append(p)

    label_list = []
    for p in _path:
        if re.findall("Registered_Filtered_3mm_20HU_Maps", p) \
            and not re.findall(".txt", p):
            label_list.append(p)

    joint_list = []
    for d in data_list:
        for l in label_list:
            if l.split("-")[-1].split("_")[0]==d.split("-")[-1].split("_")[0]:
                joint_list.append([d, l])

    return joint_list

def get_links(path, maps=False):
    '''
    description
    '''

    if maps:
        _dict = {}
        _path = os.listdir(path)
        _path.sort()
        for typ in _path:
            _dict[typ] = join_imgs(os.path.join(path, typ))
        return _dict
    else:
        return join_imgs(path)

def join_imgs(path):
    _dict = defaultdict(list)
    im_list = []
    _path = os.listdir(path)
    _path.sort()
    for im in _path:
        im_path = os.path.join(path,im)
        medical_image = dicom.read_file(im_path)
        h = str(medical_image.ImagePositionPatient[-1]) 
        _dict[h].append(im_path)

    for h in _dict.keys():
        im_list.append(_dict[h]) 
     
    return im_list

def create_metadata(_path):

    '''
    This function is responsible for creating or loading the json file containing the information 
    regarding volume-map pairs. Returns a dictionary in which 8 volumes and their maps (CBV, CBF, 
    TTP, etc.) are associated for each patient.

    '''
    print("Metadata NOT found. Dataset creation...")
    meta_name = "metadata"
    
    if not os.path.exists(os.path.join(_path,meta_name)):
        os.makedirs(os.path.join(_path,meta_name))
    
    out_path = os.path.join(_path, meta_name, "unito-brain-metadata.json")
    images_list, patients_list = [], []
    mp_dict = defaultdict(list)

    # get the joint list of volumes-maps
    joint_list = get_data_list(_path)
    n = 1
    for img, map in tqdm(joint_list, desc="Metadata creation"):

        name, num = img.split("-")
        num = num.split("_")[0]
        patient = f"{name}_{num}"

        img_ = get_links(os.path.join(_path, img))
        map_ = get_links(os.path.join(_path, map), maps=True)


        for i in range(len(img_)):
            # just to check if the considered patient has 89 instants of detection
            if len(img_[i])==89:
                images_list.append(img_[i])
                patients_list.append(patient)

                for typ in map_.keys():
                    mp_dict[typ].append(map_[typ][i])

                dict_out = {
                    "patients": patients_list,
                    "images": images_list, 
                    "maps": mp_dict}
        
        print(
            f"Patient {patient} processed ({round(n/len(joint_list)*100, 2)}% total: {len(joint_list)})"
            )
        n+=1

    with open(out_path, 'w') as outfile:
        json.dump(dict_out, outfile)
    return dict_out

def split_data(_list, split_type, val_split, test_split):
    '''
    This function allows us to divide patients so that they are not simultaneously present 
    in the train-test-val dataset. It returns the patient code, the volume and the map (the 
    map depends on the type selected).
    '''

    random.seed(42)
    images, maps, patients = _list
    unique_list = list(set(patients))
    unique_list.sort()
    #random.shuffle(unique_list)

    tot_pat = len(unique_list)
    val_split = int(math.trunc(tot_pat*val_split))
    test_split = int(math.trunc(tot_pat*test_split))
    train_split = tot_pat-val_split-test_split

    if split_type=='train':
        pat_considered = unique_list[:train_split]
    elif split_type=='val':
        pat_considered = unique_list[train_split:train_split+val_split]
    elif split_type=='test':
        pat_considered = unique_list[train_split+val_split:]
    
    print(f'-num of patients {split_type} set {len(pat_considered)}/{tot_pat}')

    _img, _map, _pat = [], [], []
    for s in pat_considered:
        for idx in find_indices(patients, s):
            _img.append(images[idx])
            _map.append(maps[idx])
            _pat.append(patients[idx])
    
    return _img, _map, _pat


def split_data_cv(_list, split_type, val_split, k):
    '''
    This function allows us to divide patients so that they are not simultaneously present 
    in the train-test-val dataset. It returns the patient code, the volume and the map (the 
    map depends on the type selected).
    Moreover, cross validation is performed in the train+valdation partition. 
    '''

    random.seed(42)
    images, maps, patients = _list
    unique_list = list(set(patients))
    unique_list.sort()

    tot_pat = len(unique_list)-14
    val_split = int(math.trunc(tot_pat*val_split))
    train_split = tot_pat-val_split

    if not split_type=='test':
        test_patients = [
            "MOL-001",  "MOL-002",  "MOL-003",  "MOL-004",  "MOL-005", 
            "MOL-059",  "MOL-141",  "MOL-142",  "MOL-163",  "MOL-164",
            "MOL-175",  "MOL-190",  "MOL-207",  "MOL-260"
        ]

        pat_considered = set(unique_list) - set(test_patients)

        pat_considered = list(pat_considered)

        pat_val = set(pat_considered[(k)*val_split:(k+1)*(val_split)])
        pat_train = set(pat_considered) - pat_val

        if split_type=='train':
            pat_considered = list(pat_train)
        elif split_type=='val':
            pat_considered = list(pat_val)

    elif split_type=='test':
        pat_considered = [
            "MOL-001",  "MOL-002",  "MOL-003",  "MOL-004",  "MOL-005", 
            "MOL-059",  "MOL-141",  "MOL-142",  "MOL-163",  "MOL-164",
            "MOL-175",  "MOL-190",  "MOL-207",  "MOL-260"
        ]
    
    print(f'-num of patients {split_type} set {len(pat_considered)}/{tot_pat}')

    print('List of patients:', pat_considered)

    _img, _map, _pat = [], [], []
    for s in pat_considered:
        for idx in find_indices(patients, s):
            _img.append(images[idx])
            _map.append(maps[idx])
            _pat.append(patients[idx])
    
    return _img, _map, _pat

def find_indices(lst, a):
    return [i for i, x in enumerate(lst) if x==a]

def plt_map(args, target, output, typ, pat, epoch, k, fold):

    '''
    This function creates an image with prediction and ground-truth
    '''

    #output = normalization(output)
    epoch = f'epoch_{epoch}'
    out_path = os.path.join(
        args.out_maps_dir, args.tag, args.map_type, typ, str(fold), str(epoch) #, pat
        )
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    f, axarr = plt.subplots(1,2, figsize=(15, 15))
    axarr[0].imshow(target.squeeze(0), cmap='jet')
    axarr[1].imshow(output.squeeze(0), cmap='jet')
    #wandb.log({f"chart_{typ}": plt})
    plt.savefig(os.path.join(
        out_path,
        f'{pat}_{args.map_type}_{typ}_{epoch}_{k}.png'), bbox_inches='tight'
        )
    plt.close(f)

def get_single_imgs_maps(
    self, imgs, mps, pat, im_size
    ):
    '''
    This function read metadata and create images and mapss for each patient according to their 
    height and instant. The funtion loads image/map as torch tensor or create it.
    '''
    img = torch.load(os.path.join(img_dir, f"{pat}_{k}.pt"))
    mp = torch.load(os.path.join(mp_dir, f"{pat}_{k}.pt"))
    

            
    return torch.cat(img_list), torch.cat(mp_list), pat_list

def get_imgs_maps(
    imgs_list, mps_list, patients, im_size, img_directory, map_type
    ):
    '''
    This function read metadata and create images and mapss for each patient according to their 
    height and instant. The funtion loads image/map as torch tensor or create it.
    '''

    img_dir, mp_dir = dump_paths(img_directory, map_type)

    img_list = []
    mp_list = []
    pat_list = []

    processed_pat = []


    for n, (im_pth, mk_pth) in enumerate(zip(imgs_list, mps_list)):
        pat = patients[n]

        # to check the correct number of acquisition time
        if len(im_pth)!=89:
            continue

        if pat not in processed_pat:
            k=0
            img_pat = []
            mp_pat = []
            processed_pat.append(pat)
        else:
            k+=1

        if os.path.exists(os.path.join(img_dir, f"{pat}_{k}.pt")) and os.path.exists(os.path.join(mp_dir, f"{pat}_{k}.pt")):
            #img = torch.load(os.path.join(img_dir, f"{pat}_{k}.pt"))
            mp = torch.load(os.path.join(mp_dir, f"{pat}_{k}.pt"))


            # to check if there are empty volumes or maps. If True, we don't consider this pair of data
            #if torch.std(img)==0 or torch.std(mp)==0 or torch.max(img)==0 or torch.max(mp)==0:
                    #continue
            if torch.std(mp)==0 or torch.max(mp)==0:
                    continue
           
            img_list.append(os.path.join(img_dir, f"{pat}_{k}.pt"))#(img.unsqueeze(0))
            mp_list.append(os.path.join(mp_dir, f"{pat}_{k}.pt"))#(mp.unsqueeze(0))
            pat_list.append(pat)
            print(
                    f"Images and maps (loaded): {round((n+1)/len(imgs_list)*100, 2)}% total: {(n+1)}/{len(imgs_list)}"
                    )

        else:
            mp = create_tensor(mk_pth, im_size)

            # we cheack if an image or a map (or both) is empty
            #if torch.std(img)==0 or torch.std(mp)==0 or torch.max(img)==0 or torch.max(mp)==0:
                #continue
            if torch.std(mp)==0 or torch.max(mp)==0:
                    continue
            
            img = create_tensor(im_pth, im_size)

            torch.save(img, os.path.join(img_dir, f"{pat}_{k}.pt"))
            torch.save(mp, os.path.join(mp_dir, f"{pat}_{k}.pt"))

            '''
            # NOTE: use the following code if you want to create volume with shapes [sections, time, height, width]
            if k<7:
                img_pat.append(img.unsqueeze(0))
                mp_pat.append(mp.unsqueeze(0))
            
            elif k==7:
                img_pat.append(img.unsqueeze(0))
                mp_pat.append(mp.unsqueeze(0))

                img_list.append(torch.cat(img_pat).unsqueeze(0))
                mp_list.append(torch.cat(mp_pat).unsqueeze(0))
                pat_list.append(pat)

                print(
                    f"Image and maps (created): {round((n+1)/len(imgs_list)*100, 2)}% total: {(n+1)}/{len(imgs_list)}"
                    )
            '''
            img_list.append(os.path.join(img_dir, f"{pat}_{k}.pt"))#(img.unsqueeze(0))
            mp_list.append(os.path.join(mp_dir, f"{pat}_{k}.pt"))#(mp.unsqueeze(0))
            pat_list.append(pat)
            print(
                    f"Images and maps (created): {round((n+1)/len(imgs_list)*100, 2)}% total: {(n+1)}/{len(imgs_list)}"
                    )
            
    return img_list, mp_list, pat_list#torch.cat(img_list), torch.cat(mp_list), pat_list

def dump_paths(img_directory, map_type):

    img_dump = os.path.join(img_directory, "dump", f"images")
    mp_dump = os.path.join(img_directory, "dump", f"maps", map_type) #_{im_size}

    for d in [img_dump, mp_dump]:
        if not os.path.exists(d):
            os.makedirs(d)

    return img_dump, mp_dump

def get_metadata(img_directory):
    
	if not os.path.exists(
		os.path.join(img_directory, "metadata", "unito-brain-metadata.json")
	):
		data = create_metadata(img_directory)
	else:
		print("Metadata found. Loading dataset...")
		with open(
			os.path.join(
				img_directory,"metadata", "unito-brain-metadata.json")
			) as json_file:
			data = json.load(json_file)
	return data