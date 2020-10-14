import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
import imageio
from tqdm import tqdm
import torchvision.transforms as transforms
from utils import *
import json

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

def save_gif(file_path, data, heatmap=None, a=.5, b=.5):
    num_images = int(data.shape[2]/3)
    with imageio.get_writer(file_path, mode='I') as writer:
        for j in range(num_images):
            if heatmap is None:
                img = (data[:,:,3*j:3*j+3] * 255).astype(np.uint8)
            else:
                img = data[:,:,3*j:3*j+3] * a + heatmap * b
                img = (img*255).astype(np.uint8)
            writer.append_data(img)

# push each prototype to the nearest patch in the training set
def push_prototypes(device,
                    dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):
    prototype_network_parallel.eval()

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    global_min_proto_dist = np.full(n_prototypes, np.inf) # saves the closest distance seen so far
    global_min_fmap_patches = np.zeros([n_prototypes,prototype_shape[1],prototype_shape[2], prototype_shape[3]]) # saves the patch representation that gives the current smallest distance
    
    bounding_box_dict = dict()

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, 'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    num_classes = prototype_network_parallel.module.num_classes
    progress = tqdm(dataloader)
    for push_iter, (video_names, frame_indicies, search_batch_data, search_y) in enumerate(progress):
        start_index_of_search_batch = push_iter * search_batch_size
        update_prototypes_on_batch(device,
                                   video_names,
                                   frame_indicies,
                                   search_batch_data,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   bounding_box_dict,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    log('Executing push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    end = time.time()
    log('Push time: \t{0}'.format(end -  start))

# update each prototype for current search batch
def update_prototypes_on_batch(device, 
                               video_names,
                               frame_indicies,
                               search_batch_data,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               bounding_box_dict,
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    with torch.no_grad():
        search_batch_data = search_batch_data.to(device)
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch_data)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    
    for j in range(n_prototypes):
        if class_specific:
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            if len(class_to_img_index_dict[target_class]) == 0: continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
        else:
            proto_dist_j = proto_dist_[:,j,:,:]  # if it is not class specific, then we will search through every example

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
                                   
            # get the whole image
            
            original_img_j = search_batch_data[img_index_in_batch].cpu().clone()   
            inv_transform = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            original_img_j = torch.cat([inv_transform(original_img_j[3*i:3*(i+1)]) for i in range(int(original_img_j.shape[0]/3))])
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.module.epsilon))
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size), interpolation=cv2.INTER_CUBIC)
                                
            # overlay (upsampled) self activation on original image and save the result
            rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
            rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
            heatmap = (np.float32(cv2.applyColorMap(img2uint8(rescaled_act_img_j), cv2.COLORMAP_JET)) / 255)[...,::-1]
            
            # crop out the image patch with high activation as prototype image
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]
            
            # save the prototype boundary (rectangular boundary of highly activated region)
            
            bounding_box_dict[j] = {'location': video_names[img_index_in_batch] + '_' + frame_indicies[img_index_in_batch],
                                    'bound': [proto_bound_j[0],  proto_bound_j[1],  proto_bound_j[2],  proto_bound_j[3]],
                                    'class': search_y[img_index_in_batch].item()}

            if dir_for_saving_prototypes is not None:
                if prototype_img_filename_prefix is not None:
                    with open(os.path.join(dir_for_saving_prototypes, 'bb.json'), 'w') as fp:
                        json.dump(bounding_box_dict, fp)
                    
                    # save the whole image containing the prototype as png
                    save_gif(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + 'orig_img_'+ str(j).zfill(3) + '.gif'),
                             original_img_j)

                    # save the whole image overlap with heatmap
                    save_gif(os.path.join(dir_for_saving_prototypes, prototype_self_act_filename_prefix + str(j).zfill(3) + '.gif'),
                             original_img_j, heatmap)
                    
                    # save the prototype image (highly activated region of the whole image)
                    save_gif(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + str(j).zfill(3) + '.gif'),
                             proto_img_j)
                      
    if class_specific:
        del class_to_img_index_dict
