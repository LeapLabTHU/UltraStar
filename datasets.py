import os
import re
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformation import Transformation
from tqdm import tqdm
import math
import torch.nn.functional as F

class CardiacSeqDataset(Dataset):
    def __init__(
            self,
            root,
            image_folder,
            train=True,
            transform=None,
            num_plane_to_select=1,
            timestep=1,
            sample_mode='segmental_random',
            quality_threshold=0.5,
            learnable_metric=None,
            topk=100
    ):
        suffix = 'train/' if train else 'val/'
        data_path = os.path.join(root, image_folder, suffix)
        self.transform = transform
        self.num_plane_to_select = num_plane_to_select
        self.timestep = timestep
        self.train = train
        self.sample_mode = sample_mode
        self.learnable_metric = learnable_metric
        self.topk = topk

        self.img_list = []
        self.img_quality_list = []
        self.img_exam_dict = {}

        pose_num_count_file_path = os.path.join(root, image_folder, suffix, 'check_data_available_pose_num.json')
        with open(pose_num_count_file_path, "r") as file:
            pose_num_data = json.load(file)

        idealx_prefix_file_path = os.path.join(root, image_folder, suffix, 'idealx_image_prefixes.json')
        with open(idealx_prefix_file_path, "r") as file:
            self.idealx_prefix = json.load(file)

        quality_file_path = os.path.join(root, image_folder, suffix, 'all_quality.json')
        with open(quality_file_path, "r") as file:
            self.all_quality = json.load(file)

        outlier_list = []
        exclude_plane = [str(i) for i in range(num_plane_to_select)]
        for key, value in pose_num_data.items():
            if key in exclude_plane:
                outlier_list += value

        if train:
            with open(os.path.join(root, image_folder, 'train_meta.json'), 'r') as f:
                label_meta = json.load(f)
        else:
            with open(os.path.join(root, image_folder, 'val_meta.json'), 'r') as f:
                label_meta = json.load(f)
        for key in label_meta.keys():
            for frame in label_meta[key]['frames']:
                self.img_list.append(os.path.join(root, image_folder, suffix, 'images', key, frame.replace('.txt', '.jpg')))

        # print(f'{self.train}, len of img_list: {len(self.img_list)}')
        setA = set(self.img_list)
        setB = set(outlier_list)
        setA -= setB
        self.img_list = list(setA)

        self.img_list = sorted(self.img_list[:len(self.img_list)])

        for i in range(len(self.img_list)):
            img_exam = self.img_list[i].split('/')[-2]
            img_name = self.img_list[i].split('/')[-1]
            if img_exam in self.img_exam_dict:
                self.img_exam_dict[img_exam].append(i)
            else:
                self.img_exam_dict[img_exam] = [i]

            self.img_quality_list.append(self.all_quality[img_exam][img_name.replace('.jpg', '.txt')])


    def _get_label(self, label_path):
        label = open(label_path, 'r').readline()
        if 'nan' in label:
            label = label.replace('nan', str([-0.001, -0.001, -0.001, -1., -1., -1.]))
        label = torch.tensor(eval(label))
        label = torch.cat([label[:, :3] * 1000, label[:, 3:]], dim=1)
        label = label.float()
        coeff = torch.ones_like(label)
        coeff[(label == -1.).all(dim=1)] = 0
        coeff[(label > 200).any(dim=1) | (label < -200).any(dim=1)] = 0
        mask = (label > 200).any(dim=1) | (label < -200).any(dim=1)
        label[mask] = 360
        mask = (label == -1.).all(dim=1)
        label[mask] = 360
        return label, coeff


    def __getitem__(self, index):
        img_source_path = self.img_list[index]
        img_exam = img_source_path.split('/')[-2]

        img_source = Image.open(img_source_path).convert('RGB')
        img_source_label_path = img_source_path.replace('images', 'labels').replace('.jpg', '.txt')
        img_source_label, coeff = self._get_label(img_source_label_path)
        img_source_name_id = int(img_source_label_path.split('/')[-1][:6])

        cur_possible_list = self.img_exam_dict[img_exam]
        if self.timestep > 1:
            if self.sample_mode == 'segmental_random':
                splits = np.array_split(cur_possible_list, self.timestep - 1)
                selected_index = [random.choice(split) for split in splits]
            elif self.sample_mode == 'semantic_aware':
                # Filter out semantically ambiguous candidates
                conf_thresh = getattr(self, "conf_thresh", 0.25)  

                probs_all = torch.tensor([self.img_quality_list[ind] for ind in cur_possible_list],
                                        dtype=torch.float32)  # (N,10)
                conf_all = probs_all.max(dim=1).values  # (N,)

                keep_mask = conf_all >= conf_thresh

                kept_indices = [cur_possible_list[i] for i in range(len(cur_possible_list)) if keep_mask[i].item()]
                dropped_indices = [cur_possible_list[i] for i in range(len(cur_possible_list)) if not keep_mask[i].item()]
                dropped_conf = conf_all[~keep_mask]  

                need = self.timestep - 1
                # Add back relatively confident samples if too few remain
                if len(kept_indices) < need:
                    if len(dropped_indices) > 0:
                        order = torch.argsort(dropped_conf, descending=True)  
                        for j in order.tolist():
                            kept_indices.append(dropped_indices[j])
                            if len(kept_indices) >= need:
                                break

                cur_possible_list_filtered = kept_indices

                # Segmental sampling
                splits = np.array_split(cur_possible_list_filtered, need)

                selected_index = []
                selected_samples_quality = [self.img_quality_list[index]]  

                for split in splits:
                    split = list(split)
                    if len(split) == 0:
                        continue  

                    cur_split_quality = [self.img_quality_list[ind] for ind in split]

                    cur_samples_quality = torch.tensor(selected_samples_quality, dtype=torch.float32).unsqueeze(1)  # (m,1,10)
                    cur_split_quality_t = torch.tensor(cur_split_quality, dtype=torch.float32).unsqueeze(0)        # (1,n,10)

                    cos_sim = F.cosine_similarity(cur_samples_quality, cur_split_quality_t, dim=2).sum(0)          # (n,)

                    if self.topk > cos_sim.size(0):
                        cur_index = random.randint(0, len(split) - 1)
                    else:
                        # Select from top-k least similar candidates
                        _, smallest_indices = torch.topk(-cos_sim, self.topk)  
                        cur_index = int(random.choice(smallest_indices).item())

                    chosen = split[cur_index]
                    selected_index.append(chosen)
                    selected_samples_quality.append(self.img_quality_list[chosen])

            elif self.sample_mode == 'random':
                selected_index = random.sample(cur_possible_list, self.timestep - 1)
            else:
                raise NotImplementedError
        else:
            selected_index = []

        img_seq = []
        img_seq_label = []
        img_seq_label_path = []
        img_seq_name_id = []
        act_seq = []

        for cur_target_index in selected_index:
            cur_target_img_path = self.img_list[cur_target_index]
            cur_target_label_path = cur_target_img_path.replace('images', 'labels').replace('.jpg', '.txt')
            cur_target_name_id = int(cur_target_label_path.split('/')[-1][:6])

            cur_target_img = Image.open(cur_target_img_path).convert('RGB')
            cur_target_label, _ = self._get_label(cur_target_label_path)

            img_seq.append(cur_target_img)
            img_seq_label.append(cur_target_label)
            img_seq_label_path.append(cur_target_label_path)
            img_seq_name_id.append(cur_target_name_id)

    
        img_seq.append(img_source)
        img_seq_label.append(img_source_label)
        img_seq_label_path.append(img_source_label_path)
        img_seq_name_id.append(img_source_name_id)

        if self.timestep > 1:
            ideal_prefix = torch.tensor(self.idealx_prefix[img_exam])
            lab_last = img_seq_label[-1]
            id_last = img_seq_name_id[-1]

            for i in range(self.timestep - 1):
                lab_a = img_seq_label[i]
                lab_b = lab_last
                id_a = img_seq_name_id[i]
                id_b = id_last

                distance = torch.abs(ideal_prefix - id_a) + torch.abs(ideal_prefix - id_b)
                distance_sorted_indices = torch.argsort(distance)

                mask1 = (lab_a == 360).all(dim=1)
                mask2 = (lab_b == 360).all(dim=1)
                indices = torch.nonzero(~mask1 & ~mask2).squeeze(1)
                selected_pose = None
                
                for item in distance_sorted_indices:
                    if item in indices:
                        selected_pose = item
                        break

                if selected_pose is None:
                    selected_pose = 0
    
                cur_action = Transformation.hexa_diff_inv(
                    lab_b[selected_pose].squeeze(),
                    lab_a[selected_pose].squeeze()
                )
                act_seq.append(cur_action)


        img_seq = [self.transform(img) for img in img_seq]
        img_seq = torch.stack(img_seq, dim=0)  

        if self.timestep > 1:
            act_seq = torch.from_numpy(np.stack(act_seq, axis=0).astype(np.float32))  
        else:
            act_seq = 0

        return img_seq, act_seq, img_source_label.flatten(), coeff.flatten()


    def __len__(self):
        return len(self.img_list)
    





class CardiacSeqDataset_OnlyVal(Dataset):
    def __init__(
            self,
            root,
            image_folder,
            train=True,
            transform=None,
            num_plane_to_select=1,
            timestep=1,
            sample_mode='segmental_random',
            quality_threshold=0.5,
            learnable_metric=None,
            topk=100
    ):
        suffix = 'train/' if train else 'val/'
        data_path = os.path.join(root, image_folder, suffix)
        self.transform = transform
        self.num_plane_to_select = num_plane_to_select
        self.timestep = timestep
        self.train = train
        self.sample_mode = sample_mode
        self.learnable_metric = learnable_metric
        self.topk = topk

        self.img_list = []
        self.img_quality_list = []
        self.img_exam_dict = {}

        pose_num_count_file_path = os.path.join(root, image_folder, suffix, 'check_data_available_pose_num.json')
        with open(pose_num_count_file_path, "r") as file:
            pose_num_data = json.load(file)

        idealx_prefix_file_path = os.path.join(root, image_folder, suffix, 'idealx_image_prefixes.json')
        with open(idealx_prefix_file_path, "r") as file:
            self.idealx_prefix = json.load(file)

        quality_file_path = os.path.join(root, image_folder, suffix, 'all_quality.json')
        with open(quality_file_path, "r") as file:
            self.all_quality = json.load(file)

        remove_idealx_file_path = os.path.join(root, image_folder, suffix, 'remove_ideal_plane_reformat.json')
        with open(remove_idealx_file_path, "r") as file:
            self.remove_idealx_ori = json.load(file)

        outlier_list = []
        exclude_plane = [str(i) for i in range(num_plane_to_select)]
        for key, value in pose_num_data.items():
            if key in exclude_plane:
                outlier_list += value

        if train:
            with open(os.path.join(root, image_folder, 'train_meta.json'), 'r') as f:
                label_meta = json.load(f)
        else:
            with open(os.path.join(root, image_folder, 'val_meta.json'), 'r') as f:
                label_meta = json.load(f)
        for key in label_meta.keys():
            for frame in label_meta[key]['frames']:
                self.img_list.append(os.path.join(root, image_folder, suffix, 'images', key, frame.replace('.txt', '.jpg')))

        # print(f'{self.train}, len of img_list: {len(self.img_list)}')
        setA = set(self.img_list)
        setB = set(outlier_list)
        setA -= setB
        self.img_list = list(setA)

        self.img_list = sorted(self.img_list[:len(self.img_list)])

        self.remove_idealx_reformat = {}
        for i in range(len(self.img_list)):
            img_exam = self.img_list[i].split('/')[-2]
            img_name = self.img_list[i].split('/')[-1]
            
            if img_exam not in self.remove_idealx_reformat:
                self.remove_idealx_reformat[img_exam] = {}
                for plane_ind in range(10):
                    self.remove_idealx_reformat[img_exam][str(plane_ind)] = []
                    
            if img_exam in self.img_exam_dict:
                self.img_exam_dict[img_exam].append(i)
            else:
                self.img_exam_dict[img_exam] = [i]

            self.img_quality_list.append(self.all_quality[img_exam][img_name.replace('.jpg', '.txt')])
            
            for plane_ind in range(10):
                if str(plane_ind) in self.remove_idealx_ori[img_exam]:
                    if img_name in self.remove_idealx_ori[img_exam][str(plane_ind)]:
                        self.remove_idealx_reformat[img_exam][str(plane_ind)].append(i)

    def _get_label(self, label_path):
        label = open(label_path, 'r').readline()
        if 'nan' in label:
            label = label.replace('nan', str([-0.001, -0.001, -0.001, -1., -1., -1.]))
        label = torch.tensor(eval(label))
        label = torch.cat([label[:, :3] * 1000, label[:, 3:]], dim=1)
        label = label.float()
        coeff = torch.ones_like(label)
        coeff[(label == -1.).all(dim=1)] = 0
        coeff[(label > 200).any(dim=1) | (label < -200).any(dim=1)] = 0
        mask = (label > 200).any(dim=1) | (label < -200).any(dim=1)
        label[mask] = 360
        mask = (label == -1.).all(dim=1)
        label[mask] = 360

        return label, coeff


    def __getitem__(self, index):
        img_source_path = self.img_list[index]
        img_exam = img_source_path.split('/')[-2]

        img_source = Image.open(img_source_path).convert('RGB')
        img_source_label_path = img_source_path.replace('images', 'labels').replace('.jpg', '.txt')
        img_source_label, coeff = self._get_label(img_source_label_path)
        img_source_name_id = int(img_source_label_path.split('/')[-1][:6])

        total_img_seq = []
        total_act_seq = []
        for plane_index in range(10):
            cur_possible_list = sorted(list(set(self.img_exam_dict[img_exam]) - set(self.remove_idealx_reformat[img_exam][str(plane_index)])))
            
            if self.sample_mode == 'segmental_random':
                splits = np.array_split(cur_possible_list, self.timestep - 1)
                selected_index = [random.choice(split) for split in splits]
                # selected_index = [split[len(split) // 2] for split in splits]
                    
            elif self.sample_mode == 'semantic_aware':
                # Filter out semantically ambiguous candidates
                conf_thresh = getattr(self, "conf_thresh", 0.25)  

                probs_all = torch.tensor([self.img_quality_list[ind] for ind in cur_possible_list],
                                        dtype=torch.float32)  # (N,10)
                conf_all = probs_all.max(dim=1).values  # (N,)

                keep_mask = conf_all >= conf_thresh

                kept_indices = [cur_possible_list[i] for i in range(len(cur_possible_list)) if keep_mask[i].item()]
                dropped_indices = [cur_possible_list[i] for i in range(len(cur_possible_list)) if not keep_mask[i].item()]
                dropped_conf = conf_all[~keep_mask]  

                need = self.timestep - 1
                # Add back relatively confident samples if too few remain
                if len(kept_indices) < need:
                    if len(dropped_indices) > 0:
                        order = torch.argsort(dropped_conf, descending=True)  
                        for j in order.tolist():
                            kept_indices.append(dropped_indices[j])
                            if len(kept_indices) >= need:
                                break

                cur_possible_list_filtered = kept_indices

                # Segmental sampling
                splits = np.array_split(cur_possible_list_filtered, need)

                selected_index = []
                selected_samples_quality = [self.img_quality_list[index]]  

                for split in splits:
                    split = list(split)
                    if len(split) == 0:
                        continue  

                    cur_split_quality = [self.img_quality_list[ind] for ind in split]

                    cur_samples_quality = torch.tensor(selected_samples_quality, dtype=torch.float32).unsqueeze(1)  # (m,1,10)
                    cur_split_quality_t = torch.tensor(cur_split_quality, dtype=torch.float32).unsqueeze(0)        # (1,n,10)

                    cos_sim = F.cosine_similarity(cur_samples_quality, cur_split_quality_t, dim=2).sum(0)          # (n,)

                    if self.topk > cos_sim.size(0):
                        cur_index = random.randint(0, len(split) - 1)
                    else:
                        # Select from top-k least similar candidates
                        _, smallest_indices = torch.topk(-cos_sim, self.topk)  
                        cur_index = int(random.choice(smallest_indices).item())

                    chosen = split[cur_index]
                    selected_index.append(chosen)
                    selected_samples_quality.append(self.img_quality_list[chosen])

            elif self.sample_mode == 'random':
                selected_index = random.sample(cur_possible_list, self.timestep - 1)
            else:
                raise NotImplementedError


            img_seq = []
            img_seq_label = []
            img_seq_label_path = []
            img_seq_name_id = []
            act_seq = []

            for cur_target_index in selected_index:
                cur_target_img_path = self.img_list[cur_target_index]
                cur_target_label_path = cur_target_img_path.replace('images', 'labels').replace('.jpg', '.txt')
                cur_target_name_id = int(cur_target_label_path.split('/')[-1][:6])

                cur_target_img = Image.open(cur_target_img_path).convert('RGB')
                cur_target_label, _ = self._get_label(cur_target_label_path)

                img_seq.append(cur_target_img)
                img_seq_label.append(cur_target_label)
                img_seq_label_path.append(cur_target_label_path)
                img_seq_name_id.append(cur_target_name_id)

        
            img_seq.append(img_source)
            img_seq_label.append(img_source_label)
            img_seq_label_path.append(img_source_label_path)
            img_seq_name_id.append(img_source_name_id)


            ideal_prefix = torch.tensor(self.idealx_prefix[img_exam])
            lab_last = img_seq_label[-1]
            id_last = img_seq_name_id[-1]

            for i in range(self.timestep - 1):
                lab_a = img_seq_label[i]
                lab_b = lab_last
                id_a = img_seq_name_id[i]
                id_b = id_last

                distance = torch.abs(ideal_prefix - id_a) + torch.abs(ideal_prefix - id_b)
                distance_sorted_indices = torch.argsort(distance)

                mask1 = (lab_a == 360).all(dim=1)
                mask2 = (lab_b == 360).all(dim=1)
                indices = torch.nonzero(~mask1 & ~mask2).squeeze(1)
                selected_pose = None
                
                for item in distance_sorted_indices:
                    if item in indices:
                        selected_pose = item
                        break
                    
                if selected_pose is None:
                    selected_pose = 0
                    
                cur_action = Transformation.hexa_diff_inv(
                    lab_b[selected_pose].squeeze(),
                    lab_a[selected_pose].squeeze()
                )
                act_seq.append(cur_action)


            img_seq = [self.transform(img) for img in img_seq]
            img_seq = torch.stack(img_seq, dim=0) 
            act_seq = torch.from_numpy(np.stack(act_seq, axis=0).astype(np.float32))  
            
            total_img_seq.append(img_seq)
            total_act_seq.append(act_seq)

        total_img_seq = torch.stack(total_img_seq, dim=0)
        total_act_seq = torch.stack(total_act_seq, dim=0)

        return total_img_seq, total_act_seq, img_source_label, coeff
    

    def __len__(self):
        return len(self.img_list)