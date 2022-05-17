import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from itertools import chain

class Basic_Encoder(nn.Module):
    def __init__(self, opt):
        super(Basic_Encoder, self).__init__()
        self.opt = opt
        self.hidden_dim = self.opt.hidden_dim
        opt.event_context_dim = self.opt.feature_dim
        opt.clip_context_dim = self.opt.feature_dim

    def forward(self, feats, vid_idx, featstamps, event_seq_idx=None, timestamps=None, vid_time_len=None):
        clip_feats, clip_mask =  self.get_clip_level_feats(feats, vid_idx, featstamps)
        event_feats = (clip_feats * clip_mask.unsqueeze(2)).sum(1) / (clip_mask.sum(1, keepdims=True) + 1e-5)
        return event_feats, clip_feats, clip_mask

    def get_clip_level_feats(self, feats, vid_idx, featstamps):
        max_att_len = max([(s[1] - s[0] + 1) for s in featstamps])
        clip_mask = feats.new(len(featstamps), max_att_len).zero_()
        clip_feats = feats.new(len(featstamps), max_att_len, feats.shape[-1]).zero_()
        for i, soi in enumerate(featstamps):
            v_idx = vid_idx[i]
            selected = feats[v_idx][soi[0]:soi[1] + 1].reshape(-1, feats.shape[-1])
            clip_feats[i, :len(selected), :] = selected
            clip_mask[i, :len(selected)] = 1
        return clip_feats, clip_mask


def extract_position_embedding(position_mat, feat_dim, wave_length=10000):
    # position_mat, [num_rois, nongt_dim, 2]
    num_rois, nongt_dim, _ = position_mat.shape
    feat_range = np.arange(0, feat_dim / 4)

    dim_mat = np.power(np.full((1,), wave_length), (4. / feat_dim) * feat_range)
    dim_mat = np.reshape(dim_mat, newshape=(1, 1, 1, -1))
    position_mat = np.expand_dims(100.0 * position_mat, axis=3)
    div_mat = np.divide(position_mat, dim_mat)
    sin_mat = np.sin(div_mat)
    cos_mat = np.cos(div_mat)
    # embedding, [num_rois, nongt_dim, 2, feat_dim/2]
    embedding = np.concatenate((sin_mat, cos_mat), axis=3)
    # embedding, [num_rois, nongt_dim, 2, feat_dim/2]
    embedding = np.reshape(embedding, newshape=(num_rois, nongt_dim, feat_dim))
    return embedding


def extract_position_matrix(bbox, nongt_dim):
    start, end = np.split(bbox, 2, axis=1)
    center = 0.5 * (start + end)
    length = (end - start).astype('float32')
    length = np.maximum(length, 1e-1)

    delta_center = np.divide(center - np.transpose(center), length)
    delta_center = delta_center
    delta_length = np.divide(np.transpose(length),length)
    delta_length = np.log(delta_length)
    delta_center = np.expand_dims(delta_center, 2)
    delta_length = np.expand_dims(delta_length, 2)
    position_matrix = np.concatenate((delta_center, delta_length), axis=2)
    return position_matrix


class TSRM_Encoder(Basic_Encoder):
    def __init__(self, opt):
        super(TSRM_Encoder, self).__init__(opt)
        self.opt = opt
        self.hidden_dim = opt.hidden_dim
        self.group = opt.group_num
        self.pre_map = nn.Sequential(nn.Linear(opt.event_context_dim, opt.hidden_dim),
                                     nn.ReLU(),
                                     nn.Dropout(0.5))

        # original code
        # self.key_map = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        # self.query_map = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        # self.value_map = nn.Linear(opt.hidden_dim, opt.hidden_dim)

        # adapted code (until horizontal line)
        self.num_heads = 8
        self.indiv_head_dim = int(opt.hidden_dim/self.num_heads)
        self.indiv_head_group = int(self.group/self.num_heads)
        # head one
        self.key_map1 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.query_map1 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.value_map1 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        # head two
        self.key_map2 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.query_map2 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.value_map2 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        # # head three
        self.key_map3 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.query_map3 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.value_map3 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        # head four
        self.key_map4 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.query_map4 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.value_map4 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        # head five
        self.key_map5 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.query_map5 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.value_map5 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        # head six
        self.key_map6 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.query_map6 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.value_map6 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        # head seven
        self.key_map7 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.query_map7 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.value_map7 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        # head eight
        self.key_map8 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.query_map8 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        self.value_map8 = nn.Linear(self.indiv_head_dim, self.indiv_head_dim)
        # ______________________

        self.drop = nn.Dropout(0.5)
        self.use_posit_branch = opt.use_posit_branch
        if self.use_posit_branch:
            self.posit_hidden_dim = opt.hidden_dim
            self.pair_pos_fc1 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
            self.pair_pos_fc2 = nn.Linear(opt.hidden_dim, opt.group_num)

        opt.event_context_dim = 2 * self.hidden_dim + 100
        opt.clip_context_dim = self.opt.feature_dim

    def forward(self, feats, vid_idx, featstamps, event_seq_idx, timestamps, vid_time_len):
        clip_feats, clip_mask = self.get_clip_level_feats(feats, vid_idx, featstamps)
        event_feats = (clip_feats * clip_mask.unsqueeze(2)).sum(1) / (clip_mask.sum(1, keepdims=True) + 1e-5)
        event_feats = self.pre_map(event_feats)

        batch_size = len(event_seq_idx)
        event_num = sum([_.shape[0] * _.shape[1] for _ in event_seq_idx])
        event_feats_expand = feats.new_zeros(event_num, event_feats.shape[-1])
        mask = feats.new_zeros(event_num, event_num)
        timestamp_expand = []
        vid_time_len_expand = []

        total_idx = 0
        for i in range(batch_size):
            vid_start_idx = (vid_idx < i).sum().item()
            for j in range(event_seq_idx[i].shape[0]):
                event_idx = vid_start_idx + event_seq_idx[i][j]
                event_feats_expand[total_idx: total_idx + len(event_idx)] = event_feats[event_idx]
                mask[total_idx: total_idx + len(event_idx), total_idx: total_idx + len(event_idx)] = 1
                timestamp_expand.extend([timestamps[ii] for ii in event_idx])
                vid_time_len_expand.extend([vid_time_len[i].item() for jj in event_idx])
                total_idx += len(event_idx)
        mask = mask.unsqueeze(0) # no idea what this does, why is there a mask

        # original code

        # # calculate Wq
        # query_mat = self.query_map(event_feats_expand).reshape(event_num, self.group,
        #                                              int(self.hidden_dim / self.group)).transpose(0, 1)
        # # calculate Wk
        # key = self.key_map(event_feats_expand).reshape(event_num, self.group, int(self.hidden_dim / self.group)).transpose(0,
        #                                                                                                              1)
        # # sim <-- scaled dot product: Wq*Wk                                                                                                             
        # cos_sim = torch.bmm(query_mat, key.transpose(1, 2))  # [self.group, event_num, event_num]
        # cos_sim = cos_sim / math.sqrt(self.hidden_dim / self.group)
        # sim = cos_sim

        # adoptated code (until horizontal line)
        # HEAD ONE
        # calculate Wq1
        # print('event-feats-expand-shape ', event_feats_expand.shape)
        # print('event-feats-expand-shape part one ', event_feats_expand[:,:self.indiv_head_dim].shape)
        # print('event-num ', event_num)
        # print('int(self.indiv_head_dim / self.group) ', int(self.indiv_head_dim / self.group))
        # print('event_feats_expand[:self.indiv_head_dim]).reshape(event_num, self.group,int(self.indiv_head_dim / self.group) shape ', event_feats_expand[:,:self.indiv_head_dim].reshape(event_num, self.group, int(self.indiv_head_dim / self.group)).shape)

        # query_mat1 dim: 2x256 reshape to 2x16x16
        
        query_mat1 = self.query_map1(event_feats_expand[:,:self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # print('query-mat-1-shape ', query_mat1.shape)
        # calculate Wk1
        key1 = self.key_map1(event_feats_expand[:,:self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # print('key-mat-1-shape ', key1.shape)                                                                                                             
        # sim <-- scaled dot product: Wq*Wk  
        #cos_sim1 = query_mat1.add(key1)                                                                                                           
        cos_sim1 = torch.bmm(query_mat1, key1.transpose(1, 2))  # [self.indiv_head_group, event_num, event_num]
        cos_sim1 = cos_sim1 / math.sqrt(self.indiv_head_dim/ self.indiv_head_group)
        #sim1 = cos_sim1

        # HEAD TWO
        # calculate Wq2
        
        query_mat2 = self.query_map2(event_feats_expand[:,self.indiv_head_dim:2*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group,int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)

        # calculate Wk2
        key2 = self.key_map2(event_feats_expand[:,self.indiv_head_dim:2*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)

        # sim <-- scaled dot product: Wq*Wk     
        # cos_sim2 = query_mat2.add(key2)                                                                                                           
        cos_sim2 = torch.bmm(query_mat2, key2.transpose(1, 2))  # [self.indiv_head_group, event_num, event_num]
        cos_sim2 = cos_sim2 / math.sqrt(self.indiv_head_dim/ self.indiv_head_group)
        #sim1 = cos_sim1

        # HEAD THREE
        # calculate Wq3
        query_mat3 = self.query_map3(event_feats_expand[:,2*self.indiv_head_dim:3*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group,int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # calculate Wk3
        key3 = self.key_map3(event_feats_expand[:,2*self.indiv_head_dim:3*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # scaled for product                                                                                                      
        cos_sim3 = torch.bmm(query_mat3, key3.transpose(1, 2))  # [self.indiv_head_group, event_num, event_num]
        cos_sim3 = cos_sim3 / math.sqrt(self.indiv_head_dim/ self.indiv_head_group)
        # HEAD FOUR
        # calculate Wq4
        query_mat4 = self.query_map4(event_feats_expand[:,3*self.indiv_head_dim:4*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group,int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # calculate Wk4
        key4 = self.key_map4(event_feats_expand[:,3*self.indiv_head_dim:4*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # scaled for product                                                                                                      
        cos_sim4 = torch.bmm(query_mat4, key4.transpose(1, 2))  # [self.indiv_head_group, event_num, event_num]
        cos_sim4 = cos_sim4 / math.sqrt(self.indiv_head_dim/ self.indiv_head_group)
        # HEAD FIVE
        # calculate Wq5
        query_mat5 = self.query_map5(event_feats_expand[:,4*self.indiv_head_dim:5*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group,int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # calculate Wk5
        key5 = self.key_map5(event_feats_expand[:,4*self.indiv_head_dim:5*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # scaled for product                                                                                                      
        cos_sim5 = torch.bmm(query_mat5, key5.transpose(1, 2))  # [self.indiv_head_group, event_num, event_num]
        cos_sim5 = cos_sim5 / math.sqrt(self.indiv_head_dim/ self.indiv_head_group)
        # HEAD SIX
        # calculate Wq6
        query_mat6 = self.query_map6(event_feats_expand[:,5*self.indiv_head_dim:6*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group,int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # calculate Wk6
        key6 = self.key_map6(event_feats_expand[:,5*self.indiv_head_dim:6*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # scaled for product                                                                                                      
        cos_sim6 = torch.bmm(query_mat6, key6.transpose(1, 2))  # [self.indiv_head_group, event_num, event_num]
        cos_sim6 = cos_sim6 / math.sqrt(self.indiv_head_dim/ self.indiv_head_group)
        # HEAD SEVEN
        # calculate Wq7
        query_mat7 = self.query_map7(event_feats_expand[:,6*self.indiv_head_dim:7*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group,int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # calculate Wk7
        key7 = self.key_map7(event_feats_expand[:,6*self.indiv_head_dim:7*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # scaled for product                                                                                                      
        cos_sim7 = torch.bmm(query_mat7, key7.transpose(1, 2))  # [self.indiv_head_group, event_num, event_num]
        cos_sim7 = cos_sim7 / math.sqrt(self.indiv_head_dim/ self.indiv_head_group)
        # HEAD EIGHT
        # calculate Wq8
        query_mat8 = self.query_map8(event_feats_expand[:,7*self.indiv_head_dim:8*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group,int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # calculate Wk8
        key8 = self.key_map8(event_feats_expand[:,7*self.indiv_head_dim:8*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, int(self.indiv_head_dim / self.indiv_head_group)).transpose(0, 1)
        # scaled for product                                                                                                      
        cos_sim8 = torch.bmm(query_mat8, key8.transpose(1, 2))  # [self.indiv_head_group, event_num, event_num]
        cos_sim8 = cos_sim8 / math.sqrt(self.indiv_head_dim/ self.indiv_head_group)
        
        

        # CONCATENATE ALL HEADS (scaled dot products)
        # print('cos sim1 shape ', cos_sim1.shape)
        # print('cos sim2 shape ', cos_sim2.shape)
        # cos_sim = torch.cat((cos_sim1, cos_sim2), 0)
        # print('cos sim shape ', cos_sim.shape)
        #__________________________________________

        # sim <-- semantical_scaled_dot_prod + temporal_weights
        if self.use_posit_branch:
            pos_matrix = extract_position_matrix(np.array(timestamp_expand), event_num)
            pos_feats = extract_position_embedding(pos_matrix,
                                                   self.posit_hidden_dim)  # [event_num, event_num, self.posit_hidden_dim]
            pos_feats = feats.new_tensor(pos_feats).reshape(-1, self.posit_hidden_dim)
            pos_sim = self.pair_pos_fc2(torch.tanh(self.pair_pos_fc1(pos_feats))).reshape(event_num, event_num, self.group)
            pos_sim = pos_sim.permute(2, 0, 1)
            # print('pos sim shape ', pos_sim.shape)
            # sim = cos_sim + pos_sim
            cos_sim1 += pos_sim[:self.indiv_head_group,:,:]
            cos_sim2 += pos_sim[self.indiv_head_group:2*self.indiv_head_group,:,:]
            cos_sim3 += pos_sim[2*self.indiv_head_group:3*self.indiv_head_group,:,:]
            cos_sim4 += pos_sim[3*self.indiv_head_group:4*self.indiv_head_group,:,:]
            cos_sim5 += pos_sim[4*self.indiv_head_group:5*self.indiv_head_group,:,:]
            cos_sim6 += pos_sim[5*self.indiv_head_group:6*self.indiv_head_group,:,:]
            cos_sim7 += pos_sim[6*self.indiv_head_group:7*self.indiv_head_group,:,:]
            cos_sim8 += pos_sim[7*self.indiv_head_group:8*self.indiv_head_group,:,:]
            


        # softmax
        cos_sim1 = F.softmax(cos_sim1, dim=2)
        cos_sim2 = F.softmax(cos_sim2, dim=2)
        cos_sim3 = F.softmax(cos_sim3, dim=2)
        cos_sim4 = F.softmax(cos_sim4, dim=2)
        cos_sim5 = F.softmax(cos_sim5, dim=2)
        cos_sim6 = F.softmax(cos_sim6, dim=2)
        cos_sim7 = F.softmax(cos_sim7, dim=2)
        cos_sim8 = F.softmax(cos_sim8, dim=2)
        # sim = F.softmax(sim, dim=2)

        # concatenate
        sim = torch.cat((cos_sim1, cos_sim2, cos_sim3, cos_sim4, cos_sim5, cos_sim6, cos_sim7, cos_sim8), 0)
        # print('sim shape ', sim.shape)
        # mask on sim
        sim = (sim * mask) / (1e-5 + torch.sum(sim * mask, dim=2, keepdim=True))

        # orginal code
        # # calculate Wv
        # value = self.value_map(event_feats_expand).reshape(event_num, self.group, -1).transpose(0, 1)

        # adapted code (until horizontal line)
        # calculate Wv1
        value1 = self.value_map1(event_feats_expand[:,:self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, -1).transpose(0, 1)
        # calculate Wv2
        value2 = self.value_map2(event_feats_expand[:,self.indiv_head_dim:2*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, -1).transpose(0, 1)
        # calculate Wv3
        value3 = self.value_map3(event_feats_expand[:,2*self.indiv_head_dim:3*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, -1).transpose(0, 1)
        # calculate Wv4
        value4 = self.value_map4(event_feats_expand[:,3*self.indiv_head_dim:4*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, -1).transpose(0, 1)
        # calculate Wv5
        value5 = self.value_map5(event_feats_expand[:,4*self.indiv_head_dim:5*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, -1).transpose(0, 1)
        # calculate Wv6
        value6 = self.value_map6(event_feats_expand[:,5*self.indiv_head_dim:6*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, -1).transpose(0, 1)
        # calculate Wv7
        value7 = self.value_map7(event_feats_expand[:,6*self.indiv_head_dim:7*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, -1).transpose(0, 1)
        # calculate Wv8
        value8 = self.value_map8(event_feats_expand[:,7*self.indiv_head_dim:8*self.indiv_head_dim]).reshape(event_num, self.indiv_head_group, -1).transpose(0, 1)

        # TODO concatenate all the value parts here
        value = torch.cat((value1, value2, value3, value4, value5, value6, value7, value8), 0) # or axis one
        #______________________________________

        # weighted sum of sim and value
        event_ctx = torch.bmm(sim, value)
        event_ctx = event_ctx.transpose(0, 1).reshape(event_num, -1)
        event_ctx = torch.cat((F.relu(event_ctx), event_feats_expand), 1)
        event_ctx = self.drop(event_ctx)

        # positional feature vector for each event
        pos_feats = feats.new(len(timestamp_expand), 100).zero_()
        for i in range(len(timestamp_expand)):
            s, e = timestamp_expand[i]
            duration = vid_time_len_expand[i]
            s, e = min(int(s / duration * 99), 99), min(int(e / duration * 99), 99)
            pos_feats[i, s: e + 1] = 1
        event_ctx = torch.cat([event_ctx, pos_feats], dim=1)

        return event_ctx, clip_feats, clip_mask


if __name__ == '__main__':
    pass
