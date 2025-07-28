import glob
import os,cv2
import numpy as np
 
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
import lmdb

#Resize original h2o imgs to (480,270)
def resize_imgs_to_480_270(h2o_root='./data/H2O'):
    def resize_seq(cdir):
        out_dir=os.path.join(cdir,'rgb480_270')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        list_imgs=glob.glob(os.path.join(cdir,'rgb','*.png'))
        for im_id in tqdm(range(0,len(list_imgs))):
            path_cimg=os.path.join(cdir,'rgb','{:06d}.png'.format(im_id))
            cimg=cv2.imread(path_cimg)
            cimg=cv2.resize(cimg,(480,270))
            cv2.imwrite(os.path.join(cdir,'rgb480_270','{:06d}.png'.format(im_id)),cimg)

    list_dirs=glob.glob(os.path.join(h2o_root,'/*/*/*/cam4/'))
    print(list_dirs)
    for x in tqdm(list_dirs):
        resize_seq(x) 


#Use lmdb to save image, and facilitate training
def convert_dataset_split_to_lmdb(h2o_root='./data/H2O/', split='train'):
    from datasets.scripts.process_h2o import read_split
    input_res = (480, 270)

    num_imgs = 0
    all_imgs = []
    seq_names = read_split(split)
    for seq in seq_names:
        sub_seqs = glob.glob(os.path.join(h2o_root, seq, '*/cam4/'))
        for sub_seq in sub_seqs:
            list_imgs=glob.glob(os.path.join(sub_seq, 'rgb480_270', '*.png'))
            list_imgs.sort()
            all_imgs.append((sub_seq, list_imgs))
            num_imgs = num_imgs + len(list_imgs)
    
    print(all_imgs[0][1])
    data_size_per_img = np.array(Image.open(all_imgs[0][1][0]).convert("RGB")).nbytes 
    data_size = data_size_per_img * num_imgs

    dir_lmdb=os.path.join('./data/motion/h2o','lmdb_imgs', split)
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)

    env = lmdb.open(dir_lmdb,map_size=data_size*10,max_dbs=1000)
    pre_seq_tag=''
    commit_interval=100
    for i, (seq, imgs) in tqdm(enumerate(all_imgs)):
        if i > 0:
            txn.commit()
        key_seq = seq.split(h2o_root)[-1] + 'rgb480_270/'
        subdb=env.open_db(key_seq.encode('ascii'))
        txn=env.begin(db=subdb,write=True)    
        print("Processing Seq: ", key_seq)
        
        for j, cur_img_path in enumerate(imgs):
            key_byte = os.path.basename(cur_img_path).encode('ascii')
            data = np.array(Image.open(cur_img_path).convert("RGB"))
            txn.put(key_byte,data)

            if (j+1) % commit_interval==0:
                txn.commit()
                txn=env.begin(db=subdb,write=True)

    txn.commit()
    env.close()

if __name__ == '__main__':
    # resize_imgs_to_480_270()
    convert_dataset_split_to_lmdb(split='train')
    convert_dataset_split_to_lmdb(split='val')