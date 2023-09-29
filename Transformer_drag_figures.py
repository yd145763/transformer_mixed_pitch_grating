# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 18:44:29 2023

@author: limyu
"""

import pandas as pd

path = "C:\\Users\\limyu\\Google Drive\\Machine Learning photonics\\transformer_codes\\"
df_1b1_hor_far = pd.read_csv(path+"onebyone_farfield_horizontal\\full_result.csv")
df_1b1_ver_far = pd.read_csv(path+"onebyone_farfield_vertical\\full_result.csv")
df_1s_hor_far = pd.read_csv(path+"oneshot_farfield_horizontal\\full_result.csv")
df_1s_ver_far = pd.read_csv(path+"oneshot_farfield_vertical\\full_result.csv")

df_1b1_hor_near = pd.read_csv(path+"onebyone_reactiveregion_horizontal\\full_result.csv")
df_1b1_ver_near = pd.read_csv(path+"onebyone_reactiveregion_vertical\\full_result.csv")
df_1s_hor_near = pd.read_csv(path+"oneshot_reactiveregion_horizontal\\full_result.csv")
df_1s_ver_near = pd.read_csv(path+"oneshot_reactiveregion_vertical\\full_result.csv")

folder_dict = {
    'onebyone_farfield_horizontal\\': df_1b1_hor_far,
    'onebyone_farfield_vertical\\': df_1b1_ver_far,
    'oneshot_farfield_horizontal\\': df_1s_hor_far,
    'oneshot_farfield_vertical\\': df_1s_ver_far,
    'onebyone_reactiveregion_horizontal\\': df_1b1_hor_near,
    'onebyone_reactiveregion_vertical\\': df_1b1_ver_near,
    'oneshot_reactiveregion_horizontal\\': df_1s_hor_near,
    'oneshot_reactiveregion_vertical\\': df_1s_ver_near,
}

folder_name = 'onebyone_reactiveregion_horizontal\\'

method = 'onebyone'

indicator = 'training validation bear'

imperfect_block = 3
imperfect_seq = 10

if method == 'onebyone':

    if indicator == 'prediction actual meow':
    
        df = folder_dict[folder_name]
        
        df_lowest = df[df['ape_label'] == min(df['ape_label'])]
        print(df[df['ape_label'] == min(df['ape_label'])].T)
        
        filtered_df = df[(df['head_size_list'] == int(df_lowest['head_size_list'])) & (df['num_head_list'] == int(df_lowest['num_head_list'])) & (df['ff_dim_list'] == int(df_lowest['ff_dim_list']))]
        from PIL import Image
        seq_str = str(int(df_lowest['sequence_length_list']))
        headsize_str = str(int(df_lowest['head_size_list']))
        numheads_str = str(int(df_lowest['num_head_list']))
        ffdim_str = str(int(df_lowest['ff_dim_list']))
        block_str = str(int(df_lowest['num_transformer_blocks_list']))
        
        # Open an image file
        image_path = path+folder_name+"PA_seq"+seq_str+"_headsize"+headsize_str+"_numheads"+numheads_str+"_ffdim"+ffdim_str+"_blocks"+block_str+"_mix8011um_2012um.jpg" 
        image = Image.open(image_path)
        
        
        # You can also display the image using the default image viewer
        image.show()
    
    if indicator == 'training validation meow':
    
        df = folder_dict[folder_name]
        
        df_lowest = df[df['ape_label'] == min(df['ape_label'])]
        print(df[df['ape_label'] == min(df['ape_label'])].T)
        
        filtered_df = df[(df['head_size_list'] == int(df_lowest['head_size_list'])) & (df['num_head_list'] == int(df_lowest['num_head_list'])) & (df['ff_dim_list'] == int(df_lowest['ff_dim_list']))]
        from PIL import Image
        seq_str = str(int(df_lowest['sequence_length_list']))
        headsize_str = str(int(df_lowest['head_size_list']))
        numheads_str = str(int(df_lowest['num_head_list']))
        ffdim_str = str(int(df_lowest['ff_dim_list']))
        block_str = str(int(df_lowest['num_transformer_blocks_list']))
        
        # Open an image file
        image_path = path+folder_name+"TV_seq"+seq_str+"_headsize"+headsize_str+"_numheads"+numheads_str+"_ffdim"+ffdim_str+"_blocks"+block_str+"_mix011um_10012um.jpg" 
        image = Image.open(image_path)
        
        
        # You can also display the image using the default image viewer
        image.show()
        
    if indicator == 'prediction actual bear':
        df = folder_dict[folder_name]
      
        df_lowest = df[df['ape_label'] == min(df['ape_label'])]
        print(df[df['ape_label'] == min(df['ape_label'])].T)
        
        filtered_df = df[(df['head_size_list'] == int(df_lowest['head_size_list'])) & (df['num_head_list'] == int(df_lowest['num_head_list'])) & (df['ff_dim_list'] == int(df_lowest['ff_dim_list']))]
       
    
        from PIL import Image
        seq_str = str(imperfect_seq)
        headsize_str = str(int(df_lowest['head_size_list']))
        numheads_str = str(int(df_lowest['num_head_list']))
        ffdim_str = str(int(df_lowest['ff_dim_list']))
        block_str = str(imperfect_block)
        
        # Open an image file
        image_path = path+folder_name+"PA_seq"+seq_str+"_headsize"+headsize_str+"_numheads"+numheads_str+"_ffdim"+ffdim_str+"_blocks"+block_str+"_mix8011um_2012um.jpg" 
        image = Image.open(image_path)
        
        
        # You can also display the image using the default image viewer
        image.show()
    
    if indicator == 'training validation bear':
        df = folder_dict[folder_name]
     
        df_lowest = df[df['ape_label'] == min(df['ape_label'])]
        print(df[df['ape_label'] == min(df['ape_label'])].T)
        
        filtered_df = df[(df['head_size_list'] == int(df_lowest['head_size_list'])) & (df['num_head_list'] == int(df_lowest['num_head_list'])) & (df['ff_dim_list'] == int(df_lowest['ff_dim_list']))]
        
    
        from PIL import Image
        seq_str = str(imperfect_seq)
        headsize_str = str(int(df_lowest['head_size_list']))
        numheads_str = str(int(df_lowest['num_head_list']))
        ffdim_str = str(int(df_lowest['ff_dim_list']))
        block_str = str(imperfect_block)
        
        # Open an image file
        image_path = path+folder_name+"TV_seq"+seq_str+"_headsize"+headsize_str+"_numheads"+numheads_str+"_ffdim"+ffdim_str+"_blocks"+block_str+"_mix011um_10012um.jpg" 
        image = Image.open(image_path)
        
        
        # You can also display the image using the default image viewer
        image.show()

if method == 'oneshot':

    if indicator == 'prediction actual meow':
    
        df = folder_dict[folder_name]
        
        df_lowest = df[df['ape_label'] == min(df['ape_label'])]
        print(df[df['ape_label'] == min(df['ape_label'])].T)
        
        filtered_df = df[(df['head_size_list'] == int(df_lowest['head_size_list'])) & (df['num_head_list'] == int(df_lowest['num_head_list'])) & (df['ff_dim_list'] == int(df_lowest['ff_dim_list']))]
        from PIL import Image
        seq_str = str(int(df_lowest['sequence_length_list']))
        headsize_str = str(int(df_lowest['head_size_list']))
        numheads_str = str(int(df_lowest['num_head_list']))
        ffdim_str = str(int(df_lowest['ff_dim_list']))
        block_str = str(int(df_lowest['num_transformer_blocks_list']))
        
        # Open an image file
        image_path = path+folder_name+"PA_seq"+seq_str+"_headsize"+headsize_str+"_numheads"+numheads_str+"_ffdim"+ffdim_str+"_blocks"+block_str+".jpg" 
        image = Image.open(image_path)
        
        
        # You can also display the image using the default image viewer
        image.show()
    
    if indicator == 'training validation meow':
    
        df = folder_dict[folder_name]
        
        df_lowest = df[df['ape_label'] == min(df['ape_label'])]
        print(df[df['ape_label'] == min(df['ape_label'])].T)
        
        filtered_df = df[(df['head_size_list'] == int(df_lowest['head_size_list'])) & (df['num_head_list'] == int(df_lowest['num_head_list'])) & (df['ff_dim_list'] == int(df_lowest['ff_dim_list']))]
        from PIL import Image
        seq_str = str(int(df_lowest['sequence_length_list']))
        headsize_str = str(int(df_lowest['head_size_list']))
        numheads_str = str(int(df_lowest['num_head_list']))
        ffdim_str = str(int(df_lowest['ff_dim_list']))
        block_str = str(int(df_lowest['num_transformer_blocks_list']))
        
        # Open an image file
        image_path = path+folder_name+"TV_seq"+seq_str+"_headsize"+headsize_str+"_numheads"+numheads_str+"_ffdim"+ffdim_str+"_blocks"+block_str+".jpg" 
        image = Image.open(image_path)
        
        
        # You can also display the image using the default image viewer
        image.show()
        
    if indicator == 'prediction actual bear':
        df = folder_dict[folder_name]
      
        df_lowest = df[df['ape_label'] == min(df['ape_label'])]
        print(df[df['ape_label'] == min(df['ape_label'])].T)
        
        filtered_df = df[(df['head_size_list'] == int(df_lowest['head_size_list'])) & (df['num_head_list'] == int(df_lowest['num_head_list'])) & (df['ff_dim_list'] == int(df_lowest['ff_dim_list']))]
       
    
        from PIL import Image
        seq_str = str(imperfect_seq)
        headsize_str = str(int(df_lowest['head_size_list']))
        numheads_str = str(int(df_lowest['num_head_list']))
        ffdim_str = str(int(df_lowest['ff_dim_list']))
        block_str = str(imperfect_block)
        
        # Open an image file
        image_path = path+folder_name+"PA_seq"+seq_str+"_headsize"+headsize_str+"_numheads"+numheads_str+"_ffdim"+ffdim_str+"_blocks"+block_str+".jpg" 
        image = Image.open(image_path)
        
        
        # You can also display the image using the default image viewer
        image.show()
    
    if indicator == 'training validation bear':
        df = folder_dict[folder_name]
     
        df_lowest = df[df['ape_label'] == min(df['ape_label'])]
        print(df[df['ape_label'] == min(df['ape_label'])].T)
        
        filtered_df = df[(df['head_size_list'] == int(df_lowest['head_size_list'])) & (df['num_head_list'] == int(df_lowest['num_head_list'])) & (df['ff_dim_list'] == int(df_lowest['ff_dim_list']))]
        
    
        from PIL import Image
        seq_str = str(imperfect_seq)
        headsize_str = str(int(df_lowest['head_size_list']))
        numheads_str = str(int(df_lowest['num_head_list']))
        ffdim_str = str(int(df_lowest['ff_dim_list']))
        block_str = str(imperfect_block)
        
        # Open an image file
        image_path = path+folder_name+"TV_seq"+seq_str+"_headsize"+headsize_str+"_numheads"+numheads_str+"_ffdim"+ffdim_str+"_blocks"+block_str+".jpg" 
        image = Image.open(image_path)
        
        
        # You can also display the image using the default image viewer
        image.show()

import seaborn as sns
import matplotlib.pyplot as plt

backcandlesS = 5,10,20

head_sizeS=16,32,64
num_headsS=2,3,4
ff_dimS=2,3,4
num_transformer_blocksS=2,3,4


mat = filtered_df.pivot('num_transformer_blocks_list', 'sequence_length_list', 'ape_label')
mat_list = mat.values.tolist()

plt.figure(figsize=(6, 4))
ax = sns.heatmap(mat_list, annot=True, cmap='viridis', fmt=".1f", annot_kws={"size": 14, "weight": "bold"})

ax.set_xticklabels(backcandlesS, fontweight="bold", fontsize = 14)
ax.set_yticklabels(num_transformer_blocksS, fontweight="bold", fontsize = 14)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Average\nPercentage Error (%)", fontweight="bold")
font = {'color': 'black', 'weight': 'bold', 'size': 14}
ax.set_ylabel("Number of Blocks", fontdict=font)
ax.set_xlabel("Sequence Length", fontdict=font)
ax.tick_params(axis='both', labelsize=14, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(30)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
plt.show()
plt.close()

if method == 'onebyone':


    mat = filtered_df.pivot('num_transformer_blocks_list', 'sequence_length_list', 'train_test_ape r = 0')
    mat_list = mat.values.tolist()
    
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(mat_list, annot=True, cmap='viridis', fmt=".3f", annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_xticklabels(backcandlesS, fontweight="bold", fontsize = 14)
    ax.set_yticklabels(num_transformer_blocksS, fontweight="bold", fontsize = 14)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")
    cbar.ax.set_title("Average\nPercentage Error (%)", fontweight="bold")
    font = {'color': 'black', 'weight': 'bold', 'size': 14}
    ax.set_ylabel("Number of Blocks", fontdict=font)
    ax.set_xlabel("Sequence Length", fontdict=font)
    ax.tick_params(axis='both', labelsize=14, weight='bold')
    for i, text in enumerate(ax.texts):
        text.set_fontsize(30)
    for i, text in enumerate(ax.texts):
        text.set_fontweight('bold')
    plt.show()
    plt.close()

if method =='oneshot':
    filtered_df.columns

    mat = filtered_df.pivot('num_transformer_blocks_list', 'sequence_length_list', 'train_test_ape')
    mat_list = mat.values.tolist()
    
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(mat_list, annot=True, cmap='viridis', fmt=".3f", annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_xticklabels(backcandlesS, fontweight="bold", fontsize = 14)
    ax.set_yticklabels(num_transformer_blocksS, fontweight="bold", fontsize = 14)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")
    cbar.ax.set_title("Average\nPercentage Error (%)", fontweight="bold")
    font = {'color': 'black', 'weight': 'bold', 'size': 14}
    ax.set_ylabel("Number of Blocks", fontdict=font)
    ax.set_xlabel("Sequence Length", fontdict=font)
    ax.tick_params(axis='both', labelsize=14, weight='bold')
    for i, text in enumerate(ax.texts):
        text.set_fontsize(30)
    for i, text in enumerate(ax.texts):
        text.set_fontweight('bold')
    plt.show()
    plt.close()
    