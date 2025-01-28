import torch
import pandas as pd  

def keyword_per_img(similarity_matrix, list_images, keywords_class):
        max_value = torch.argmax(similarity_matrix, dim=1)
        keyword_list = [keywords_class[i] for i in max_value]
        dict_img_keywords = {'Images': list_images, 'Bias keywords': keyword_list} 
        keywords_df = pd.DataFrame(dict_img_keywords)   
        return keywords_df
