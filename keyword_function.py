import torch
import pandas as pd  

def keyword_per_img(similarity_matrix, list_images, pred, actual, caption, keywords_class):
        # max_value = torch.argmax(similarity_matrix, dim=1)
        max_clip_values, max_indices = torch.max(similarity_matrix, dim=0)
        best_match_images = [list_images[i] for i in max_indices]
        predicted_labels = [pred[i] for i in max_indices]
        actual_labels = [actual[i] for i in max_indices]
        caption_images = [caption[i] for i in max_indices]
        # keyword_list = [keywords_class[i] for i in max_value]
        dict_img_keywords = {'Bias keywords': keywords_class, 'Images': best_match_images, 'Clip values': max_clip_values.numpy(), 
                             'Predictions': predicted_labels, 'Actual': actual_labels, 'Captions': caption_images} 
        keywords_df = pd.DataFrame(dict_img_keywords)   
        return keywords_df
