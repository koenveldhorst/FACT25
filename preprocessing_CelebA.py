import pandas as pd 
# make a new txt file
output_file = "output.txt"  

# preprocessing text file
with open("list_eval_partition.txt", "r") as file, open(output_file, 'w') as outfile:
    lines = file.readlines()
    for line in lines: 
        # insert commas between the single spaces
        updated_line = line.replace(' ', ', ')
        outfile.write(updated_line)

dataframe1 = pd.read_csv("output.txt", names=['image_id', 'partition'])
# convert file to csv file
# dataframe1.to_csv('data/celebA/data/list_eval_partition.csv', index = None) 
# dataframe1

# I already have removed the first row manually and added image_id to the column names in text file 
# make a new txt file
output_file = "output2.txt"  

# preprocessing text file
with open("list_attr_celeba.txt", "r") as file, open(output_file, 'w') as outfile:
    lines = file.readlines()
    for line in lines:
        # line = re.sub(r'\s+', ' ', line.strip())
        # replace double spaces with single spaces and insert commas between the single spaces
        updated_line = line.replace("  ", " ")
        updated_line = updated_line.replace(" ", ",")
        outfile.write(updated_line)


dataframe2 = pd.read_csv("output2.txt")
# convert file to csv file
# dataframe2.to_csv('data/celebA/data/list_attr_celeba.csv', index = None) 
# dataframe2