output_file = "output_mapping.txt"  

# preprocessing text file
with open("mapping_dict.txt", "r") as file, open(output_file, 'w') as outfile:
    lines = file.readlines()
    for line in lines: 
        # insert commas between the single spaces
        # updated_line = line.replace(' ', ': ')
        # updated_line = updated_line.strip() + ",\n"
        parts = line.split()
        updated_line = f'{parts[0]}: "{parts[1]}",\n' 
        outfile.write(updated_line)
