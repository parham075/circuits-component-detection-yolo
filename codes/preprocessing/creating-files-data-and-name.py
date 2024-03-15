full_path_to_images = r'C:\Users\ASUS\Desktop\yolov4\sdp_classifier\dataset'
saved_weights = r'C:\Users\ASUS\Desktop\yolov4\sdp_classifier\Outputs\backup'

# """
# End of:
# Setting up full path to directory with labelled images
# """
# os.chdir(full_path_to_images)

# """
# Start of:
# Creating file classes.names
# """

# # Defining counter for classes
c = 0

# # Creating file classes.names from existing one classes.txt

with open(full_path_to_images + '/' + 'classes.names', 'w') as names, \
     open(full_path_to_images + '/' + 'classes.txt', 'r') as txt:

    # Going through all lines in txt file and writing them into names file
    for line in txt:
        names.write(line)  # Copying all info from file txt to names

        # Increasing counter
        c += 1

# """
# End of:
# Creating file classes.names
# """


# """
# Start of:
# Creating file labelled_data.data
# """

# # Creating file labelled_data.data

with open(full_path_to_images + '/' + 'labelled_dataa.data', 'w') as data:
    # Writing needed 5 lines
    # Number of classes
    # By using '\n' we move to the next line
    data.write('classes = ' + str(c) + '\n')

   # Location of the train.txt file
    data.write('train = ' + full_path_to_images + '/' + 'train.txt' + '\n')

    # Location of the test.txt file
    data.write('valid = ' + full_path_to_images + '/' + 'test.txt' + '\n')

    # Location of the classes.names file
    data.write('names = ' + full_path_to_images + '/' + 'classes.names' + '\n')

    # Location where to save weights
    data.write('backup = ' + saved_weights)

