import os
#By standard, the test images to test the model against are saved in the data_directory
data_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)),"input")
#The main program is saved in the same directory as config
code_directory = os.path.dirname(os.path.abspath(__file__))
#The training model is saved in the training directory, the same directory as config
training_directory = os.path.dirname(os.path.abspath(__file__))