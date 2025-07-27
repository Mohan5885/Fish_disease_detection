Fish_disease_detection
Python Version: 3.11.4
IMPORTANT
Our program might take a while to load during the very first time you run the program. When you first run, the model hasn't been trained yet. So it will take a while for the model to train. Subsequently after you run the program for the first time, it should take faster to run as the model has already been trained.

Enhancing Fish Disease Diagnosis through Machine Learning
Through this solution, we intend to benefit 2 main entities: fish owners and lastly the fishes themselves.

We want to be able to speed up the processes of identifying diseases in fish such that it is able to seek treatment much faster and relieve the stress for fish owners to constantly take care after their fishes.

Python Packages Used
os (standard library)
numpy
streamlit
tensorflow
PIL (Python Imaging Library, installed via Pillow)
Installation
pip install numpy streamlit tensorflow Pillow -> This installs ALL the packages required for our program.

NOTE
In our program there's a specific segment which requires customisation on your end. As the path to our datasets will be different from yours, it will require some changing.
Whenever the code requires for you to specify a path, for example:
base_dir = '/Users/aathithyaj/Desktop/GitHub/fish-disease-diagnosis-py/Dataset.csv' OR healthy_img = preprocess_image("/Users/aathithyaj/Desktop/GitHub/fish-disease-diagnosis-py/healthy.png")
You should replace it with:
base_dir = '/Users/{yourusername}/path/to/fish-disease-diagnosis-py/Dataset.csv'
OR healthy_img = preprocess_image("/Users/{yourusername}/path/to/fish-disease-diagnosis-py/healthy.png")
^ Change it to the path in which our program is found on your computer. [NOTE: These are only examples. Please look through the entire program to see which part requires you to specify your own path.

Step by Step Guide:
Also since you can't see the images we attached in TextEdit, we do recommend copying and pasting the links into your browser to view the images. Thanks (The links are the GitHub links btw).
Step 1:
Open our program (ENTIRE coursework folder) in your desired code editor. image

Step 2:
Open Main.py image

Step 3:
In your terminal go to the directory and type streamlit run main.py. Click enter. image

Step 4:
You should see a screen pop up. If you don't, manually click on the links displayed in the terminal after completing Step 3. image

Step 5:
After the program opens, you should see something like this: Screenshot 2024-02-15 at 7 10 40 PM

Step 6:
Click on "Browse Files" and your finder should pop up as shown: Screenshot 2024-02-15 at 7 11 32 PM

Step 7:
Go to the Dataset.csv/validation_set and choose an image from the "healthy" or "diseased" class. The reason is because we trained our model on a different set of images (training_set), and by testing the model on a set of images different from the training images (validation_set OR testing_set), it helps to prove the accuracy of our model. Screenshot 2024-02-15 at 7 16 46 PM

Step 8 (last step):
You should be directed back to the Home screen. The results of the prediction (either diseased or not diseased) is in the red box as shown in the image. The green box represents a threshold. The threshold is the decision boundary separating different classes. For example the threshold in this case is 0.11576238, and if the image exceeds this threshold, it's classified as "not diseased". Else if it doesn't exceed this threshold, it's classified as "diseased". Screenshot 2024-02-15 at 7 18 12 PM

Thank you! ❤️
