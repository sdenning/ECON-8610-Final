Dependencies:
pip install ordered-set

Setup
Copy the files from each of the folders on Github into the correspond folders in the provided exam_data folder. The paths used by the code expect the original provided folder hierarchy and will not be able to find the input files if the directory structure changes. 

Command lines for part 1:
python3 part1Bayes.py -i training_data/dataset.csv -m fit
python3 part1Bayes.py -i sample_new_data/sample_new.csv -m pred
python3 part1Logit.py -i training_data/dataset.csv -m fit
python3 part1Logit.py -i sample_new_data/sample_new.csv -m pred

Command lines for part 2:
python part2.py -i training_data/dataset.csv -m fit
python part2.py -i sample_new_data/sample_new.csv -m pred

Command lines for part 3:
python3 logit.py -i training_data/dataset.csv -m fit
python3 part3.py -i sample_new_data/sample_new.csv -m pred