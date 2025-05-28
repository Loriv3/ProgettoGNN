Please check the [Competition Website](https://sites.google.com/view/hackathongraphnoisylabels/home) for the set of Rules and Instructions on how to submit your exam. 

<!-- Your Exam Submission Consists of 2 parts:
1) Submit your  solutions to the [Hugging Face competition space](https://huggingface.co/spaces/examhackaton/GraphClassificationNoisyLabels) that hosts the challenge.
2) Complete the submission Form providing the Link to your project GitHub repository. 
The name of the team in the submission [Form](https://docs.google.com/forms/d/e/1FAIpQLScRelE-v612ZP4PO_gsB5prTup3YWbhcWSFn6HM7l_3P5ilCA/viewform?usp=sharing&ouid=104681444587430158482) and in Hugging Face

### Rules for Participation  
- Projects should be carried out in groups of 2 people.
- Submissions must not use any copyrighted, proprietary, or closed-source data or content.  
- Participants are restricted to using only the datasets provided for training.  
- Submissions can be novel solutions or modifications of existing approaches, with clear references for prior work.  -->
#### Submission Details for you GitHub Repo
   - `All submissions must follow the file and folder structure below:  
   
   - **`main.py`**  
      - The script must accept the following command-line arguments:  
        ```bash
        python main.py --test_path <path_to_test.json.gz> --train_path <optional_path_to_train.json.gz>
        ```
      - **Behavior**:  
        - If `--train_path` is provided, the script must train the model using the specified `train.json.gz` file.  
        - If `--train_path` is not provided, the script should **only generate predictions** using the pre-trained model checkpoints provided.  
        - The output must be a **CSV file** named as:  
          ```
          testset_<foldername>.csv
          ```  
          Here, `<foldername>` corresponds to the dataset folder name (e.g., `A`, `B`, `C`, or `D`).  
        - Ensure the correct mapping between test and training datasets:  
          - Example: If `test.json.gz` is located in `./datasets/A/`, the script must use the pre-trained model that was trained on `./datasets/A/train.json.gz`.  
   
   - **Folder and File Naming Conventions**  
     - `checkpoints/`: Directory containing trained model checkpoints. Use filenames such as:  
       ```
       model_<foldername>_epoch_<number>.pth
       ```
       Example: `model_A_epoch_10.pth` 

       Save at least 5 checkpoints for each model.

     - `source/`: Directory for all implemented code (e.g., models, loss functions, data loaders).  
     - `submission/`: Folder containing the predicted CSV files for the four test sets:  
       ```
       testset_A.csv, testset_B.csv, testset_C.csv, testset_D.csv
       ```  
     - `logs/`: Log files for **each training dataset**. Include logs of accuracy and loss recorded every **10 epochs**.  
     - `requirements.txt`: A file listing all dependencies and the Python version. Example:  
       ```
       python==3.8.5
       torch==1.10.0
       numpy==1.21.0
       ```  
     - `README.md`: A clear and concise description of the solution, including:  
       - Image teaser explaning the procedure
       - Overview of the method 


- Ensure that your solution is fully reproducible. Include any random seeds or initialization details used to ensure consistent results (e.g., `torch.manual_seed()` or `np.random.seed()`) and If using a pre-trained model, include the instructions for downloading or specifying the model path.
- **Submission Limits**:
   - Teams or individuals can submit **up to 6 submissions per day**. 
   - Multiple submissions are allowed, but only the **best-performing** model will count toward the leaderboard.
- **Note:** Use `zipthefolder.py` to create submission.gz from submission folder for submission to hugging face.
---

### Dataset Details  

The dataset used in this competition is a subset of the publicly available Protein-Protein Association (PPA) dataset. We have selected 30% of the original dataset, focusing on 6 classes out of the 37 available in the full dataset. For more information about the PPA dataset, including its source and detailed description, please visit the [Hugging Face competition space](https://huggingface.co/spaces/examhackaton/GraphClassificationNoisyLabels).

### How to Run the Code  

This code serves as an example of how to load a dataset and utilize it effectively for training and testing a GNN model:
1. The data set can be download from https://drive.google.com/drive/folders/1Z-1JkPJ6q4C6jX4brvq1VRbJH5RPUCAk?usp=drive_link
2. The `main` file contains the implementation of the GNN model.
3. It uses the traindataset located in one of the data folders (A, B, C, or D) based on the `path_train` argument.
4. The GNN model is trained on the specified traindataset from the folder corresponding to the `path_train` argument.
5. After training, the code generates a CSV file for the test dataset, named based on the `test_path` argument.
6. For example, if `test_path` points to folder B, the output file will be named `testset_B.csv`.
7. If only the `test_path` argument is provided , the code should generate the respective test dataset’s CSV file using the pre-trained model.( This functionality is for you to implement).
