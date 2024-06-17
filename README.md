## Prerequisites
Before running the project, ensure that you have all the necessary packages installed. I suggest running the project on Jupyter Notebook with Anaconda to manage the packages for you.
To install the required packages, type:
```bash
pip install -r requirements.txt
```

## Files Overview
There are a few files, each serving their own purpose:

- **`functions.py`**: This is a library that holds all of the helper functions for the ML model/data detection.
- **`label_data.ipynb`**: A notebook used to label more data. The more data you label, the better the results. Follow the comments inside the file to see how to do this.
- **`train_model.ipynb`**: A notebook to train the model. This uses the data from `label_data.ipynb`, so make sure to label your data first.
- **`classify_image.ipynb`**: A notebook to read an image and print its classification to be copied over to a csv.

## Instructions

1. **Get required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Label Data**:
   If your data is already labeled and the model is trained, skip to step 4. Otherwise, label some data. Save images that you will label to `/training_images` directory.
    - Follow the format of currently labeled data inside the file.

3. **Train Model**:
   If you have already labeled and trained your data, you can skip to step 4. Otherwise, train the model using `train_model.ipynb`. 
    - Properly label and run `label_data.ipynb`.
    - Run `train_model.ipynb`.
    - Check accuracy and validation accuracy after running `history = model.fit(...)`.
    - The output should look like this:
        ```bash
        Epoch 1/40
        12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 40ms/step - accuracy: 0.8534 - loss: 0.3476 - val_accuracy: 0.9375 - val_loss: 0.2434
        ```
    - Ensure that these values are decent. If you run multiple times, you will get slightly different results. On the 5 trained images now, the highest accuracy achieved was around 0.91 or 91%.

4. **Classify Image**:
   Run `classify_image.ipynb` on any image you want to classify. Make sure to change the filepath correctly. The output should be in a format that can easily be inserted into a CSV.

## Notes for Better Results

1. Take pictures that show as much of each well as possible. Avoid having the corners be tiny slivers of whatever solution is inside them.
2. Either train a lot of data with multiple background colors or stick to one background color for the wells. The base 5 images are for a pink background, so I suggest using pink paper for the well backdrop moving forward.
3. If possible, get most pictures from the same distance and angle. While the code will work from pretty much any angle and reasonable distance, more consistent training data should yield better results.

---

PS: I hope this saves you all some time. If you have any questions, don't hesitate to have Cameron reach out to me, or contact me on [LinkedIn](https://www.linkedin.com/in/kaelan-trowbridge-7ba03b1b0/)

