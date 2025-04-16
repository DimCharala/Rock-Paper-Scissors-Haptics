# Rock-Paper-Scissors Game with Computer Vision and Haptics

This project was completed by [Stavropoulos Konstantinos](https://github.com/StavropoulosK) and Charalampopoulos Dimitrios for the "Interactive Technologies" course within the Department of Electrical and Computer Engineering of University of Patras during the Winter Semester of the 2024-2025 academic year.

The goal of this project was to create an interactive "Rock-Paper-Scissors" game using optical and haptic data from the player. The optical part of the project is described in detail in [Konstantinos' Github repository](https://github.com/StavropoulosK/Human-Interaction-with-Computer-Vision). 

For the haptics of the project a Myo armband, which uses electromyography signals, was used using [myo-python library](https://github.com/NiklasRosenstein/myo-python). A support vector machine was trained with the help of [MYO toolbox for Ecole Centrale de Nantes](https://github.com/smetanadvorak/myo_ecn) and custom data created from our team.

## Installation Requirements

Before running the game, ensure you have the following installed:

*   **Python 64-bit >= 3.9**:  Download from [python.org](https://www.python.org/downloads/)
*   **Python packages**: Install these using pip:

    ```
    pip install mediapipe==0.10.13
    pip install cvzone
    pip install myo-python
    pip install numpy
    pip install scikit-learn
    ```

### Optional: Training the Model

If you wish to retrain the gesture recognition model (this is **not required** to run the `rockPaperScissors.py` game), you can install the project in editable mode.  Navigate to the project directory in your terminal and run:

```
pip install -e .
```

