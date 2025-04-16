import sys
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from EMG_Classification import FeatureExtractor, ClassificationModel
import itertools

def analyze_feature_combinations(base_extractor, feature_matrix, labels, gestures):
    # Define all possible feature method names
    all_feature_methods = [
        'mav',          # Mean Absolute Value
        'variance',     # Variance
        'signal_energy',# Signal Energy
        'waveform_length', # Total Signal Movement
        'rms'          # Root Mean Square
    ]

    # Results storage
    results = []

    # Generate all possible feature combinations
    for r in range(1, len(all_feature_methods) + 1):
        for combo in itertools.combinations(all_feature_methods, r):
            print(f"\n--- Testing Feature Combination: {combo} ---")
            
            # Create a new feature extractor with selected features
            current_extractor = FeatureExtractor(winlen=base_extractor.winlen, overlap=base_extractor.overlap)
            
            # Dynamically set features using method names
            current_extractor.features = [
                getattr(current_extractor, method_name) 
                for method_name in combo
            ]

            # Determine the number of channels and original features
            n_channels = feature_matrix.shape[1] // len(base_extractor.features)
            
            # Reshape the original feature matrix
            reshaped_matrix = feature_matrix.reshape(-1, len(base_extractor.features), n_channels)
            
            # Find the indices of the features we want to keep
            # This is the key change to handle method reference comparisons
            feature_indices = [
                base_extractor.features.index(
                    next(f for f in base_extractor.features if f.__name__ == method_name)
                ) 
                for method_name in combo
            ]

            # Select features for the current combination
            selected_features_matrix = np.hstack([
                reshaped_matrix[:, idx, :] 
                for idx in feature_indices
            ])

            # Balance Dataset with SMOTE
            try:
                smote = SMOTE(random_state=42)
                balanced_matrix, balanced_labels = smote.fit_resample(selected_features_matrix, labels)
            except Exception as e:
                print(f"SMOTE error: {e}")
                continue

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                balanced_matrix, balanced_labels, test_size=0.3, random_state=42
            )

            # Train and Evaluate Model
            mdl = ClassificationModel()
            try:
                mdl.fit(X_train, y_train)
                y_pred = mdl.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                print(f"Features: {combo}")
                print(f"Accuracy: {accuracy:.4f}")

                results.append({
                    'features': combo,
                    'accuracy': accuracy
                })
            except Exception as e:
                print(f"Model training error: {e}")

    # Sort and print top results
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    print("\n--- Top 5 Feature Combinations ---")
    for result in results[:5]:
        print(f"Features: {result['features']}, Accuracy: {result['accuracy']:.4f}")

    return results

def analyze_window_parameters(data_folder, gestures):
    # Define parameter ranges to explore
    window_lengths = [100,125,150,175,200,225,250,300]
    overlap_percentages = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

    # Results storage
    results = []

    for winlen in window_lengths:
        for overlap_ratio in overlap_percentages:
            overlap = int(winlen * overlap_ratio)
            
            print(f"\n--- Analyzing Window Length: {winlen}, Overlap: {overlap} ---")
            
            # Setup the windowing parameters and feature extractor
            extractor = FeatureExtractor(winlen=winlen, overlap=overlap)

            # Define feature matrix and label vector
            n_features = len(extractor.features)
            n_channels = 8
            feature_matrix = np.zeros((0, n_features * n_channels))
            labels = []

            # Get the dataset folder, figure out which classes are present
            gesture_folders = [os.path.join(data_folder, c) for c in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, c))]

            # Extract feature matrix
            for g, gesture in enumerate(gesture_folders):
                trial_files = [os.path.join(gesture, t) for t in os.listdir(gesture) if os.path.isfile(os.path.join(gesture, t))]
                for trial in trial_files:
                    try:
                        emg = np.loadtxt(trial, delimiter=',')
                        if emg.shape[1] != n_channels:
                            print(f"Warning: Skipping {trial} due to incorrect channel count")
                            continue

                        to_add = extractor.extract_feature_matrix(emg)
                        if to_add.size > 0:
                            feature_matrix = np.vstack((feature_matrix, to_add))
                            labels += [g] * to_add.shape[0]
                    except Exception as e:
                        print(f"Error processing {trial}: {e}")

            # Balance Dataset with SMOTE
            try:
                smote = SMOTE(random_state=42)
                feature_matrix, labels = smote.fit_resample(feature_matrix, labels)
            except Exception as e:
                print(f"Error during dataset balancing: {e}")
                continue

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.3, random_state=42)

            # Fit Classification Model
            mdl = ClassificationModel()
            mdl.fit(X_train, y_train)

            # Evaluate the Model
            y_pred = mdl.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Window Length: {winlen}, Overlap: {overlap}")
            print(f"Accuracy: {accuracy:.4f}")

            results.append({
                'window_length': winlen,
                'overlap': overlap,
                'accuracy': accuracy
            })

    # Sort and print top results
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    print("\n--- Top 5 Window/Overlap Configurations ---")
    for result in results[:5]:
        print(f"Window Length: {result['window_length']}, Overlap: {result['overlap']}, Accuracy: {result['accuracy']:.4f}")

    return results

def calculate_class_accuracies(y_true, y_pred, gestures):
    """
    Calculate accuracy for each class
    
    Parameters:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    gestures (list): List of gesture names corresponding to label indices
    
    Returns:
    dict: Accuracy for each class
    """
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for i, gesture in enumerate(gestures):
        # Find indices of this class
        class_indices = np.where(np.array(y_true) == i)[0]
        
        # Skip if no samples for this class
        if len(class_indices) == 0:
            print(f"Warning: No samples found for class {gesture}")
            class_accuracies[gesture] = {
                'correct': 0,
                'total': 0,
                'accuracy': 0.0
            }
            continue
        
        # Calculate correct predictions for this class
        class_correct = np.sum(
            (np.array(y_true)[class_indices] == np.array(y_pred)[class_indices]).astype(int)
        )
        class_total = len(class_indices)
        
        class_accuracies[gesture] = {
            'correct': int(class_correct),
            'total': class_total,
            'accuracy': float(class_correct) / class_total if class_total > 0 else 0
        }
    
    return class_accuracies

def main():
    # Parse command line inputs, if any
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Get script's directory
    data_folder = os.path.join(base_dir, 'data')
    output_file = os.path.join(base_dir, 'classification', 'models', 'trained_model')

    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    # Setup the windowing parameters and feature extractor
    extractor = FeatureExtractor(winlen=125, overlap=75)

    # Define feature matrix and label vector
    n_features = len(extractor.features)
    n_channels = 8
    feature_matrix = np.zeros((0, n_features * n_channels))
    labels = []

    # Get the dataset folder, figure out which classes are present
    gesture_folders = [os.path.join(data_folder, c) for c in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, c))]
    gestures = [os.path.basename(gesture) for gesture in gesture_folders]

    # # # Analyze window and overlap parameters
    # window_results = analyze_window_parameters(data_folder, gestures)

    # # You can add further processing or model training based on the best configuration here
    # best_config = window_results[0]
    # return
    
    

    # Extract feature matrix (done ONCE)
    for g, gesture in enumerate(gesture_folders):
        trial_files = [os.path.join(gesture, t) for t in os.listdir(gesture) if os.path.isfile(os.path.join(gesture, t))]
        for trial in trial_files:
            try:
                emg = np.loadtxt(trial, delimiter=',')
                if emg.shape[1] != n_channels:
                    print(f"Warning: Skipping {trial} due to incorrect channel count")
                    continue

                to_add = extractor.extract_feature_matrix(emg)
                if to_add.size > 0:
                    feature_matrix = np.vstack((feature_matrix, to_add))
                    labels += [g] * to_add.shape[0]
            except Exception as e:
                print(f"Error processing {trial}: {e}")

    # Check if the feature matrix is empty
    if feature_matrix.size == 0 or len(labels) == 0:
        print("Error: Feature matrix or labels are empty. Please check your data.")
        return

    # Optional: Analyze Feature Combinations
    # Uncomment the following line to run feature combination analysis
    # analyze_feature_combinations(extractor, feature_matrix, labels, gestures)
    # return  # Uncomment this to stop after analysis

    # Balance Dataset with SMOTE
    try:
        smote = SMOTE(random_state=42)
        feature_matrix, labels = smote.fit_resample(feature_matrix, labels)
    except Exception as e:
        print(f"Error during dataset balancing: {e}")
        return

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.3, random_state=42)

    # Fit Classification Model
    mdl = ClassificationModel()
    mdl.fit(X_train, y_train)

    # Evaluate the Model
    y_pred = mdl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model evaluation completed. Test accuracy: {accuracy:.2f}")

    class_accuracies = calculate_class_accuracies(y_test, y_pred, gestures)
    
    # Print per-class accuracies
    print("\nPer-Class Accuracies:")
    for gesture, stats in class_accuracies.items():
        print(f"{gesture}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']} correct)")

    # Optional: Print full classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=gestures))

    #return

    # Save results
    filename = output_file + '.pkl'
    saved_model = {'feature_extractor': extractor, 'mdl': mdl, 'gestures': gestures}

    with open(filename, 'wb') as file:
        pickle.dump(saved_model, file)

    print("Training complete. Model saved to:", filename)

if __name__ == '__main__':
    main()