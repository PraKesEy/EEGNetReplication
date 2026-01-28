"""Train a hazard classifier from cached NASA NeoWs data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from eegnet_repl.config import Paths
from eegnet_repl.dataset import BCICI2ADataset, build_dataset_from_preprocessed
from eegnet_repl.logger import logger
from eegnet_repl.model import EEGNet, train, evaluate_model
# from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import KFold

import numpy as np
import torch
# import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.001


def within_subject_training(epochs=EPOCHS):
    '''
    Train and validate EEGNet model for within-subject classification.

    Arguments:
    - epochs: Number of training epochs
    
    Returns:
    - Per subject test accuracies array
    - Average test accuracy across subjects
    - Best model state dicts per subject array
    '''
    
    paths = Paths.from_here()
    
    # Initial split into train and test sets
    per_subject_test_acc = []
    best_model_states_all_subjects = []

    # Per subject training in a loop
    for subject_id in range(1, 10):  # Assuming subject IDs are from 1 to 9
        logger.info(f"Training for Subject {subject_id}")
        
        # Load preprocessed data for the subject
        subject_train_data = build_dataset_from_preprocessed(subject=subject_id)
        subject_eval_data = build_dataset_from_preprocessed(subject=subject_id, mode="Eval")

        # Combine the data for K-Fold CV by concatenating numpy arrays
        combined_X = np.concatenate([subject_train_data.X, subject_eval_data.X], axis=0)
        combined_y = np.concatenate([subject_train_data.y, subject_eval_data.y], axis=0)
        subject_data = BCICI2ADataset(combined_X, combined_y)

        # test accuracy array for this subject
        subject_test_acc = []

        # track best model for this subject
        best_val_acc = 0  # Track best validation accuracy instead of loss
        best_model_state = None

        # K-Fold Cross Validation on train_data
        splits = 4
        kf = KFold(n_splits=splits, shuffle=True, random_state=42)
        
        for fold, (train_val_ids, test_ids) in enumerate(kf.split(subject_data)):
            logger.info(f"  Fold {fold+1}/{splits}")

            # Further split train_val into train and validation (80-20 split for larger val set)
            val_size = len(train_val_ids) // 5  # 20% validation, 80% train
            train_ids = train_val_ids[val_size:]
            val_ids = train_val_ids[:val_size]

            # Create data subsets
            train_subset = Subset(subject_data, train_ids)
            val_subset = Subset(subject_data, val_ids)
            test_subset = Subset(subject_data, test_ids)

            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)  # Reduced from 64
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

            # Initialize model, loss function, and optimizer
            model = EEGNet(C=subject_data.X.shape[1], T=subject_data.X.shape[2], p=0.5)  # p=0.5 for within-subject
            
            optimizer = optim.Adam(
                params=model.parameters(),
                lr=LEARNING_RATE,
                # weight_decay=1e-4,  # Add L2 regularization
                eps=1e-07,
                foreach=None,
                fused=None,
            )
            
            loss_fn = nn.CrossEntropyLoss()

            # Train the model
            best_fold_model, _, val_losses, val_accuracies = train(
                model, optimizer, loss_fn, train_loader, val_loader, 
                nepochs=epochs
            )
            
            # Load best model from training
            model.load_state_dict(best_fold_model)
            
            # Get final validation and test accuracies
            final_val_acc = max(val_accuracies)
            test_acc = evaluate_model(model, test_loader)
            
            # Log validation and test accuracy
            logger.info(f"    Validation Accuracy: {final_val_acc:.2f}%")
            logger.info(f"    Test Accuracy: {test_acc:.2f}%")
            
            # Store test accuracy for this fold
            subject_test_acc.append(test_acc)
            
            # Track best model based on validation accuracy (not loss)
            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                best_model_state = best_fold_model
        
        # Calculate average test accuracy for this subject
        avg_test_acc = sum(subject_test_acc) / len(subject_test_acc)
        per_subject_test_acc.append(avg_test_acc)
        logger.info(f"Subject {subject_id} - Average Test Accuracy: {avg_test_acc:.2f}%")
        
        # Save best model for this subject
        model_path = paths.models / f"subject_{subject_id:02d}_best_model.pth"
        paths.models.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, model_path)
        logger.info(f"Saved best model for Subject {subject_id}: {model_path}")
        
        # Store best model state dict
        best_model_states_all_subjects.append(best_model_state)
    
    # Calculate average test accuracy across all subjects
    avg_test_acc_all_subjects = sum(per_subject_test_acc) / len(per_subject_test_acc)
    logger.info(f"\nOverall Average Test Accuracy across all subjects: {avg_test_acc_all_subjects:.2f}%")
    
    return per_subject_test_acc, avg_test_acc_all_subjects, best_model_states_all_subjects


def cross_subject_training(epochs=EPOCHS):
    '''
    Train and validate EEGNet model for cross-subject classification.
    For each subject: train on 5 random subjects, validate on 3 others, test on target subject.
    Repeat 10 times per subject (90 folds total).

    Returns:
    - Best model state dict based on validation accuracy
    - Per subject test accuracies array
    - Average test accuracy across subjects
    '''
    
    paths = Paths.from_here()
    
    # Load data for all subjects
    logger.info("Loading data for all subjects...")
    all_subjects_data = {}
    all_subjects_eval_data = {}
    
    for subject_id in range(1, 10):
        all_subjects_data[subject_id] = build_dataset_from_preprocessed(subject=subject_id)
        all_subjects_eval_data[subject_id] = build_dataset_from_preprocessed(subject=subject_id, mode="Eval")
    
    # Track results
    all_fold_test_acc = []
    per_subject_test_acc = []
    best_val_loss = 100
    best_model_state = None
    fold_count = 0
    
    # For each subject as test subject
    for subject_id in range(1, 10):
        logger.info(f"\nTraining with Subject {subject_id} as test subject")
        
        subject_test_acc = []
        
        # Get test data for this subject
        test_data = all_subjects_eval_data[subject_id]
        
        # Get other subjects
        other_subjects = [s for s in range(1, 10) if s != subject_id]
        
        # Repeat 10 times
        for repeat in range(10):
            fold_count += 1
            logger.info(f"  Fold {fold_count}/90 (Subject {subject_id}, Repeat {repeat+1}/10)")
            
            # Randomly select 5 subjects for training, 3 for validation
            rng = np.random.RandomState(42 + fold_count)  # For reproducibility
            shuffled_subjects = rng.permutation(other_subjects)
            train_subjects = shuffled_subjects[:5]
            val_subjects = shuffled_subjects[5:]
            
            # Combine training data from selected subjects
            train_X_list = []
            train_y_list = []
            for subj in train_subjects:
                train_X_list.append(all_subjects_data[subj].X)
                train_y_list.append(all_subjects_data[subj].y)
            
            train_X = np.concatenate(train_X_list, axis=0)
            train_y = np.concatenate(train_y_list, axis=0)
            
            # Combine validation data from selected subjects
            val_X_list = []
            val_y_list = []
            for subj in val_subjects:
                val_X_list.append(all_subjects_data[subj].X)
                val_y_list.append(all_subjects_data[subj].y)
            
            val_X = np.concatenate(val_X_list, axis=0)
            val_y = np.concatenate(val_y_list, axis=0)
            
            # Create datasets
            train_dataset = BCICI2ADataset(train_X, train_y)
            val_dataset = BCICI2ADataset(val_X, val_y)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
            
            # Initialize model with p=0.25 for cross-subject
            model = EEGNet(C=train_X.shape[1], T=train_X.shape[2], p=0.25)
            
            optimizer = optim.Adam(
                params=model.parameters(),
                lr=LEARNING_RATE,
                eps=1e-07,
                foreach=None,
                fused=None,
            )
            loss_fn = nn.CrossEntropyLoss()
            
            # Train the model
            best_fold_model, _, val_losses, val_accuracies = train(
                model, optimizer, loss_fn, train_loader, val_loader, nepochs=epochs
            )
            
            # Load best model from training
            model.load_state_dict(best_fold_model)
            
            # Get final validation loss and accuracy
            final_val_acc = max(val_accuracies)
            final_val_loss = min(val_losses)
            
            # Test the model
            test_acc = evaluate_model(model, test_loader)
            
            # Log results
            logger.info(f"    Validation Accuracy: {final_val_acc:.2f}%")
            logger.info(f"    Test Accuracy: {test_acc:.2f}%")
            
            # Store test accuracy
            subject_test_acc.append(test_acc)
            all_fold_test_acc.append(test_acc)
            
            # Track best model globally
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_model_state = best_fold_model
        
        # Calculate average test accuracy for this subject
        avg_test_acc = sum(subject_test_acc) / len(subject_test_acc)
        per_subject_test_acc.append(avg_test_acc)
        logger.info(f"Subject {subject_id} - Average Test Accuracy: {avg_test_acc:.2f}%")
    
    # Calculate overall statistics
    avg_test_acc_all = sum(all_fold_test_acc) / len(all_fold_test_acc)
    std_error = np.std(all_fold_test_acc) / np.sqrt(len(all_fold_test_acc))
    
    logger.info(f"\nOverall Average Test Accuracy: {avg_test_acc_all:.2f}% ± {std_error:.2f}%")
    logger.info(f"Mean per-subject test accuracy: {np.mean(per_subject_test_acc):.2f}%")
    
    # Save best model
    model_path = paths.models / "cross_subject_best_model.pth"
    paths.models.mkdir(parents=True, exist_ok=True)
    torch.save(best_model_state, model_path)
    logger.info(f"Saved best cross-subject model: {model_path}")
    
    return best_model_state, per_subject_test_acc, avg_test_acc_all


def generate_ws_report(per_subject_test_acc, avg_test_acc_all_subjects, best_model_states_all_subjects):
    """
    Generate a within-subject training report in JSON format.
    
    Args:
        per_subject_test_acc: List of average test accuracies per subject
        avg_test_acc_all_subjects: Overall average test accuracy across all subjects
        best_model_states_all_subjects: List of best model state dictionaries per subject
    """
    paths = Paths.from_here()
    
    # Create reports directory if it doesn't exist
    paths.reports.mkdir(parents=True, exist_ok=True)
    
    # Create report data structure
    report_data = {
        "training_type": "Within-Subject",
        "timestamp": datetime.now().isoformat(),
        "model_parameters": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "dropout_probability": 0.5,
            "cross_validation_folds": 4
        },
        "overall_results": {
            "average_test_accuracy": round(avg_test_acc_all_subjects, 2),
            "number_of_subjects": len(per_subject_test_acc),
            "best_subject_accuracy": round(max(per_subject_test_acc), 2),
            "worst_subject_accuracy": round(min(per_subject_test_acc), 2),
            "accuracy_std": round(float(np.std(per_subject_test_acc)), 2)
        },
        "per_subject_results": [],
        "model_info": {
            "architecture": "EEGNet",
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "saved_models_count": len(best_model_states_all_subjects)
        }
    }
    
    # Add per-subject results
    for subject_id in range(1, len(per_subject_test_acc) + 1):
        subject_result = {
            "subject_id": subject_id,
            "test_accuracy": round(per_subject_test_acc[subject_id - 1], 2),
            "model_saved": f"subject_{subject_id:02d}_best_model.pth",
            "performance_rank": 0  # Will be filled after sorting
        }
        report_data["per_subject_results"].append(subject_result)
    
    # Sort subjects by accuracy and assign ranks
    sorted_subjects = sorted(report_data["per_subject_results"], 
                           key=lambda x: x["test_accuracy"], reverse=True)
    
    for rank, subject in enumerate(sorted_subjects, 1):
        # Find the subject in original list and update rank
        for subj in report_data["per_subject_results"]:
            if subj["subject_id"] == subject["subject_id"]:
                subj["performance_rank"] = rank
                break
    
    # Add summary statistics
    report_data["summary_statistics"] = {
        "accuracy_distribution": {
            "above_average_subjects": len([acc for acc in per_subject_test_acc if acc > avg_test_acc_all_subjects]),
            "below_average_subjects": len([acc for acc in per_subject_test_acc if acc < avg_test_acc_all_subjects]),
            "at_average_subjects": len([acc for acc in per_subject_test_acc if acc == avg_test_acc_all_subjects])
        },
        "accuracy_quartiles": {
            "q1": round(float(np.percentile(per_subject_test_acc, 25)), 2),
            "q2_median": round(float(np.percentile(per_subject_test_acc, 50)), 2),
            "q3": round(float(np.percentile(per_subject_test_acc, 75)), 2)
        }
    }
    
    # Generate filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"within_subject_training_report_{timestamp_str}.json"
    report_path = paths.reports / report_filename
    
    # Save report to JSON file
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Within-Subject training report saved to: {report_path}")
    
    # Also save a latest report (overwrite previous)
    latest_report_path = paths.reports / "latest_within_subject_report.json"
    with open(latest_report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Latest Within-Subject report updated: {latest_report_path}")
    
    return report_path


def generate_cs_report(best_model_state, per_subject_test_acc, avg_test_acc_all):
    """
    Generate a cross-subject training report in JSON format.
    
    Args:
        best_model_state: Best model state dictionary
        per_subject_test_acc: List of average test accuracies per subject
        avg_test_acc_all: Overall average test accuracy
    """
    paths = Paths.from_here()
    
    # Create reports directory if it doesn't exist
    paths.reports.mkdir(parents=True, exist_ok=True)
    
    # Create report data structure
    report_data = {
        "training_type": "Cross-Subject",
        "timestamp": datetime.now().isoformat(),
        "model_parameters": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "dropout_probability": 0.25,
            "total_folds": 90,
            "repeats_per_subject": 10,
            "train_subjects_per_fold": 5,
            "validation_subjects_per_fold": 3
        },
        "overall_results": {
            "average_test_accuracy": round(avg_test_acc_all, 2),
            "standard_error": round(float(np.std(per_subject_test_acc) / np.sqrt(len(per_subject_test_acc))), 2),
            "number_of_test_subjects": len(per_subject_test_acc),
            "best_subject_accuracy": round(max(per_subject_test_acc), 2),
            "worst_subject_accuracy": round(min(per_subject_test_acc), 2),
            "accuracy_std": round(float(np.std(per_subject_test_acc)), 2)
        },
        "per_subject_results": [],
        "model_info": {
            "architecture": "EEGNet",
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "saved_model": "cross_subject_best_model.pth"
        }
    }
    
    # Add per-subject results
    for subject_id in range(1, len(per_subject_test_acc) + 1):
        subject_result = {
            "test_subject_id": subject_id,
            "test_accuracy": round(per_subject_test_acc[subject_id - 1], 2),
            "performance_rank": 0  # Will be filled after sorting
        }
        report_data["per_subject_results"].append(subject_result)
    
    # Sort subjects by accuracy and assign ranks
    sorted_subjects = sorted(report_data["per_subject_results"], 
                           key=lambda x: x["test_accuracy"], reverse=True)
    
    for rank, subject in enumerate(sorted_subjects, 1):
        # Find the subject in original list and update rank
        for subj in report_data["per_subject_results"]:
            if subj["test_subject_id"] == subject["test_subject_id"]:
                subj["performance_rank"] = rank
                break
    
    # Add summary statistics
    report_data["summary_statistics"] = {
        "accuracy_distribution": {
            "above_average_subjects": len([acc for acc in per_subject_test_acc if acc > avg_test_acc_all]),
            "below_average_subjects": len([acc for acc in per_subject_test_acc if acc < avg_test_acc_all]),
            "at_average_subjects": len([acc for acc in per_subject_test_acc if acc == avg_test_acc_all])
        },
        "accuracy_quartiles": {
            "q1": round(float(np.percentile(per_subject_test_acc, 25)), 2),
            "q2_median": round(float(np.percentile(per_subject_test_acc, 50)), 2),
            "q3": round(float(np.percentile(per_subject_test_acc, 75)), 2)
        }
    }
    
    # Generate filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"cross_subject_training_report_{timestamp_str}.json"
    report_path = paths.reports / report_filename
    
    # Save report to JSON file
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Cross-Subject training report saved to: {report_path}")
    
    # Also save a latest report (overwrite previous)
    latest_report_path = paths.reports / "latest_cross_subject_report.json"
    with open(latest_report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Latest Cross-Subject report updated: {latest_report_path}")
    
    return report_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train a EEGNet model.")
    parser.add_argument("--trainingType", type=str, help="Training type [Cross-Subject, Within-Subject].", default="Within-Subject")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--generateReport", type=bool, default=True, help="Generate report after training.")
    args = parser.parse_args()

    if args.trainingType == "Within-Subject":
        logger.info("Training Within-Subject models for all subjects...")
    else:
        logger.info("Training Cross-Subject model...")


    if args.trainingType == "Within-Subject":
        per_subject_test_acc, avg_test_acc_all_subjects, best_model_states_all_subjects = within_subject_training(epochs=args.epochs)
        if args.generateReport:
            generate_ws_report(per_subject_test_acc, avg_test_acc_all_subjects, best_model_states_all_subjects)
    else:
        best_model_state, per_subject_test_acc, avg_test_acc_all = cross_subject_training(epochs=args.epochs)
        if args.generateReport:
            generate_cs_report(best_model_state, per_subject_test_acc, avg_test_acc_all)

if __name__ == "__main__":
    main()
