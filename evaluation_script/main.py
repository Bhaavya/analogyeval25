import json
import numpy as np
from scipy.stats import kendalltau


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    # 1. Load the Ground Truth and User Submission files
    with open(test_annotation_file, 'r') as f:
        ground_truth = json.load(f)
    
    with open(user_submission_file, 'r') as f:
        submission = json.load(f)

    # 2. Identify common keys and available metrics
    # We look at the first item in ground_truth to determine which metrics to score
    # (e.g., Phase 1 has 3 metrics, Phase 2 has 6)
    if not ground_truth:
        raise ValueError("Ground truth file is empty.")
    
    sample_key = next(iter(ground_truth))
    metric_names = list(ground_truth[sample_key].keys())
    
    # Storage for vectors
    # Structure: { "metric_name": { "gt": [], "pred": [] } }
    metric_vectors = {m: {"gt": [], "pred": []} for m in metric_names}
    
    # 3. Align data based on Example IDs
    missing_ids = 0
    for example_id, gt_scores in ground_truth.items():
        if example_id in submission:
            pred_scores = submission[example_id]
            for metric in metric_names:
                # Ensure the metric exists in the submission
                if metric in pred_scores:
                    metric_vectors[metric]["gt"].append(gt_scores[metric])
                    metric_vectors[metric]["pred"].append(pred_scores[metric])
        else:
            missing_ids += 1

    if missing_ids > 0:
        print(f"Warning: {missing_ids} IDs from ground truth were missing in the submission.")

    # 4. Compute Kendall Tau for each metric
    final_scores = {}
    tau_values = []

    for metric in metric_names:
        gt_vec = metric_vectors[metric]["gt"]
        pred_vec = metric_vectors[metric]["pred"]
        
        if len(gt_vec) < 2:
            print(f"Not enough data to compute Kendall Tau for {metric}")
            final_scores[metric] = 0.0
            continue

        # Calculate Kendall's tau
        tau, _ = kendalltau(gt_vec, pred_vec)
        
        # Handle cases where tau is NaN (e.g., constant predictions)
        if np.isnan(tau):
            tau = 0.0
            
        final_scores[metric] = tau
        tau_values.append(tau)

    # Calculate Total (Average of all metrics)
    if tau_values:
        final_scores["Total"] = sum(tau_values) / len(tau_values)
    else:
        final_scores["Total"] = 0.0

    # 5. Format Output
    # The platform expects a specific structure. 
    # We map our flat 'final_scores' dict to the required nested format.
    
    output = {}
    
    # Note: The original script used specific logic for 'dev' vs 'test' regarding split names.
    # We preserve that structure here.
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "train_split": final_scores
            }
        ]
        # To display the results in the main leaderboard view
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
        
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "train_split": final_scores
            },
            {
                "test_split": final_scores
            },
        ]
        # To display the results in the main leaderboard view
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Test Phase")
        
    return output