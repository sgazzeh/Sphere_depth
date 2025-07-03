import numpy as np

def compute_lambda_and_errors(pred_train, gt_train):
    """
    Compute lambda per image.
    """
    lambdas = []
    errors_train = []

    for i in range(pred_train.shape[1]):
        d_pred = pred_train[:, i]
        d_gt = gt_train[:, i]

        lambda_img = (np.sum(d_pred * d_gt)) / (np.sum(d_pred ** 2))
        lambdas.append(lambda_img)

        error_img = 0.5 * np.sum((lambda_img * d_pred - d_gt) ** 2)
        errors_train.append(error_img)

        print(f"Train Image {i+1}: λ = {lambda_img:.4f}, Error = {error_img:.4f}")

    lambda_avg = np.mean(lambdas)
    print(f"\nAverage λ: {lambda_avg:.4f}")

    return lambdas, errors_train, lambda_avg

def compute_test_errors(pred_test, gt_test, lambda_avg):
    """
    Compute test errors using average lambda.
    """
    errors_test = []

    for i in range(pred_test.shape[1]):
        d_pred = pred_test[:, i]
        d_gt = gt_test[:, i]

        error_img = 0.5 * np.sum((lambda_avg * d_pred - d_gt) ** 2)
        errors_test.append(error_img)

        print(f"Test Image {i+1}: Error = {error_img:.4f}")

    final_error = np.mean(errors_test)
    print(f"\nFinal Test Error: {final_error:.4f}")

    return errors_test, final_error
