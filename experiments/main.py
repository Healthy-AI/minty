import sys
sys.path.append('/Users/stempfle/Documents/Interpretablity+Missingness/Code/Working_code/minty/experiment')

sys.path.append('../')

import pandas as pd
from models.config import *
from models.evaluation import *
from data.data import *
from models.Neumiss.src.test import *
from util.neumiss_util import *

import argparse
import pickle
from datetime import datetime
import torch


if __name__ == "__main__":
    """
     Argument parsing
     """
    parser = argparse.ArgumentParser(description='Train MINTY and baselines')
    parser.add_argument('-ds', '--dataset', type=str, dest='dataset', help='dataset help')
    parser.add_argument('-es', '--estimator', type=str, default=None, dest='estimator', help='estimator help')
    parser.add_argument('-pa', '--parameters', dest='parameters', help='parameter help', nargs='*')
    parser.add_argument('-i', '--imputation', type=str, default='zero', dest='imputation', help='imputation help')
    parser.add_argument('-sp', '--split', type=float, default=0.33, dest='split', help='split help')
    parser.add_argument('-s', '--seed', type=int, default=0, dest='seed', help='seed help')
    parser.add_argument('-fr', '--frac', type=float, default=1.0, dest='frac', help='frac help')
    parser.add_argument('-c', '--classifier', type=int, default=False, dest='classifier', help='classifier help')
    parser.add_argument('-r', '--results', type=str, default='../results', dest='results', help='results help')
    parser.add_argument('-it', '--iteration', type=int, default=1, dest='iteration', help="iteration help")
    args = parser.parse_args()

    print('Summary of parsed args:', args)

    if args.estimator is None:
        raise Exception('No estimator specified')

    """
    Load data 
    """
    Data_ldr=Data_Loader(seed=args.seed, dataset=args.dataset, imputation=args.imputation, split=args.split, frac=args.frac)
    I,S , X_train, X_test, X_val, y_train, y_test, y_val, X_df = Data_ldr.load_data()

    if args.estimator == 'neumiss':
        ds_train, ds_val, ds_test = Preporcess_neumiss().transform_to_tensor(X_train, X_test, X_val, y_train, y_test, y_val)

    """
    Gather estimator parameters
    """
    parameter_list = {}  # initialize empty dic for the parameters
    #extract key words and value from list
    if not args.parameters is None and len(args.parameters)>0:
        for key, value in zip(args.parameters[0::2], args.parameters[1::2]):
            # Extract keywords from parser into a dic
            try:
                parameter_list = {**parameter_list, key: float(value)}
            except:
                parameter_list = {**parameter_list, key: str(value)}

    BOOL_PARAMETERS = ['fit_intercept', 'precompute', 'reg_refit', 'reg_relaxed', 'silent']
    for k, v in parameter_list.items():
        if k in BOOL_PARAMETERS:
            parameter_list[k] = (v == 'True') or (v == 'true') or (v == 1)

    INT_PARAMETERS = ['max_depth', 'min_samples_leaf']
    for k, v in parameter_list.items():
        if k in INT_PARAMETERS:
            parameter_list[k] = int(v)
    print('These are the parameters', parameter_list)

    """
    Impute, fit estimator and evaluate 
    """
    results_df = pd.DataFrame()

    #run impute and fit estimator i times
    if args.estimator == 'neumiss':
        p = ds_train.tensors[0].shape[1]
        #pass parameters from config to neumiss model
        n_features_value_ = parameter_list.get('n_features')
        n_features_value = int(n_features_value_)
        mlp_depth_value_ = parameter_list.get('mlp_depth')
        mlp_depth_value = int(mlp_depth_value_)
        mlp_width_value_ = parameter_list.get('mlp_width')
        mlp_width_value = int(mlp_width_value_)
        neumiss_depth_value_ = parameter_list.get('neumiss_depth')
        neumiss_depth_value = int(neumiss_depth_value_)

        preds, neumiss_estimator = Neumiss_class(ds_train, ds_val, ds_test, n_features=n_features_value, neumiss_depth=neumiss_depth_value,
                                     mlp_depth=mlp_depth_value, mlp_width=mlp_width_value).run_neumiss()

        torch.save(neumiss_estimator.state_dict(), 'estimator_params.pth')
        #estimator.load_state_dict(torch.load('estimator_params.pth'))
        #estimator.eval()
        # torch.save(model, 'model.pth')
        # model = torch.load('model.pth')
        """
        Evaluation procedure
        """
        y_pred_val = preds[1]
        y_pred_train = preds[0]
        y_pred_test = preds[2]
        #probabilities_names but not used in Neumiss 
        y_pred_prob_val = preds[1]
        y_pred_prob_train = preds[0]
        y_pred_prob_test = preds[2]
        results_val = Evaluation().evaluate(y_pred_val, y_pred_prob_val, y_val, S, label='validation', classifier=False)
        missingness_R_val = Evaluation().missingness_reliance(X_val, neumiss_estimator, label='validation')
        results_val = {**results_val, **missingness_R_val}
        print('These are the validation results', results_val)
        results_train = Evaluation().evaluate(y_pred_train, y_pred_prob_train, y_train, S, label='train', classifier=False)
        missingness_R_train = Evaluation().missingness_reliance(X_train, neumiss_estimator, label='train')
        results_train = {**results_train, **missingness_R_train}
        print('These are the train results', results_train)
        results_test = Evaluation().evaluate(y_pred_test, y_pred_prob_test, y_test, S, label='test', classifier=False)
        missingness_R_test = Evaluation().missingness_reliance(X_test, neumiss_estimator, label='test')
        results_test = {**results_test, **missingness_R_test}
        print('These are the test results', results_test)

        results_df = pd.DataFrame(
            {**results_val, **results_train, **results_test, 'dataset': args.dataset, **parameter_list,
             'estimator': args.estimator, 'imputation': args.imputation, 'split': args.split, 'seed': args.seed,
             'frac': args.frac, 'iteration': args.iteration, 'X_train_size': X_train.shape[0],
             'X_val_size': X_val.shape[0], 'X_test_size': X_test.shape[0]})

    elif args.imputation == "mice":
        for it in range(args.iteration):
            # Estimator from config file
            estimator = get_estimators()[args.estimator]()
            # Set parameters obtained from args.parameters
            estimator.set_params(**parameter_list)
            # Fit estimator on imputed data
            estimator.fit(I.transform(X_train), y_train)
            """
            Evaluation procedure
            """
            y_pred_val, y_pred_prob_val = Evaluation().predict(X_val, I, estimator, classifier=args.classifier)
            results_val = Evaluation().evaluate(y_pred_val, y_pred_prob_val, y_val, S, label='validation', classifier=args.classifier)
            missingness_R_val = Evaluation().missingness_reliance(X_val, estimator, label='validation')
            results_val = {**results_val, **missingness_R_val}
            print('These are the validation results', results_val)
            y_pred_train, y_pred_prob_train = Evaluation().predict(X_train, I, estimator, classifier=args.classifier)
            results_train = Evaluation().evaluate(y_pred_train, y_pred_prob_train, y_train, S, label='train', classifier=args.classifier)
            missingness_R_train = Evaluation().missingness_reliance(X_train, estimator, label='train')
            results_train = {**results_train, **missingness_R_train}
            print('These are the train results', results_train)
            y_pred_test, y_pred_prob_test = Evaluation().predict(X_test, I, estimator, classifier=args.classifier)
            results_test = Evaluation().evaluate(y_pred_test, y_pred_prob_train, y_test, S, label='test', classifier=args.classifier)
            missingness_R_test= Evaluation().missingness_reliance(X_test, estimator, label='test')
            results_test = {**results_test, **missingness_R_test}
            print('These are the test results', results_test)

        results_df_iteration = pd.DataFrame(
            {**results_val, **results_train, **results_test, 'dataset': args.dataset, **parameter_list,
             'estimator': args.estimator, 'imputation': args.imputation, 'split': args.split, 'seed': args.seed,
             'frac': args.frac, 'iteration': args.iteration, 'X_train_size': X_train.shape[0],
             'X_val_size': X_val.shape[0],
             'X_test_size': X_test.shape[0]})
        # save results after each iteration
        results_df = pd.concat([results_df, results_df_iteration], axis=0)

    else:
        # Estimator from config file
        estimator = get_estimators()[args.estimator]()
        # Set parameters obtained from args.parameters
        estimator.set_params(**parameter_list)
        # Fit estimator on imputed data
        if args.estimator == 'rulefit':
            X_train = I.transform(X_train)
            X_train = pd.DataFrame(X_train, columns=X_df.columns)
            for column in X_train.columns:
                 X_train[column] = X_train[column].astype(np.float32)
            estimator.fit(X_train, y_train.ravel(), feature_names=X_train.columns)
        else:
            estimator.fit(I.transform(X_train), y_train)
        """
        Evaluation procedure
        """
        y_pred_val, y_pred_prob_val = Evaluation().predict(X_val, I, estimator, classifier=args.classifier)
        results_val = Evaluation().evaluate(y_pred_val, y_pred_prob_val, y_val, S, label='validation', classifier=args.classifier)
        missingness_R_val = Evaluation().missingness_reliance(X_val, estimator, label='validation')
        results_val = {**results_val, **missingness_R_val}
        print('These are the validation results', results_val)
        y_pred_train, y_pred_prob_train = Evaluation().predict(X_train, I, estimator, classifier=args.classifier)
        results_train = Evaluation().evaluate(y_pred_train, y_pred_prob_train, y_train, S,label='train', classifier=args.classifier)
        missingness_R_train = Evaluation().missingness_reliance(X_train, estimator, label='train')
        results_train = {**results_train, **missingness_R_train}
        print('These are the train results', results_train)
        y_pred_test, y_pred_prob_test = Evaluation().predict(X_test, I, estimator, classifier=args.classifier)
        results_test = Evaluation().evaluate(y_pred_test, y_pred_prob_test, y_test, S, label='test', classifier=args.classifier)
        missingness_R_test = Evaluation().missingness_reliance(X_test, estimator, label='test')
        results_test = {**results_test, **missingness_R_test}
        print('These are the test results', results_test)

        """
        Save results
        """
        results_df = pd.DataFrame(
            {**results_val, **results_train, **results_test, 'dataset': args.dataset, **parameter_list,
             'estimator': args.estimator, 'imputation': args.imputation, 'split': args.split, 'seed': args.seed,
             'frac': args.frac, 'iteration': args.iteration, 'X_train_size': X_train.shape[0],
             'X_val_size': X_val.shape[0], 'X_test_size': X_test.shape[0]})

    """
    Save results and model in csv and pkl 
    """
    parameter_str = '_'.join(['%s=%s' % (k, str(v)) for k, v in parameter_list.items()])
    exp_id = "%s_%s_%s_%d" % (args.dataset, args.estimator, args.imputation, np.random.randint(10000))

    date = datetime.now().strftime("%I_%M_%S_%p")

    model_filename = f"model_{exp_id}.pkl"
    model_filename = f'{args.results}/{date}_%s' % model_filename

    results_df['model_file'] = model_filename

    results_filename = f"results_{exp_id}.csv"
    results_filename = f'{args.results}/{date}_%s' % results_filename
    results_df.to_csv(results_filename, index=False)

    # save model in pkl
    if args.estimator == 'neumiss':
        neumiss_estimator.load_state_dict(torch.load('estimator_params.pth'))
        neumiss_estimator.eval()
        model_info = {'estimator': neumiss_estimator, 'imputer': I, 'scaler': S, 'arguments': args}
    else:
        model_info = {'estimator': estimator, 'imputer': I, 'scaler': S, 'arguments': args, 'results_file': results_filename}

    with open(model_filename, 'wb') as file:
        pickle.dump(model_info, file)
    file.close()

