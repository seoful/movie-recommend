import lightfm.evaluation as lfm_eval
import warnings
import pickle
import argparse
warnings.filterwarnings('ignore') 

def evaluate(checkpoint, k):
    with open(checkpoint, 'rb') as inp:
        fitted_model = pickle.load(inp)
    with open('benchmark/data/benchmark_data.pkl', 'rb') as inp:
        data = pickle.load(inp)
    
    test_interactions = data['test_interactions']
    train_interactions = data['train_interactions']
    item_features = data['item_features']
    user_features = data['user_features']

    print("Test auc: %.2f" % lfm_eval.auc_score(fitted_model, test_interactions, train_interactions=train_interactions,item_features=item_features,user_features=user_features).mean())
    print(f"Test precision@{k}: %.2f" % lfm_eval.precision_at_k(fitted_model, test_interactions, train_interactions=train_interactions, k=k,item_features=item_features,user_features=user_features).mean())
    print(f"Test recall@{k}: %.2f" % lfm_eval.recall_at_k(fitted_model, test_interactions, k=k,train_interactions=train_interactions,item_features=item_features,user_features=user_features).mean())
    print("Test reciporial rank: %.2f" % lfm_eval.reciprocal_rank(fitted_model, test_interactions, train_interactions=train_interactions,item_features=item_features,user_features=user_features).mean())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image colorization')

    parser.add_argument('-c','--checkpoint', type=str, help='Checkpoint to use. Defaults to already trained one', default='models/lfm_1.pkl')
    parser.add_argument('-k', type=int, help='First k positions to calculate Precision@k and Recall@k',default=10)


    args = parser.parse_args()
    evaluate(args.checkpoint,args.k)