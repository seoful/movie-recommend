import pandas as pd
import numpy as np
import warnings
import pickle
import argparse
warnings.filterwarnings('ignore') 


def predict(checkpoint, k, id):
    with open(checkpoint, 'rb') as inp:
        fitted_model = pickle.load(inp)
    with open('src/data/dataset_internal.pkl', 'rb') as inp:
        dataset = pickle.load(inp)
    
    items = pd.read_csv('data/interim/items.csv')

    uid_map, ufeature_map, iid_map, ifeature_map = dataset.mapping()

    iid_map_reversed = {v:k for k,v in iid_map.items()}

    if id not in uid_map:
        print ("No such user ID")
        return

    def sample_recommendation(model, user_ids, k=3):

        n_users, n_items = dataset.interactions_shape()

        user_ids_internal = [uid_map[user_id] for user_id in user_ids]

        for user_id, user_id_internal in zip(user_ids, user_ids_internal):

            scores = model.predict(user_id_internal, np.arange(n_items))
            top_items_idxs_internal = np.argsort(-scores)
            top_items_idxs = [iid_map_reversed[item_id] for item_id in top_items_idxs_internal]
            top_items = [items[items['item_id'] == item_id]['title'].values[0] for item_id in top_items_idxs]

            print("User %s" % user_id)
            print("     Recommended:")

            for x in top_items[:k]:
                print("        %s" % x)
    
    sample_recommendation(fitted_model, [id], k)
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image colorization')

    parser.add_argument('id', type=int, help='ID of a user')
    parser.add_argument('-c','--checkpoint', type=str, help='Checkpoint to use. Defaults to already trained one', default='models/lfm_1.pkl')
    parser.add_argument('-k', type=int, help='First k movies to choose',default=3)


    args = parser.parse_args()
    predict(args.checkpoint,args.k, args.id)