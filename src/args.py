import argparse



def topic_model_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save_model',
        action='store_true',  # If this flag is provided, it sets save_model to True
        help="Flag to indicate whether to save the model"
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='/Users/sarthaktyagi/Desktop/LLM_Perspective/src/topic_model_ckpt',  
        help="Directory to save or load the model checkpoints (default: 'checkpoints/')"
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Users/sarthaktyagi/Desktop/LLM_Perspective/data/train.json',  
        help="load the dataset"
    )


    args = parser.parse_args()
    
    
    return args
