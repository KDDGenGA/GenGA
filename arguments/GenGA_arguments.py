import argparse

'''
Define the arg parset for args.
'''

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset name.')
    parser.add_argument('--loss_type', type=str, default='l2', help='Loss type, supported: l1, l2, huber.')

    # Diffusion settings
    parser.add_argument('--scheduler', type=str, default='cosine', help='linear, cosine')
    parser.add_argument('--diffsteps', type=int, default=1000, help='Steps for defusion')

    # denoising model arguments
    parser.add_argument('--gnn', type=str, default='GCN', help='GNN Layer')
    parser.add_argument('--gnn_dim', type=int, default=256, help='GNN hideen dimension')
    parser.add_argument('--with_time_emb', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--diff_epochs', type=int, default=10000, help='Diffusion Training epochs.')
    parser.add_argument('--class_cond', type=lambda x: (str(x).lower() == 'true'), default=False)

    # noisy classifier arguments
    parser.add_argument('--noisy_clf', type=str, default='GCN', help='Classifier')
    parser.add_argument('--clf', type=str, default='GCN', help='Classifier')
    parser.add_argument('--clf_dim', type=int, default=128, help='Classifier hideen dimension')
    parser.add_argument('--clfepochs', type=int, default=1000, help='Classifier Training epochs.')

    parser.add_argument('--heads', type=int, default=5, help='GAT Heads.')

    # Sampling settings
    parser.add_argument('--class_guide', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--classifier_scale', type=float, default=1, help='classifier scale for sampling.')

    parser.add_argument('--seed', type=int, default=1, help='Seed.')
    parser.add_argument('--fold', type=int, default=5, help='Test fold.')
    parser.add_argument('--gs', type=int, default=1, help='Number of augmented graphs for training.')
    parser.add_argument('--saveg', type=str, default="yes", help='save the generated graph? yes or no')

    parser.add_argument('--beta', type=float, default=1., help='Beta for balancing org and aug clf loss.')
    parser.add_argument('--tr', type=float, default=0.2, help='Train ratio.')

    parser.add_argument('--loc', type=str, default="nci", help='lab or nci')

    # early stop
    parser.add_argument('--patience', type=int, default=100, help='Patience epochs.')
    
    parser.add_argument('--runs', type=int, default=20, help='Random runs.')

    args = parser.parse_args()
    return args
