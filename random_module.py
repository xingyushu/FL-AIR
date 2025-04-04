
malicious_list = []
for i in range(int(args.num_users * args.malicious)):
        malicious_list.append(i)
    
all_list = list(set(range(args.num_users)))
benigh_list = list(set(range(args.num_users)) - set(malicious_list))


for iter in range(args.epochs):
        
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_updates = []

        # args.client_selection_args.random_seed_
          # Selection strategy: rounds 0-4 use 50% clients; round 5+ use 10% clients
        if iter < 5:
            m = int(0.5 * args.num_users)  # 50% of clients
        else:
            m = int(0.1 * args.num_users)  # 10% of clients

        # Set total number of selected clients per round
        if args.client_selection == 'random':
            # np.random.seed(args.random_seed)  # e.g., 1, 2, or 3 depending on your experiment if needed
            # Randomly select m clients from all users (both benign and malicious)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        elif args.client_selection == 'fixed':
            # For fixed selection, choose exactly m-4 benign clients and 4 malicious clients
            idxs_users = np.random.choice(benigh_list, m-4, replace=False)
            malicious_client = np.random.choice(malicious_list, 4, replace=False)
            # Combine benign and malicious client indices
            idxs_users = np.append(idxs_users, malicious_client)

'''
the remaning code
'''
