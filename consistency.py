def consistency_defense(w_updates, w_glob, net_glob, args, iter):
    """
    New defense based on consistency between model directions using DBSCAN clustering.
    
    Steps:
      1. Convert each local update into a 1D vector.
      2. Perform DBSCAN clustering using the cosine metric.
      3. Identify the largest cluster (ignoring noise) as the benign group.
      4. Aggregate the updates from that cluster using a weighted aggregation.
      5. If previous aggregates exist, compare the current majority aggregate with the previous one (and with
         the minority aggregate if available) using cosine similarity. If the cross similarity is higher than
         the intra-majority similarity, revert to the previous global model.
      6. Otherwise, accept the current majority aggregate as the new global model.
    
    Returns:
      The new global model after aggregation.
    """
    num_clients = len(w_updates)
    
    # Convert each update into a 1D vector.
    update_vectors = [parameters_dict_to_vector_flt(update) for update in w_updates]
    # Convert to numpy array.
    update_vectors_np = np.stack([vec.cpu().numpy() for vec in update_vectors])
    
    # Use DBSCAN clustering with cosine metric.
    # eps may need tuning depending on your data; here we use 0.3 and min_samples=2.
    dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(update_vectors_np)
    labels = dbscan.labels_
    
    # Exclude noise points (label == -1).
    valid_labels = [label for label in labels if label != -1]
    if len(valid_labels) == 0:
        print("DBSCAN found no clusters; falling back to FLTrust aggregation on all updates.")
        return fltrust(w_updates, w_glob, w_glob, args)
    
    # Count the occurrences of each valid label.
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    # Select the label corresponding to the largest cluster.
    majority_cluster = unique_labels[np.argmax(counts)]
    
    # Get indices for the majority cluster and the remaining (non-noise, non-majority) points.
    majority_indices = [i for i, label in enumerate(labels) if label == majority_cluster]
    minority_indices = [i for i, label in enumerate(labels) if label != majority_cluster and label != -1]
    
    # Aggregate updates using FLTrust-style weighted aggregation.
    majority_updates = [w_updates[i] for i in majority_indices]
    majority_agg = fltrust(majority_updates, w_glob, w_glob, args)
    
    if len(minority_indices) > 0:
        minority_updates = [w_updates[i] for i in minority_indices]
        minority_agg = fltrust(minority_updates, w_glob, w_glob, args)
    else:
        minority_agg = None
    
    # If this is the first round applying consistency defense, store aggregates and return the majority aggregate.
    if not hasattr(args, 'prev_majority'):
        args.prev_majority = majority_agg
        args.prev_minority = minority_agg
        args.prev_global = w_glob
        return majority_agg
    
    # Compute cosine similarities between stored and current aggregates.
    prev_maj_vector = parameters_dict_to_vector_flt(args.prev_majority)
    curr_maj_vector = parameters_dict_to_vector_flt(majority_agg)
    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_maj_maj = cos_sim(prev_maj_vector, curr_maj_vector)
    
    if minority_agg is not None and hasattr(args, 'prev_minority') and args.prev_minority is not None:
        prev_min_vector = parameters_dict_to_vector_flt(args.prev_minority)
        curr_min_vector = parameters_dict_to_vector_flt(minority_agg)
        cos_maj_min = cos_sim(prev_maj_vector, curr_min_vector)
        cos_min_maj = cos_sim(prev_min_vector, curr_maj_vector)
        # If either cross similarity exceeds the intra-majority similarity, revert to the previous global model.
        if cos_maj_min > cos_maj_maj or cos_min_maj > cos_maj_maj:
            print("Consistency defense (DBSCAN) triggered: reverting to previous global model.")
            return args.prev_global
        else:
            # Update stored aggregates and accept the current majority aggregate.
            args.prev_majority = majority_agg
            args.prev_minority = minority_agg
            args.prev_global = majority_agg
            return majority_agg
    else:
        # If no minority aggregate is available, update stored aggregates and return the majority aggregate.
        args.prev_majority = majority_agg
        args.prev_minority = minority_agg
        args.prev_global = majority_agg
        return majority_agg
