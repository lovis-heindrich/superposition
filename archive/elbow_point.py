# Taking the elbow point doesn't make much sense when you're taking the top 100 of 100,000 activations - you're already at the extreme end of the elbow section

# from collections import Counter


# def find_elbow_index(acts):
#     """Heuristic for finding the most likely elbow point in a line. Used to select important activations."""
#     n_acts = acts.shape[0]
#     all_coords = torch.vstack((torch.arange(n_acts).cuda(), acts))
#     first_point = all_coords[:, 0]
#     last_point = all_coords[:, -1]
#     # normalized line vector between the two points
#     line_vec = (last_point - first_point) / torch.sqrt(torch.sum((last_point - first_point)**2))
    
#     # Distance from each point to the line:
#     # vector between all points and first point
#     vec_from_first = all_coords - first_point[:, None]
#     # project vec_from_first onto the line
#     scalar_product = (vec_from_first * line_vec[:, None]).sum(dim=0)
#     vec_from_first_parallel = scalar_product * line_vec[:, None]
#     # find the perpendicular vector
#     vec_to_line = vec_from_first - vec_from_first_parallel
#     # distance to line is the norm of vec_to_line
#     dist_to_line = torch.sqrt((vec_to_line ** 2).sum(dim=0))
    
#     max_dist, elbow_index = torch.max(dist_to_line, dim=0)
#     return elbow_index


# def print_top_k_act_token_frequency_counts(layer, neuron, data, k=100, elbow=False):
#     top_acts, top_tokens, top_next_tokens = get_top_k_act_tokens_for_neuron(layer, neuron, data, k)
    
#     total = k
#     if elbow:
#         elbow_index = find_elbow_index(top_acts)
#         top_acts = top_acts[:elbow_index]
#         top_tokens = top_tokens[:elbow_index]
#         top_next_tokens = top_next_tokens[:elbow_index]
#         total = elbow_index.item() + 1

#     # Pretty print the tokens and next tokens associated with the highest neuron activations, and their frequencies
#     top_token_strings = model.tokenizer.convert_ids_to_tokens(top_tokens)
#     top_next_token_strings = model.tokenizer.convert_ids_to_tokens(top_next_tokens)

#     print(f"Percentage of the top {total} L{layer}N{neuron} activations triggered on token:")
#     current_tokens_percentage_dict = {key: str(round(value / total * 100, 2)) + '%' for key, value in Counter(top_token_strings).most_common()}
#     print(current_tokens_percentage_dict)
    
#     print(f"Percentage of the top {total} L{layer}N{neuron} activations triggered on token with next token:")
#     current_tokens_percentage_dict = {key: str(round(value / total * 100, 2)) + '%' for key, value in Counter(top_next_token_strings).most_common()}
#     print(current_tokens_percentage_dict, '\n')