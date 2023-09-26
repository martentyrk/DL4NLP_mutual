def test(model, test_set, recall=1):
	"""
	Testing accuracies for different recalls

	"""
	acc_count = 0
	for sample in test_set:
		input_ids, attention_mask, token_type_ids = sample[0].input_ids, sample[0].attention_mask, sample[0].token_type_ids
		label = sample[1]
		pred = model(input_ids, attention_mask, token_type_ids, label)
		indices = np.argpartition(pred.logits.detach().numpy()[0], recall)[-recall:]
		acc_count += label in indices
	return acc_count/len(test_set)