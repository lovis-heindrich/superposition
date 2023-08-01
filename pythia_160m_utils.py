import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics
import haystack_utils

### Sparse probing utils

def run_single_neuron_lr(layer, neuron, german_activations, english_activations, num_samples=5000, ):
    """For German context neurons"""
    # Check accuracy of logistic regression
    A = torch.concat([german_activations[layer][:num_samples, neuron], english_activations[layer][:num_samples, neuron]]).view(-1, 1).cpu().numpy()
    y = torch.concat([torch.ones(num_samples), torch.zeros(num_samples)]).cpu().numpy()
    A_train, A_test, y_train, y_test = train_test_split(A, y, test_size=0.2)
    lr_model = LogisticRegression()
    lr_model.fit(A_train, y_train)
    test_acc = lr_model.score(A_test, y_test)
    train_acc = lr_model.score(A_train, y_train)
    f1 = sklearn.metrics.f1_score(y_test, lr_model.predict(A_test))
    return train_acc, test_acc, f1
    
def get_neuron_accuracy(layer, neuron, german_activations, english_activations, plot=False, print_f1s=True):
    """For German context neurons"""
    mean_english_activation = english_activations[layer][:,neuron].mean()
    mean_german_activation = german_activations[layer][:,neuron].mean()
    
    if plot:
        haystack_utils.two_histogram(english_activations[layer][:,neuron], german_activations[layer][:,neuron], "English", "German", "Activation", "Frequency", f"L{layer}N{neuron} activations on English vs German text")
    train_acc, test_acc, f1 = run_single_neuron_lr(layer, neuron, german_activations=german_activations, english_activations=english_activations)
    if print_f1s:
        print(f"\nL{layer}N{neuron}: F1={f1:.2f}, Train acc={train_acc:.2f}, and test acc={test_acc:.2f}")
        print(f"Mean activation English={mean_english_activation:.2f}, German={mean_german_activation:.2f}")
    return f1

def ablation_effect(model, data, fwd_hooks):
    """Full ablation accuracy"""
    original_losses = []
    ablated_losses = []
    batch_size = 50
    for i in range(4):
        original_losses.append(model(data[i * batch_size:i * batch_size + 50], return_type='loss').cpu())
        with model.hooks(fwd_hooks):
            ablated_losses.append(model(data[i * batch_size:i * batch_size + 50], return_type='loss').cpu())

    original_loss = sum(original_losses) / len(original_losses)
    ablated_loss = sum(ablated_losses) / len(ablated_losses)

    print(original_loss, ablated_loss)
    print(f'{(ablated_loss - original_loss) / original_loss * 100:2f}% loss increase')