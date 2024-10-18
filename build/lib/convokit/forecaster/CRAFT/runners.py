try:
    import torch
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "torch is not currently installed. Run 'pip install convokit[craft]' if you would like to use the CRAFT model."
    )

import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
from copy import deepcopy


class Predictor(nn.Module):
    """This helper module encapsulates the CRAFT pipeline, defining the logic of passing an input through each consecutive sub-module."""

    def __init__(self, encoder, context_encoder, classifier):
        super(Predictor, self).__init__()
        self.encoder = encoder
        self.context_encoder = context_encoder
        self.classifier = classifier

    def forward(
        self,
        input_batch,
        dialog_lengths,
        dialog_lengths_list,
        utt_lengths,
        batch_indices,
        dialog_indices,
        batch_size,
        max_length,
    ):
        # Forward input through encoder model
        _, utt_encoder_hidden = self.encoder(input_batch, utt_lengths)

        # Convert utterance encoder final states to batched dialogs for use by context encoder
        context_encoder_input = makeContextEncoderInput(
            utt_encoder_hidden, dialog_lengths_list, batch_size, batch_indices, dialog_indices
        )

        # Forward pass through context encoder
        context_encoder_outputs, context_encoder_hidden = self.context_encoder(
            context_encoder_input, dialog_lengths
        )

        # Forward pass through classifier to get prediction logits
        logits = self.classifier(context_encoder_outputs, dialog_lengths)

        # Apply sigmoid activation
        predictions = F.sigmoid(logits)
        return predictions


def makeContextEncoderInput(
    utt_encoder_hidden, dialog_lengths, batch_size, batch_indices, dialog_indices
):
    # first, sum the forward and backward encoder states
    utt_encoder_summed = utt_encoder_hidden[-2, :, :] + utt_encoder_hidden[-1, :, :]
    # we now have hidden state of shape [utterance_batch_size, hidden_size]
    # split it into a list of [hidden_size,] x utterance_batch_size
    last_states = [t.squeeze() for t in utt_encoder_summed.split(1, dim=0)]

    # create a placeholder list of tensors to group the states by source dialog
    states_dialog_batched = [[None for _ in range(dialog_lengths[i])] for i in range(batch_size)]

    # group the states by source dialog
    for hidden_state, batch_idx, dialog_idx in zip(last_states, batch_indices, dialog_indices):
        states_dialog_batched[batch_idx][dialog_idx] = hidden_state

    # stack each dialog into a tensor of shape [dialog_length, hidden_size]
    states_dialog_batched = [torch.stack(d) for d in states_dialog_batched]

    # finally, condense all the dialog tensors into a single zero-padded tensor
    # of shape [max_dialog_length, batch_size, hidden_size]
    return torch.nn.utils.rnn.pad_sequence(states_dialog_batched)


def train(
    input_variable,
    dialog_lengths,
    dialog_lengths_list,
    utt_lengths,
    batch_indices,
    dialog_indices,
    labels,  # input/output arguments
    encoder,
    context_encoder,
    attack_clf,  # network arguments
    encoder_optimizer,
    context_encoder_optimizer,
    attack_clf_optimizer,  # optimization arguments
    batch_size,
    clip,
    device,
):  # misc arguments
    # Zero gradients
    encoder_optimizer.zero_grad()
    context_encoder_optimizer.zero_grad()
    attack_clf_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    labels = labels.to(device)

    # Forward pass through utterance encoder
    _, utt_encoder_hidden = encoder(input_variable, utt_lengths)

    # Convert utterance encoder final states to batched dialogs for use by context encoder
    context_encoder_input = makeContextEncoderInput(
        utt_encoder_hidden, dialog_lengths_list, batch_size, batch_indices, dialog_indices
    )

    # Forward pass through context encoder
    context_encoder_outputs, _ = context_encoder(context_encoder_input, dialog_lengths)

    # Forward pass through classifier to get prediction logits
    logits = attack_clf(context_encoder_outputs, dialog_lengths)

    # Calculate loss
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(attack_clf.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    context_encoder_optimizer.step()
    attack_clf_optimizer.step()

    return loss.item()


def evaluateBatch(
    encoder,
    context_encoder,
    predictor,
    voc,
    input_batch,
    dialog_lengths,
    dialog_lengths_list,
    utt_lengths,
    batch_indices,
    dialog_indices,
    batch_size,
    device,
    max_length,
    threshold=0.5,
):
    # Set device options
    input_batch = input_batch.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    # Predict future attack using predictor
    scores = predictor(
        input_batch,
        dialog_lengths,
        dialog_lengths_list,
        utt_lengths,
        batch_indices,
        dialog_indices,
        batch_size,
        max_length,
    )
    predictions = (scores > threshold).float()
    return predictions, scores


def validate(
    dataset,
    encoder,
    context_encoder,
    predictor,
    voc,
    batch_size,
    device,
    max_length,
    batch_iterator_func,
):
    # create a batch iterator for the given data
    batch_iterator = batch_iterator_func(voc, dataset, batch_size, shuffle=False)
    # find out how many iterations we will need to cover the whole dataset
    n_iters = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)
    # containers for full prediction results so we can compute accuracy at the end
    all_preds = []
    all_labels = []
    for iteration in range(1, n_iters + 1):
        batch, batch_dialogs, _, true_batch_size = next(batch_iterator)
        # Extract fields from batch
        (
            input_variable,
            dialog_lengths,
            utt_lengths,
            batch_indices,
            dialog_indices,
            labels,
            convo_ids,
            target_variable,
            mask,
            max_target_len,
        ) = batch
        dialog_lengths_list = [len(x) for x in batch_dialogs]
        # run the model
        predictions, scores = evaluateBatch(
            encoder,
            context_encoder,
            predictor,
            voc,
            input_variable,
            dialog_lengths,
            dialog_lengths_list,
            utt_lengths,
            batch_indices,
            dialog_indices,
            true_batch_size,
            device,
            max_length,
        )
        # aggregate results for computing accuracy at the end
        all_preds += [p.item() for p in predictions]
        all_labels += [l.item() for l in labels]
        print(
            "Iteration: {}; Percent complete: {:.1f}%".format(iteration, iteration / n_iters * 100)
        )

    # compute and return the accuracy
    return (np.asarray(all_preds) == np.asarray(all_labels)).mean()


def trainIters(
    voc,
    pairs,
    val_pairs,
    encoder,
    context_encoder,
    attack_clf,
    encoder_optimizer,
    context_encoder_optimizer,
    attack_clf_optimizer,
    embedding,
    n_iteration,
    batch_size,
    print_every,
    validate_every,
    clip,
    device,
    max_length,
    batch_iterator_func,
):
    # create a batch iterator for training data
    batch_iterator = batch_iterator_func(voc, pairs, batch_size)

    # Initializations
    print("Initializing ...")
    start_iteration = 1
    print_loss = 0

    # Training loop
    print("Training...")
    # keep track of best validation accuracy - only save when we have a model that beats the current best
    best_acc = 0
    best_model = None
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch, training_dialogs, _, true_batch_size = next(batch_iterator)
        # Extract fields from batch
        (
            input_variable,
            dialog_lengths,
            utt_lengths,
            batch_indices,
            dialog_indices,
            labels,
            _,
            target_variable,
            mask,
            max_target_len,
        ) = training_batch
        dialog_lengths_list = [len(x) for x in training_dialogs]

        # Run a training iteration with batch
        loss = train(
            input_variable,
            dialog_lengths,
            dialog_lengths_list,
            utt_lengths,
            batch_indices,
            dialog_indices,
            labels,  # input/output arguments
            encoder,
            context_encoder,
            attack_clf,  # network arguments
            encoder_optimizer,
            context_encoder_optimizer,
            attack_clf_optimizer,  # optimization arguments
            true_batch_size,
            clip,
            device,
        )  # misc arguments
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(
                "Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                    iteration, iteration / n_iteration * 100, print_loss_avg
                )
            )
            print_loss = 0

        # Evaluate on validation set
        if iteration % validate_every == 0:
            print("Validating!")
            # put the network components into evaluation mode
            encoder.eval()
            context_encoder.eval()
            attack_clf.eval()

            predictor = Predictor(encoder, context_encoder, attack_clf)
            accuracy = validate(
                val_pairs,
                encoder,
                context_encoder,
                predictor,
                voc,
                batch_size,
                device,
                max_length,
                batch_iterator_func,
            )
            print("Validation set accuracy: {:.2f}%".format(accuracy * 100))

            # keep track of our best model so far
            if accuracy > best_acc:
                print("Validation accuracy better than current best; saving model...")
                best_acc = accuracy
                best_model = deepcopy(
                    {
                        "iteration": iteration,
                        "en": encoder.state_dict(),
                        "ctx": context_encoder.state_dict(),
                        "atk_clf": attack_clf.state_dict(),
                        "en_opt": encoder_optimizer.state_dict(),
                        "ctx_opt": context_encoder_optimizer.state_dict(),
                        "atk_clf_opt": attack_clf_optimizer.state_dict(),
                        "loss": loss,
                        "voc_dict": voc.__dict__,
                        "embedding": embedding.state_dict(),
                    }
                )

            # put the network components back into training mode
            encoder.train()
            context_encoder.train()
            attack_clf.train()

    return best_model


def evaluateDataset(
    dataset,
    encoder,
    context_encoder,
    predictor,
    voc,
    batch_size,
    device,
    max_length,
    batch_iterator_func,
    threshold,
    pred_col_name,
    score_col_name,
):
    # create a batch iterator for the given data
    batch_iterator = batch_iterator_func(voc, dataset, batch_size, shuffle=False)
    # find out how many iterations we will need to cover the whole dataset
    n_iters = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)
    output_df = {"id": [], pred_col_name: [], score_col_name: []}
    for iteration in range(1, n_iters + 1):
        batch, batch_dialogs, _, true_batch_size = next(batch_iterator)
        # Extract fields from batch
        (
            input_variable,
            dialog_lengths,
            utt_lengths,
            batch_indices,
            dialog_indices,
            labels,
            convo_ids,
            target_variable,
            mask,
            max_target_len,
        ) = batch
        dialog_lengths_list = [len(x) for x in batch_dialogs]
        # run the model
        predictions, scores = evaluateBatch(
            encoder,
            context_encoder,
            predictor,
            voc,
            input_variable,
            dialog_lengths,
            dialog_lengths_list,
            utt_lengths,
            batch_indices,
            dialog_indices,
            true_batch_size,
            device,
            max_length,
            threshold,
        )

        # format the output as a dataframe (which we can later re-join with the corpus)
        for i in range(true_batch_size):
            convo_id = convo_ids[i]
            pred = predictions[i].item()
            score = scores[i].item()
            output_df["id"].append(convo_id)
            output_df[pred_col_name].append(pred)
            output_df[score_col_name].append(score)

        print(
            "Iteration: {}; Percent complete: {:.1f}%".format(iteration, iteration / n_iters * 100)
        )

    return pd.DataFrame(output_df).set_index("id")
