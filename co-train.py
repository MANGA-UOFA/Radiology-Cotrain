import os
import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
# from torch.optim import AdamW
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
from sklearn.metrics import (
    confusion_matrix,
    auc,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
)
import math
import sys
from dateutil import tz

from dataset import RadiologyLabeledDataset, RadiologyUnlabeledDataset
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.encoding = "UTF-8"

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush


class CoTrain:
    def __init__(
        self,
        model_name: str,
        logdir: str,
        train_df,
        test_df,
        val_df,
        view1_name: str,
        view2_name: str,
        unlabeled_df,
        max_length: int,
        batch_size: int,
        num_classes: int,
        learning_rate: float,
        num_epochs: int,
        target: str,
        cotrain_steps: int,
        init_coverage: float,
        # all_results_path: str,
    ):
        tzone = tz.gettz("America/Edmonton")
        self.timestamp = (
            datetime.datetime.now().astimezone(tzone).strftime("%Y-%m-%d_%H:%M:%S")
        )

        self.model_name = model_name
        self.view1 = view1_name
        self.view2 = view2_name

        # writer to log information:
        self.logdir = logdir
        self.writer = SummaryWriter(self.logdir)
        self.logger = Logger(os.path.join(
            self.logdir, self.timestamp + ".log"))
        sys.stdout = self.logger
        sys.stderr = self.logger
        self.results_df = pd.DataFrame(columns=["Ensemble accuracy",
                                                "View1 accuracy",
                                                "View2 accuracy",
                                                "Cotrain step",
                                                'Val/Test'])

        # assign all of the various data sets that we need
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.unlabeled_df = unlabeled_df

        # Set training parameters of each separate model:
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.target = target

        # Set co-training parameters:
        self.cotrain_steps = cotrain_steps
        self.init_coverage = init_coverage

        # Track global step counts for metrics
        self.findings_global_step = 0
        self.impressions_global_step = 0

        # Load datasets
        self.view1_test_dataloader = self.load_dataset(
            section=self.view1,
            df=self.test_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )

        self.view2_test_dataloader = self.load_dataset(
            section=self.view2,
            df=self.test_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )

        self.view1_val_dataloader = self.load_dataset(
            section=self.view1,
            df=self.val_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )

        self.view2_val_dataloader = self.load_dataset(
            section=self.view2,
            df=self.val_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )

        self.view1_unlabeled_dataloader = self.load_dataset(
            section=self.view1,
            df=self.unlabeled_df.reset_index(drop=True),
            labeled=False,
            other_section=self.view2,
        )

        self.view2_unlabeled_dataloader = self.load_dataset(
            section=self.view2,
            df=self.unlabeled_df.reset_index(drop=True),
            labeled=False,
            other_section=self.view1,
        )

        # Both views are initialized from the same model
        self.init_model1 = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,  # The number of output labels
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
        ).cuda()

        self.init_model2 = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,  # The number of output labels
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
        ).cuda()

        # Set optimizers for each view:
        self.view1_optimizer = AdamW(
            self.init_model1.parameters(), lr=self.learning_rate)
        self.view2_optimizer = AdamW(
            self.init_model2.parameters(), lr=self.learning_rate)

    def load_dataset(self, section, df, labeled=True, other_section="", shuffle=True):
        # load data into dataloader + tokenize
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name)
        if labeled:
            dataset = RadiologyLabeledDataset(
                tokenizer,
                max_length=self.max_length,
                df=df,
                target=self.target,
                view_name=section,
            )
            dataloader = DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=shuffle,
            )
        else:
            dataset = RadiologyUnlabeledDataset(
                tokenizer,
                max_length=self.max_length,
                df=df,
                view_name=section,
                other_view_name=other_section,
            )

            dataloader = DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=False
            )
        return dataloader

    def save_checkpoint(self, model, optimizer, step, section):
        filename = section + "_" + "cotrain_step" + \
            str(step) + "_" + self.target + ".pt"
        torch.save(
            {
                "cotrain_step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(self.logdir, filename),
        )

    def resume_from_checkpoint(self, step, section):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )

        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        filename = section + "_" + "cotrain_step" + \
            str(step) + "_" + self.target + ".pt"

        checkpoint = torch.load(os.path.join(self.logdir, filename))

        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

    def train(self, model, val_dataloader, df, section, optimizer, epochs, step):

        dataloader = self.load_dataset(section, df.reset_index(drop=True))

        total_steps = len(dataloader) * epochs

        last_epoch_accuracy = 0

        for epoch_i in range(self.num_epochs):
            model.train()
            # For each batch of training data...
            for i, batch in enumerate(dataloader):
                # Unpack this training batch from our dataloader.
                b_input_ids = batch["ids"].cuda()
                b_input_mask = batch["mask"].cuda()
                b_labels = batch["target"].cuda()

                model.zero_grad()

                result = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )

                loss = result.loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                if section == "impressions":
                    self.impressions_global_step += 1
                    title = "Loss_" + section
                    self.writer.add_scalar(
                        title, loss, self.impressions_global_step
                    )

                else:
                    self.findings_global_step += 1
                    title = "Loss_" + section
                    self.writer.add_scalar(
                        title, loss, self.findings_global_step)

            sample_f1, _, _, _ = self.eval(
                model=model, dataloader=val_dataloader)
            if sample_f1 < last_epoch_accuracy:
                print("The best epoch is", epoch_i-1,
                      "for step", step, "of cotraining")
                break

            self.save_checkpoint(model, optimizer, step, section)
            last_epoch_accuracy = sample_f1

        model, optimizer = self.resume_from_checkpoint(step, section)

        return model, optimizer

    def get_conf_data(
        self,
        model_inf,
        model_trn,
        dataloader_inf,
        dataloader_trn,
        coverage,
        section,
        other_section,
        step,
    ):
        softmax = nn.Softmax(dim=-1)

        model_inf.eval()
        model_trn.eval()
        # the df created must have the same y column and also
        df_inf = pd.DataFrame(
            columns=["File Name", "probability_inf",
                     other_section, self.target]
        )
        for i, batch in enumerate(dataloader_inf):
            # Unpack this training batch from our dataloader.
            b_input_ids = batch["ids"].cuda()
            b_input_mask = batch["mask"].cuda()
            b_file_name = list(batch["file"][0])
            # for unlabeled data, there is an extra column where the other view is also returned
            b_other_view = list(batch["other_view"][0])

            with torch.no_grad():
                result_inf = model_inf(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    return_dict=True,
                )

            probabilities_inf = (
                np.max(softmax(result_inf.logits).detach().cpu().numpy(), axis=1)
                .flatten()
                .tolist()
            )

            # we only need the labels for the inference model...because we will pick the ones that are somewhat contrastrive to them...
            labels_inf = (
                np.argmax(
                    softmax(result_inf.logits).detach().cpu().numpy(), axis=1)
                .flatten()
                .tolist()
            )

            new_batch = pd.DataFrame(
                {
                    "File Name": b_file_name,
                    "probability_inf": probabilities_inf,
                    other_section: b_other_view,
                    self.target: labels_inf,
                }
            )

            df_inf = pd.concat([df_inf, new_batch])

        # the df created must have the same y column and also
        df_trn = pd.DataFrame(
            columns=["File Name", "probability_trn", self.target + "_trn"]
        )

        for i, batch in enumerate(dataloader_trn):
            # Unpack this training batch from our dataloader.
            b_input_ids = batch["ids"].cuda()
            b_input_mask = batch["mask"].cuda()
            b_file_name = list(batch["file"][0])

            with torch.no_grad():
                result_trn = model_trn(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    return_dict=True,
                )

            # change logits into probability scores and then save them with the file ids...this way we can
            # then use these file ids to output the unlabeled df with the confident scores
            probabilities_trn = (
                np.max(softmax(result_trn.logits).detach().cpu().numpy(), axis=1)
                .flatten()
                .tolist()
            )

            labels_trn = (
                np.argmax(
                    softmax(result_trn.logits).detach().cpu().numpy(), axis=1)
                .flatten()
                .tolist()
            )

            new_batch = pd.DataFrame(
                {
                    "File Name": b_file_name,
                    "probability_trn": probabilities_trn,
                    self.target + "_trn": labels_trn,
                }
            )

            df_trn = pd.concat([df_trn, new_batch])

        # merge the two df together
        df = df_inf.merge(df_trn, on=["File Name"], how="inner")

        df_agreement = df[(df[self.target] == df[self.target + "_trn"])]

        min_size = 10000
        for i in range(self.num_classes):
            nrows = df_agreement[df_agreement[self.target] == i].shape[0]
            if nrows < min_size:
                min_size = nrows

        sample_size = math.ceil(min_size*coverage)
        conf_df = pd.DataFrame(columns=list(df_agreement.columns.values))
        for i in range(self.num_classes):
            df_sort = df_agreement[df_agreement[self.target] == i].sort_values(
                by=["probability_inf"], ascending=False)
            subset_df = df_sort.iloc[:sample_size, :]
            conf_df = pd.concat([conf_df, subset_df])

        return conf_df[["File Name", other_section, self.target]]

    def eval(self, model, dataloader, save=True):

        # softmax function that we need for metric calculations:
        softmax = nn.Softmax(dim=-1)

        # store the prob, preds and labels
        probs = np.zeros((0, self.num_classes))
        preds = []
        labels = []
        file_names = []

        model.eval()
        for i, batch in enumerate(dataloader):
            b_input_ids = batch["ids"].cuda()
            b_input_mask = batch["mask"].cuda()
            b_labels = batch["target"].cuda()
            b_file_name = list(batch["file"][0])

            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                result = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )

            logits = result.logits

            # Transform probabilities and labels to a list so that we can use them to calculate auroc, auprc, other metrics
            probabilities = (softmax(logits).detach().cpu().numpy())
            predictions = (
                np.argmax(logits.detach().cpu().numpy(),
                          axis=1).flatten().tolist()
            )
            label_ids = b_labels.cpu().numpy().flatten().tolist()

            probs = np.concatenate((probs, probabilities), axis=0)
            preds += predictions
            labels += label_ids
            file_names += b_file_name

        model_pred_df = pd.DataFrame(
            {"File Name": file_names, "Labels": labels, "Predicted": preds})

        accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
        print("Accuracy:", accuracy)
        return accuracy, probs, labels, file_names

    def ensemble_eval(
        self, view1_model, view2_model, dataloader1, dataloader2, step, type_df
    ):
        # softmax function that we need for metric calculations:
        softmax = nn.Softmax(dim=-1)

        # store the prob, preds and labels
        view1_acc, view1_probs, view1_labels, view1_file_names = self.eval(
            view1_model, dataloader1)
        view2_acc, view2_probs, view2_labels, view2_file_names = self.eval(
            view2_model, dataloader2)

        # first check if the view1 and view2 labels are the same:
        if view1_file_names == view2_file_names:
            print("The files names are the same")
        else:
            print("Check code")

        # combine the probabilities together
        # since they are for the 1 label, then if avg_prob < 0.5 then we choose 0, else we choose 1 for the pred_label
        probabilities = np.add(view1_probs, view2_probs)
        avg_probs = np.divide(probabilities, 2)
        pred_labels = np.argmax(avg_probs, axis=1)
        model_pred_df = pd.DataFrame(
            {"File Name": view1_file_names, "Labels": view1_labels, "Predicted": pred_labels})

        print("Results for ensemble")

        # accuracy:
        ensemble_accuracy = np.sum(np.array(pred_labels) == np.array(view1_labels)) / len(
            view1_labels
        )
        print("Accuracy : {0: .6f} ".format(ensemble_accuracy))
        self.writer.add_scalar("Accuracy" + "_Ensemble" +
                               "_"+type_df, ensemble_accuracy, step)

        return ensemble_accuracy, view1_acc, view2_acc

    def record_results(self, view1_model, view2_model, dataloader1, dataloader2, step, type='test'):

        ensemble_acc, view1_acc, view2_acc = self.ensemble_eval(
            view1_model=view1_model,
            view2_model=view2_model,
            dataloader1=dataloader1,
            dataloader2=dataloader2,
            step=step,
            type_df=type,
        )

        # record save the file:
        results = pd.DataFrame({
            "Ensemble accuracy": ensemble_acc,
            "View1 accuracy": view1_acc,
            "View2 accuracy": view2_acc,
            "Cotrain step": step,
            'Val/Test': type,
        }, index=[0])

        self.results_df = pd.concat([self.results_df, results])
        self.results_df.to_csv(os.path.join(
            self.logdir,  self.target + "_" + 'results.csv'))

    def cotrain(self, seed):

        print("Finetuning the models")
        print("Shape of train_df:", self.train_df.shape)

        # Finetune each view
        view1_model, _ = self.train(
            model=self.init_model1,
            df=self.train_df,
            val_dataloader=self.view1_val_dataloader,
            section=self.view1,
            optimizer=self.view1_optimizer,
            epochs=self.num_epochs,
            step=0,
        )

        view2_model, _ = self.train(
            model=self.init_model2,
            df=self.train_df,
            val_dataloader=self.view2_val_dataloader,
            section=self.view2,
            optimizer=self.view2_optimizer,
            epochs=self.num_epochs,
            step=0,
        )

        print("Cotraining begins")

        view1_df = self.train_df[["File Name", self.view1, self.target]]
        view2_df = self.train_df[["File Name", self.view2, self.target]]

        self.record_results(
            view1_model=view1_model,
            view2_model=view2_model,
            dataloader1=self.view1_test_dataloader,
            dataloader2=self.view2_test_dataloader,
            step=0, type='test')

        # Co-train each view:
        for i in range(1, self.cotrain_steps + 1):
            print("cotrain step", i)
            coverage = self.init_coverage

            view1_labeled = self.get_conf_data(
                model_inf=view1_model,
                model_trn=view2_model,
                dataloader_inf=self.view1_unlabeled_dataloader,
                dataloader_trn=self.view2_unlabeled_dataloader,
                section=self.view1,
                coverage=coverage,
                other_section=self.view2,
                step=i,
            )

            # append it to the train df
            # create a subset of the train df
            view2_train = pd.concat([view2_df, view1_labeled])

            view2_model, _ = self.train(
                model=self.init_model2,
                df=view2_train,
                val_dataloader=self.view2_val_dataloader,
                section=self.view2,
                optimizer=self.view2_optimizer,
                step=i,
                epochs=self.num_epochs,
            )

            view2_labeled = self.get_conf_data(
                model_inf=view2_model,
                model_trn=view1_model,
                dataloader_inf=self.view2_unlabeled_dataloader,
                dataloader_trn=self.view1_unlabeled_dataloader,
                section=self.view2,
                coverage=coverage,
                other_section=self.view1,
                step=i,
            )

            view1_train = pd.concat([view1_df, view2_labeled])

            view1_model, _ = self.train(
                model=self.init_model1,
                df=view1_train,
                val_dataloader=self.view1_val_dataloader,
                section=self.view1,
                optimizer=self.view1_optimizer,
                step=i,
                epochs=self.num_epochs,
            )

            print("The {0:.0f} cotrain step".format(i))

            # ensemble
            self.record_results(
                view1_model=view1_model,
                view2_model=view2_model,
                dataloader1=self.view1_val_dataloader,
                dataloader2=self.view2_val_dataloader,
                step=i, type='val')

            self.record_results(
                view1_model=view1_model,
                view2_model=view2_model,
                dataloader1=self.view1_test_dataloader,
                dataloader2=self.view2_test_dataloader,
                step=0, type='test')

        return


if __name__ == "__main__":
    # parse some arguments that are needed
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--labeled-pickle",
        type=str,
        help="Stores the cleaned labeled data. If store-data is activated then this argument sets where the new pkl file will be stored",
        required=True,
    )

    argparser.add_argument(
        "--unlabeled-pickle",
        type=str,
        help="Stores the cleaned unlabeled data. If store-data is activated then this argument sets where the new pkl file will be stored",
        required=True,
    )

    argparser.add_argument(
        "--max-length",
        type=int,
        help="The max number of tokens per sequence",
        default=512,
    )
    argparser.add_argument("--random-seed", type=int, default=0)
    argparser.add_argument("--test-split", type=float, default=0.4)

    # Model Settings:
    argparser.add_argument("--model-name", type=str, required=True)
    argparser.add_argument("--view1-name", type=str, required=True)
    argparser.add_argument("--view2-name", type=str, required=True)

    # Training
    argparser.add_argument("--batch-size", type=int, default=16)
    argparser.add_argument("--num_classes", type=int, default=2)
    argparser.add_argument("--learning-rate", type=float, default=5e-5)
    argparser.add_argument("--num-epochs", type=int, default=3)
    argparser.add_argument("--target", type=str, required=True)

    # Cotraining
    argparser.add_argument("--cotrain-steps", type=int, default=5)
    argparser.add_argument("--init-coverage", type=float, default=0.25)
    argparser.add_argument("--unlabeled-size", type=int,
                           default=10000, help="1k, 3k,5k, 10k")

    # Path to tensorboard and csv of results:
    argparser.add_argument(
        "--logdir",
        type=str,
        default="log/",
        help="Path to save results to",
    )

    # Overall results:
    argparser.add_argument(
        "--all-results-path",
        type=str,
        default="cotrain.csv",
        help="Path to save best cotrain step results"
    )

    args = argparser.parse_args()

    # Set device:
    device = torch.device("cuda")

    # load pd.dataframe
    labeled_df = pd.read_pickle(args.labeled_pickle)
    unlabeled_df = pd.read_pickle(args.unlabeled_pickle)
    unlabeled_df = unlabeled_df.iloc[:args.unlabeled_size, :]

    torch.cuda.empty_cache()

    kf = KFold(n_splits=5, random_state=0, shuffle=True)

    run = 0

    for train_index, test_index in kf.split(labeled_df):
        print("Run" + str(run) + "for random seed", +  args.random_seed)
        df_tr_va, df_test = labeled_df.iloc[train_index], labeled_df.iloc[test_index]
        train_df, val_df = train_test_split(
            df_tr_va, test_size=0.25, random_state=0)
        test_df = labeled_df.iloc[test_index]

        # Set the seed value all over the place to make this reproducible.
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

        os.makedirs(args.logdir+"_"+str(run), exist_ok=True)

        # initialize co-training model:
        method = CoTrain(
            model_name=args.model_name,
            view1_name=args.view1_name,
            view2_name=args.view2_name,
            logdir=args.logdir+"_"+str(run),
            train_df=train_df,
            test_df=test_df,
            val_df=val_df,
            unlabeled_df=unlabeled_df.iloc[:args.unlabeled_size, :],
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            target=args.target,
            cotrain_steps=args.cotrain_steps,
            init_coverage=args.init_coverage,
        )

        method.cotrain(seed=args.random_seed)

        run += 1
