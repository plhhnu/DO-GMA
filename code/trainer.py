import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from models import binary_cross_entropy
from prettytable import PrettyTable
from tqdm import tqdm
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
import pandas as pd

def remove_last_line(file_path):

    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = lines[:-1]
    with open(file_path, 'w') as file:
        file.writelines(lines)

class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, opt_da=None, discriminator=None,
                 experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        if opt_da:
            self.optim_da = opt_da
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]
        self.step = 0
        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}



        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
        train_metric_header = ["# Epoch", "Train_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.df_tps = None

        self.protein_feature_sequence=None
        self.drug_feature_sequence=None
        self.drug_feature_GCN=None

        self.best_score = dict(y_true=[], y_pred=[], y_score=[])

    def train(self, out_path=f"./result/"):
        os.makedirs(out_path, exist_ok=True)
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            print(i, "/100")
            self.current_epoch += 1
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc

                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))
            torch.save(self.best_model.state_dict(),
                       os.path.join(out_path, f"best_model.pth"))

        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision,recall,mcc_score,score = self.test(dataloader="test")

        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, thred_optim, test_loss]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " f1 " + str(f1) + " precision " + str(precision) + " recall " +
              str(recall) + " Accuracy " + str(accuracy) + " mcc " + str(mcc_score) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision

        self.test_metrics["recall"] = recall
        self.test_metrics["mcc_score"] = mcc_score

        score.to_csv(out_path + f"ture_pred_score.csv", index=False)
        self.save_result(out_path)
        AUCs=[auroc, auprc, accuracy, precision, recall, f1, mcc_score]
        return AUCs

    def save_result(self, out_path="./result/"):

        val_prettytable_file = os.path.join(out_path, f"valid_markdowntable.txt")
        test_prettytable_file = os.path.join(out_path, f"test_markdowntable.txt")
        train_prettytable_file = os.path.join(out_path, f"train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())


    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_d_2, v_p, labels) in enumerate(self.train_dataloader):
            self.step += 1
            v_d,v_d_2, v_p, labels = v_d.to(self.device),v_d_2.to(self.device), v_p.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad()
            v_d,v_d_2, v_p, f, score = self.model(v_d,v_d_2, v_p)
            n, loss = binary_cross_entropy(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch


    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        tys = dict(y_true=[], y_pred=[], y_score=[])
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()

            for i, (v_d,v_d_2,v_p, labels) in enumerate(data_loader):
                v_d,v_d_2, v_p, labels = v_d.to(self.device),v_d_2.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                if dataloader == "val":
                    v_d,v_d_2, v_p, f, score = self.model(v_d,v_d_2, v_p)
                elif dataloader == "test":
                    v_d,v_d_2, v_p, f, score = self.best_model(v_d,v_d_2, v_p)
                n, loss = binary_cross_entropy(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            precision1 = precision_score(y_label, y_pred_s)
            mcc_score = matthews_corrcoef(y_label, y_pred_s)
            recall = recall_score(y_label, y_pred_s)
            pred_result = {"y_true": y_label, "y_pred": y_pred_s, "y_score": y_pred}
            self.df_tps = pd.DataFrame(pred_result)


            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1,recall,mcc_score,self.df_tps
        else:
            return auroc, auprc, test_loss
