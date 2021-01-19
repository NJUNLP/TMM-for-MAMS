import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pickle

import numpy
import sklearn.metrics
import torch
from tqdm import tqdm

import utils
from configs import RobertaConfigs
from dataset import Dataset
from instance import ACSAInstance
import torch.nn.functional as F
from module_ACSA import RobertaACSA



def train(epoches,configs,model_name):
    model=RobertaACSA(configs).to(configs.DEVICE)

    instances_train_raw = pickle.load(open('data/ACSA/train.pickle','rb'))
    instances_dev = pickle.load(open('data/ACSA/dev.pickle', 'rb'))

    instances_dev=utils.merge_instances_ACSA(instances_dev)
    devset=Dataset(instances_dev,configs.BATCH_SIZE,shuffle=False)

    parameters=set(model.parameters())
    roberta_parameters=set(model.roberta.parameters())

    optimizer = torch.optim.Adam([
        {'params': list(roberta_parameters), 'lr': 1e-5},
        {'params': list(parameters-roberta_parameters)}
    ], lr=1e-5)

    best_f1=0
    best_epoch=0
    for i in range(epoches):
        # shuffle order of aspects during training
        instances_train = utils.merge_instances_ACSA(instances_train_raw)
        trainset = Dataset(instances_train, configs.BATCH_SIZE, shuffle=True)

        for j in tqdm(range(trainset.batch_count)):
            batch=trainset.get_batch(j)

            predictions=model.forward_ACSA(batch)

            labels=[]
            for x in batch:
                labels+=x.polarity
            labels=torch.tensor(labels,device=configs.DEVICE)

            loss=F.cross_entropy(predictions, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        acc,precision,recall,f1=evaluate(model,devset)

        print('epoch:{},ACSA-accuracy:{},precision:{},recall:{},f1:{}'.format(i,acc, precision, recall, f1))

        if f1>best_f1:
            best_f1=f1
            best_epoch=i
            torch.save(model.state_dict(),model_name,pickle_protocol=pickle.HIGHEST_PROTOCOL)

    print('best f1:{},best epoch:{}'.format(best_f1,best_epoch))

def evaluate(model, dataset):
    model.eval()
    with torch.no_grad():
        predictions_all = []
        labels_all = []

        for i in range(dataset.batch_count):
            batch = dataset.get_batch(i)
            predictions = model.forward_ACSA(batch)
            predictions = torch.argmax(predictions, dim=1)
            predictions_all.append(predictions)

            for x in batch:
                labels_all+=x.polarity

        predictions_all = torch.cat(predictions_all).cpu().numpy()
        labels_all = numpy.array(labels_all)

        accuracy = sklearn.metrics.accuracy_score(labels_all, predictions_all)
        precision = sklearn.metrics.precision_score(labels_all, predictions_all, average='macro')
        recall = sklearn.metrics.recall_score(labels_all, predictions_all, average='macro')
        f1 = sklearn.metrics.f1_score(labels_all, predictions_all, average='macro')

    model.train()
    return accuracy, precision, recall, f1

def test(model_name,configs):
    instances_test=pickle.load(open('data/ACSA/test.pickle', 'rb'))
    instances_test=utils.merge_instances_ACSA(instances_test)
    testset=Dataset(instances_test,configs.BATCH_SIZE,shuffle=False)

    model=RobertaACSA(configs).to(configs.DEVICE)
    model.load_state_dict(torch.load(model_name))

    acc, precision, recall, f1 = evaluate(model, testset)
    print('ACSA-accuracy:{:.5f},precision:{:.5f},recall:{:.5f},f1:{:.5f}'.format(acc, precision, recall, f1))

if __name__=='__main__':
    configs=RobertaConfigs()
    model_name='roberta_acsa'

    train(30,configs,model_name)
    test(model_name,configs)
