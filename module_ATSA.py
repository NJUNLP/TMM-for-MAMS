import torch.nn
import torch.hub
import torch.nn.functional as F

from fairseq.models.roberta import RobertaModel
import re

class RobertaATSA(torch.nn.Module):
    def __init__(self,configs):
        super(RobertaATSA, self).__init__()

        self.configs=configs
        self.roberta=RobertaModel.from_pretrained('pretrained/roberta.large', checkpoint_file='model.pt')

        self.linear_hidden=torch.nn.Linear(configs.ROBERTA_DIM,configs.LINEAR_HIDDEN_DIM)
        self.linear_output=torch.nn.Linear(configs.LINEAR_HIDDEN_DIM,3)

        self.dropout_output=torch.nn.Dropout(0.1)

    def forward_ATSA(self,batch):
        aspect_features = []
        for i,x in enumerate(batch):
            sentence=x.sentence
            for aspect_term,aspect_from_pos in zip(x.aspect_term,x.aspect_from_pos):
                sentence=sentence[:aspect_from_pos+len(sentence)-len(x.sentence)]+' <AS> {} <AE> '.format(aspect_term)+x.sentence[aspect_from_pos+len(aspect_term):]
            sentence=re.sub('\s+',' ',sentence)
            features = self.roberta.extract_features_aligned_to_words(sentence.strip())

            for t,x in enumerate(features[1:]):
                if str(x)=='AS' and str(features[t])=='<':
                    aspect_features.append(x.vector)

        aspect_features=torch.stack(aspect_features,dim=0).to(self.configs.DEVICE)

        atsa_features=self.dropout_output(aspect_features)
        atsa_features=self.linear_hidden(atsa_features)
        atsa_features=F.gelu(atsa_features)
        predictions=self.linear_output(atsa_features)

        return predictions