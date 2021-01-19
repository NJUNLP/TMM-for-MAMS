import torch.nn
import torch.hub
import torch.nn.functional as F

from fairseq.models.roberta import RobertaModel
import re

class RobertaACSA(torch.nn.Module):
    def __init__(self,configs):
        super(RobertaACSA, self).__init__()

        self.configs=configs
        self.roberta=RobertaModel.from_pretrained('pretrained/roberta.large', checkpoint_file='model.pt')

        self.linear_hidden=torch.nn.Linear(configs.ROBERTA_DIM,configs.LINEAR_HIDDEN_DIM)
        self.linear_output=torch.nn.Linear(configs.LINEAR_HIDDEN_DIM,3)

        self.dropout_output=torch.nn.Dropout(0.1)

    def forward_ACSA(self,batch):
        aspect_features = []
        for i,x in enumerate(batch):
            sentence=x.sentence+' ACSA'
            for category in x.category:
                sentence=sentence+' {}'.format(x.index2category[category])

            sentence=re.sub('\s+',' ',sentence)
            features = self.roberta.extract_features_aligned_to_words(sentence)

            is_category=False
            for f in features[1:-1]:
                if is_category:
                    aspect_features.append(f.vector)
                if str(f)=='ACSA':
                    is_category=True

        aspect_features=torch.stack(aspect_features,dim=0).to(self.configs.DEVICE)

        atsa_features=self.dropout_output(aspect_features)
        atsa_features=self.linear_hidden(atsa_features)
        atsa_features=F.gelu(atsa_features)
        predictions=self.linear_output(atsa_features)

        return predictions