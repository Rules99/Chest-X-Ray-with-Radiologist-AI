
tags = ['normal', 'right', 'lung', 'calcified granuloma', 'upper lobe', 'lingula', 'opacity', 'pulmonary atelectasis', 'interstitial', 'bilateral', 'diffuse', 'markings', 'prominent', 'left', 'density', 'retrocardiac', 'metabolic', 'spine', 'calcinosis', 'base', 'bone diseases', 'tortuous', 'indwelling', 'degenerative', 'aorta', 'catheters', 'thoracic vertebrae', 'mild', 'cardiomegaly', 'severe', 'diaphragm', 'elevated', 'hypoinflation', 'pulmonary congestion', 'technical quality of image unsatisfactory', 'chronic', 'pleural effusion', 'consolidation', 'costophrenic angle', 'airspace disease', 'blunted', 'surgical instruments', 'implanted medical device', 'patchy', 'streaky', 'pleura', 'thickening', 'focal', 'cicatrix', 'hilum', 'lower lobe', 'round', 'small', 'hyperdistention', 'mediastinum', 'nodule', 'no indexing', 'posterior', 'obscured', 'scoliosis', 'bronchovascular', 'granulomatous disease', 'multiple', 'osteophyte', 'middle lobe', 'hernia', 'hiatal', 'thoracic', 'pulmonary emphysema', 'lymph nodes', 'atherosclerosis', 'deformity', 'anterior', 'ribs', 'lucency', 'scattered', 'lumbar vertebrae', 'flattened', 'spondylosis', 'bone', 'borderline', 'fractures', 'thorax', 'healed', 'kyphosis', 'chronic obstructive', 'emphysema', 'pulmonary disease', 'infiltrate', 'pulmonary edema', 'moderate', 'enlarged', 'cardiac shadow', 'foreign bodies', 'spinal fusion', 'apex', 'diaphragmatic eventration', 'arthritis', 'pneumonia', 'abdomen', 'large', 'tube', 'inserted', 'paratracheal', 'granuloma']


class argHandler(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    _descriptions = {'help, --h, -h': 'show this super helpful message and exit'}
    def setDefaults(self):
    
  
        self.define('all_data_csv', 'data/interim/all_data_sample.csv',
                    'path to all data csv containing the images names and the labels')
     
        self.define('data_dir', './interim',
                    'path to folder containing the patient folders which containing the images')
        # 
        self.define('visual_model_name', 'fine_tuned_chexnet',
                    'path to folder containing the patient folders which containing the images')
      
        self.define('visual_model_pop_layers', 2,
                    'number of conv layers to pop to get visual features')
        # 
        self.define('csv_label_columns', ['Caption'], 'the name of the label columns in the csv')
  
        self.define('max_sequence_length', 200,
                    'Maximum number of words in a sentence')
        self.define('num_epochs', 100, 'maximum number of epochs')
        # 
        self.define('encoder_layers', [0.4], 'a list describing the hidden layers of the encoder. Example [10,0.4,5] will create a hidden layer with size 10 then dropout wth drop prob 0.4, then hidden layer with size 5. If empty it will connect to output nodes directly.')
        
        self.define('tags_threshold', -1,
                    'The threshold from which to detect a tag. -1 will multiply the tags embeddings according to prediction')
        # 
        self.define('tokenizer_vocab_size', 1001,
                    'The number of words to tokinze, the rest will be set as <unk>')
        
        self.define('beam_width', 7, 'The beam search width during evaluation')
        
        self.define('ckpt_path', './models/checkpoints/nlp/CDGPT2/',
                    'where to save the checkpoints. The path will be created if it does not exist. The system saves every epoch by default')
        self.define('tags', tags,
                    'the names of the tags')

    def define(self, argName, default, description):
        self[argName] = default
        self._descriptions[argName] = description

    def help(self):
        print('Arguments:')
        spacing = max([len(i) for i in self._descriptions.keys()]) + 2
        for item in self._descriptions:
            currentSpacing = spacing - len(item)
            print('  --' + item + (' ' * currentSpacing) + self._descriptions[item])
        exit()