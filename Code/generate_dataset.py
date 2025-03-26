from torch.utils.data import Dataset
import scipy.io as sio
import os

class GeneretedInputOutput(Dataset):
    '''Generated Input and Output to the model'''

    def __init__(self,path,mic_ref):
        '''
        Args:
            path (string): Directory with all the features.
            mic_ref (int): Reference microphone
        '''
        self.path = path
        self.mic_ref = mic_ref - 1 

    def __len__(self): 
        # Return the number of examples we have in the folder in the path
        file_list = os.listdir(self.path)  
        return len(file_list)

    def __getitem__(self,i):
        # Get audio file name 
        new_path = self.path + 'my_feature_vector_' + str(i) + '.mat'   # The name of the file ## Change it to feature_vector!
        train = sio.loadmat(new_path) 

        # Loads data
        self.y = train['feature']
        try:
            self.x = train['original']
        except:
            self.x = train['fulloriginal']     
        
        # 1 Noise
        #self.fullnoise = train['fullnoise']


        # 20.03 - Two noise terms:
        self.fullnoise_first = train['fullnoise_first']
        self.fullnoise_second = train['fullnoise_second']


        
        # Normalizing the input and output signals
        max_y = abs(self.y[:,self.mic_ref]).max(0)
        self.y = self.y/max_y
        self.x = self.x//max_y
        self.fullnoise_first = self.fullnoise_first/max_y #/max_noise
        self.fullnoise_second = self.fullnoise_second/max_y #/max_noise
        return self.y,self.x, self.fullnoise_first, self.fullnoise_second


