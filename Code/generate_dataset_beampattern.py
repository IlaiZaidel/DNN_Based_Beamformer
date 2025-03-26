from torch.utils.data import Dataset
import scipy.io as sio
import os

class GeneretedInputOutputBeamPattern(Dataset):
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
        # Get audio file name DNN_Based_Beamformer/Code/beampattern_gen/my_reference_feature_vector_0.mat
        new_path ='/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/white_reference_feature_vector_0.mat' #self.path + 'my_second_reference_feature_vector.mat' # 'my_second_reference_feature_vector.mat' # 'reference_feature_vector_0.mat' # The name of the file
        '/dsi/gannot-lab1/datasets/Ilai_data/Two_Directional_Noises_Test/my_feature_vector_'
        
        train = sio.loadmat(new_path) 

        # Loads data
        self.y = train['feature']
        try:
            self.x = train['original']
        except:
            self.x = train['fulloriginal']     
         # Ilai Z: Added fullnoise output so i could use it as a regularization term
        self.fullnoise = train['fullnoise']

        # Normalizing the input and output signals
        max_y = abs(self.y[:,self.mic_ref]).max(0)
        self.y = self.y/max_y
        max_x = abs(self.x[:,self.mic_ref]).max(0)
        self.x = self.x/max_y
        max_noise = abs(self.fullnoise[:, self.mic_ref]).max(0)
        self.fullnoise = self.fullnoise/max_y
        return self.y,self.x, self.fullnoise

