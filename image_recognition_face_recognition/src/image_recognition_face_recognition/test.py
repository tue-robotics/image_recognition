# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
print("Hello world")
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

_trained_faces = []
class TrainedFace:
    def __init__(self, label):
        self.label = label
        self.representations = []
        
    def get_label(self) -> str : return self.label 
    
embeddings = [40,400]

_trained_faces.append(TrainedFace('jason'))
_trained_faces.append(TrainedFace('kona'))
_trained_faces[0].representations.append(50)
_trained_faces[0].representations.append(40)

_trained_faces[1].representations.append(300)
_trained_faces[1].representations.append(407)
_trained_faces[1].representations.append(700)
for trained_face in _trained_faces:
            print(f"Label: {trained_face.label}, Representations: {trained_face.representations}")
            
#dist = [(e1.representations[e3] - e2) for e1 in _trained_faces]for e3 in range(len(e1.representations)) for e2 in embeddings]
dist_per_emb_final = []
dist = []
dist_per_emb = []

min_of_emb_final = [] 

min_index_list_per_emb = []
min_index_list= []

for e2 in embeddings:
    for e1 in _trained_faces:
        for e3 in (e1.representations):
            dist_per_emb.append(abs(e3 - e2)) 
        dist.append(dist_per_emb)
        logging.info(f"{dist_per_emb} dist_per_emb")
        print(dist_per_emb,'dist_per_emb')

        dist_per_emb = []
    dist_per_emb_final.append(dist)
    dist = []
   
print(dist_per_emb_final,'dist_per_emb_final')
for i in dist_per_emb_final:
    min_of_emb = [min(j) for j in i]
    print(min_of_emb,'min_of_emb')
    min_of_emb_final.append(min_of_emb)
print(min_of_emb_final,'min_of_emb_final')


for idx in min_of_emb_final:
    print(idx, 'min_index_list_per_emb')
    min_index_list_per_emb.append(idx.index(min(idx)))
    min_index_list.append(min(idx))
    print(idx.index(min(idx)),'minimum index')
print(min_index_list,'min_index_list')

labelling = [_trained_faces[i].get_label() for i in min_index_list_per_emb]
print(labelling, min_index_list)


def _get_dists(self, embeddings):
        '''
        HERE WE NEED TO FIND THE SHOURTEST EUKLIDEAN DISTANCE BETWEEN ALL OUR VECTORS
        SPLIT THE FACES WE HAVE DETECT AND APPEND IT IN THE LIST 
        THEN FOR EVERY FACE WE DETECT IN embeddedings[i] WE NEED TO CHECK THE CLOSEST DETECTION WITH THE LIST 
        '''
        dist_per_emb_final = []
        dist = []
        dist_per_emb = []

        min_of_emb_final = [] 

        min_index_list_per_emb = []
        min_index_list= []

        for e2 in embeddings:
            for e1 in _trained_faces:
                for e3 in (e1.representations):
                    dist_per_emb.append(abs(e3 - e2)) 
                dist.append(dist_per_emb)
                print(dist_per_emb,'dist_per_emb')
                dist_per_emb = []
            dist_per_emb_final.append(dist)
            dist = []
        
        print(dist_per_emb_final,'dist_per_emb_final')
        for i in dist_per_emb_final:
            min_of_emb = [min(j) for j in i]
            print(min_of_emb,'min_of_emb')
            min_of_emb_final.append(min_of_emb)
        print(min_of_emb_final,'min_of_emb_final')


        for idx in min_of_emb_final:
            print(idx, 'min_index_list_per_emb')
            min_index_list_per_emb.append(idx.index(min(idx)))
            min_index_list.append(min(idx))

        labelling = [_trained_faces[i].get_label() for i in min_index_list_per_emb]
        print(labelling, min_index_list)
                    
        return dist, labelling
    
    

def _get_dists(self, embeddings):
    '''
    HERE WE NEED TO FIND THE SHOURTEST EUKLIDEAN DISTANCE BETWEEN ALL OUR VECTORS
    SPLIT THE FACES WE HAVE DETECT AND APPEND IT IN THE LIST 
    THEN FOR EVERY FACE WE DETECT IN embeddedings[i] WE NEED TO CHECK THE CLOSEST DETECTION WITH THE LIST 
    '''
    dist = []
    min_index=[]
    labelling = []
    #this is tries for l2 norm but non needed since this technique uses euclidean distanses.
    #dist.append(min([[np.subtract(vector_a, embeddings[i,:]) for vector_a in self._trained_faces] for i in range(embeddings.size(0))]))
    #dist.append(min([[np.dot(vector_a - embeddings[i,:], vector_a - embeddings[i,:]) for vector_a in self._trained_faces] for i in range(embeddings.size(0))]))

    #for face_recognition in embeddings:
        #if save_images:
    #this is only for one face stored in every label in self._trained_faces THIS THIS THIS
    #dist = [[(e1.representations[0] - e2).norm().item() for e2 in embeddings] for e1 in self._trained_faces]
    #dist = [(e1.representations[e3] - e2).norm().item() for e1 in self._trained_faces for e3 in range(len(e1.representations)) for e2 in embeddings]
    #   dist = [[(e1.representations[e3] - e2).norm().item() for e2 in embeddings] for e1 in self._trained_faces for e3 in range(len(e1.representations))]
    #dist = [[[(e1.representations[e3] - e2).norm().item() for e1 in self._trained_faces] for e3 in range(len(self._trained_faces[0].representations))] for e2 in embeddings]
    ll = []
    for e2 in embeddings:
        for e1 in self._trained_faces:
            for e3 in (e1.representations):
                ll.append((e3 - e2).norm().item()) 
            dist.append(ll)
            ll = []
    
    print(dist, '<-- Dist')
    distance = [min(dist[index]) for index in range(len(dist))]
    print(min(dist),'minsistmist')
    print(distance, '<-- distance')
    # get the index of the minimum distance (this would correspond to the appropriate label)
    min_index = [idx.index(min(idx)) for idx in dist]
    print(min_index,'min index')

    labelling = [self._trained_faces[i].get_label() for i in min_index]
    #labelling = [np.argmin(dist[index]) for index in range(len(dist))]
    print(distance,labelling,'<--Distance with the close label is')
            
    return dist, labelling