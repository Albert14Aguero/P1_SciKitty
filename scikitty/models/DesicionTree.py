import numpy as np

class Node:
    def __init__(self, impurity = 1, dataset = np.array([]), classPredicted = "", height = 0, amountEachClass = []):
        self.impurity = impurity
        self.dataset = dataset
        self.classPredicted = classPredicted
        self.samples = dataset.shape[0]
        self.childs = []
        self.height = height
        self.amountEachClass = amountEachClass

############################################################################################################################################################################

class DecisionTree:
    def __init__(self, dataset, target, height = 5, method = 'entropy'):
        self.target = target
        self.feature_indexes = {n:i for i, n in enumerate(dataset[0, :])}
        self.height = height
        self.dataset = cleanDataset(dataset)
        self.method = method
        self.root = Node(dataset = self.dataset, classPredicted= next(iter(self.feature_indexes)), height = 0)
      

    def fit(self):
        actualHeight = 0
        actualDataset = self.dataset
        self.root.amountEachClass = np.unique(self.target, return_counts = True)[1]
        actualNode = self.root
        bestFeature = self.getBestFeature(actualDataset)

        print("Actual dataset", actualDataset)
        
        print(self.root.amountEachClass)
        print(self.root.samples)
        print(self.root.classPredicted)
        print(self.feature_indexes)
        print(bestFeature)
        print(self.columnName(next(iter(bestFeature))))
        # while actualDataset.shape[1] > 0:
        #     actualHeight = actualHeight+1
        #     if actualHeight >= self.height: 
        #         break
        #     bestFeature = self.getBestFeature(actualDataset)

    def columnName(self, column):
        return self.feature_indexes[column]  
            
    def getBestFeature(self, dataset):
        minorImpurity = {next(iter(self.feature_indexes)): Impurity(dataset[:, 0], self.target, self.method)}
        print("start", minorImpurity)
        for columnName, column in self.feature_indexes.items():
            if np.unique(dataset[:, column]).size == 2:
                actualImpurity = Impurity(dataset[:, column], self.target, self.method)
                print("Impurity ", actualImpurity)
                if actualImpurity != 0 : 
                    if actualImpurity < next(iter(minorImpurity.values())):
                        minorImpurity = {columnName : actualImpurity}
                    if next(iter(minorImpurity.values())) == 0:
                        minorImpurity = {columnName : actualImpurity}
        print("Minor ", minorImpurity)
        return minorImpurity
    
    
    
############################################################################################################################################################################



def cleanDataset(dataset):
    newDataset = dataset #Keep de binary columns
    if np.ndim(newDataset) == 2: #if the first column is ID
        for column in range(newDataset.shape[1]): #iterate until de number of columns
            values = np.unique(newDataset[1:, column])
            try:
                 #get the unique values of the column
                if values.size == 2: #if the column has only 2 unique values
                    newDataset[1:, column] = np.where(newDataset[1:, column] == values[0], 0, 1) #replace the values for 0 and 1
                else:
                    values.astype(int)
            except ValueError:
                newDataset[1:, column] = -1
    else:
        values = np.unique(newDataset[1:])
        newDataset = np.where(newDataset[1:] == values[0], 0, 1)
        return newDataset.astype(int)
    print("new Dataset", newDataset)
    return newDataset[1:, :].astype(int) #return the dataset as integer

def Impurity(feature, target, method = 'entropy'): #receives column of the feature that wants to get de entropy, the column of target, 
    # and the method that wants to use, could be entropy(default) or gini
    print("Feature", feature)
    print("Target", target)
    if np.unique(feature).size == 2: #if the column has only 2 unique values
        feature_target = np.column_stack((feature, target)) #combine the feature and target column in one matrix 
        p_feature_true = np.count_nonzero(feature == 1)/feature.size #proportion of elements that are 1 at the feature
        p_feature_false = 1-p_feature_true #proportion of elements that are 0 at the feature
        if method == 'entropy': #if the choosen method is entropy
            bits_feature_true = 0 #needed bits to codificate feature value 1 
            bits_feature_false = 0 #needed bits to codificate feature value 0

            if p_feature_true != 0:
                p_feature_t_target_t = np.sum(np.all(feature_target == [1, 1], axis=1))/np.count_nonzero(feature == 1) #proportion of elements that are 1 at the feature and 1 at target
                p_feature_t_target_f = 1-p_feature_t_target_t #proportion of elements that are 1 at the feature and 0 at target
                if p_feature_t_target_f != 0 and p_feature_t_target_t != 0:
                    bits_feature_true = p_feature_true*((p_feature_t_target_t*np.log2(1/p_feature_t_target_t)) + 
                                                        (p_feature_t_target_f*np.log2(1/p_feature_t_target_f))) 
                
            if p_feature_false != 0:
                p_feature_f_target_t = np.sum(np.all(feature_target == [0, 1], axis=1))/np.count_nonzero(feature == 0)#proportion of elements that are 0 at the feature and 1 at target
                p_feature_f_target_f = 1-p_feature_f_target_t #proportion of elements that are 0 at the feature and 0 at target
                if p_feature_f_target_f != 0 and p_feature_f_target_t != 0:
                    bits_feature_false = p_feature_false*((p_feature_f_target_t*np.log2(1/p_feature_f_target_t)) + 
                                                            (p_feature_f_target_f*np.log2(1/p_feature_f_target_f)))
                
            return bits_feature_false + bits_feature_true #Total amount of bits to codificate the feature

        if method == 'gini':
            p_feature_t_target_t = np.sum(np.all(feature_target == [1, 1], axis=1))/np.count_nonzero(feature == 1) #proportion of elements that are 1 at the feature and 1 at target
            p_feature_t_target_f = 1-p_feature_t_target_t
            gini_feature_true = p_feature_true * (1 - (p_feature_t_target_t**2 + p_feature_t_target_f**2))

            p_feature_f_target_t = np.sum(np.all(feature_target == [0, 1], axis=1))/np.count_nonzero(feature == 0)#proportion of elements that are 0 at the feature and 1 at target
            p_feature_f_target_f = 1-p_feature_f_target_t
            gini_feature_false = p_feature_false * (1 - (p_feature_f_target_t**2 + p_feature_f_target_f**2))
            return gini_feature_true + gini_feature_false
    else:
        if method == 'entropy':
            # Entropy for continuous features using discretization (e.g., binning)
            # 1. Discretize the continuous feature using techniques like equal-width binning
            bins = np.unique(np.linspace(feature.min(), feature.max(), num=2))  # Example with 10 bins
            print("Bins: ", bins)
            discretized_feature = np.digitize(feature, bins) - 1  # Discretized feature (0-based indexing)
            print("Discretizacio: ", discretized_feature)
            # 2. Calculate entropy based on the discretized feature and target
            entropy = 0
            for bin in np.unique(discretized_feature):
                p_bin = np.sum(discretized_feature == bin) / feature.size  # Proportion of elements in the bin
                if p_bin != 0:
                    p_bin_target_true = np.sum(np.all([discretized_feature == bin, target == 1], axis=0)) / p_bin
                    entropy -= p_bin * np.log2(p_bin_target_true)
            return entropy

        elif method == 'gini':
            # Gini impurity for continuous features using discretization
            # (similar approach as for entropy)
            bins = np.unique(np.linspace(feature.min(), feature.max()))
            discretized_feature = np.digitize(feature, bins) - 1

            gini = 1
            for bin in np.unique(discretized_feature):
                p_bin = np.sum(discretized_feature == bin) / feature.size
                p_bin_target = np.array([np.sum(target[discretized_feature == bin] == val) / p_bin for val in np.unique(target)])
                gini -= np.sum(p_bin_target**2)
            return gini

############################################################################################################################################################################

#Pruebas

data = np.loadtxt('../datasets/fictional_reading_place.csv', dtype = str, delimiter = ',')

model = DecisionTree(data, np.array([0,1,0,0,1,0,0,1,0,0,0,0,1,1,1,1,1,1]), height = 5, method = 'entropy')

# target fictional_reading_place np.array([0,1,0,0,1,0,0,1,0,0,0,0,1,1,1,1,1,1])
# target fictional_disease np.array([1,0,1,1,0,0,1,1,0,0,1,1,1,1,0,1,0,0,1,1,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0])
# target playTennis np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,1,0])

model.fit()
