import numpy as np  # For createImbIdxs() & make_imb_data()

def createImbIdxs(labels, n_data_per_class) :
    '''
    Creates a List containing Indexes of the Imbalanced Classification
    Input: 
        labels: Ground Truth of Dataset
        n_data_per_class: Class Distribution of Dataset desired
    Output:
        data_idxs: List containing indexes for Dataset 
    '''
    labels = np.array(labels) # Classification Ground Truth 
    data_idxs = []  # Collect Ground Truth Indexes

    for i in range( len(n_data_per_class) ) :
        idxs = np.where(labels == i)[0]
        data_idxs.extend(idxs[ :n_data_per_class[i] ])

    return data_idxs

def checkReverseDistb(imb_ratio) :
    reverse = False
    if imb_ratio / abs(imb_ratio) == -1 :
        reverse = True
        imb_ratio = imb_ratio * -1

    return reverse, imb_ratio
  
  def make_imb_data(max_num, class_num, gamma):
    reverse, gamma = checkReverseDistb(gamma)

    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if reverse :
        class_num_list.reverse()
    print(class_num_list)
    return list(class_num_list)
