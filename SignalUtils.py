import scipy.io as spio

def saveSubmission(saveFile, d, Index, Class, insertData=True):
    """
    Saves the results to a .mat file formatted for submission.

    Args:
        saveFile (str): The destination file path.
        d (np.ndarray): Raw signal data.
        Index (list | np.ndarray): List of detected spike indices.
        Class (list | np.ndarray): List of corresponding class labels.
        insertData (Optional[bool]): If True, includes the raw signal 'd' in the .mat file. Defaults to True.

    Returns:
        None
    """
    submissionData = {}
    if insertData: 
        submissionData['d'] = d
    submissionData['Index'] = Index
    submissionData['Class'] = Class
    spio.savemat(saveFile, submissionData)

def loadData(filePath, isTraining=False):
    """
    Loads the data from a specified .mat file path.

    Args:
        filePath (str): File path to .mat file
        isTraining (Optional[bool]): If True, attempts to load 'Index' and 'Class' labels alongside the signal. Defaults to False.

    Returns:
        np.ndarray | tuple: 
            - If isTraining=False: Returns the raw signal array `rawData`.
            - If isTraining=True: Returns `(rawData, signalLocationIndex, sampleClass)`.
            - Returns (None, None, None) if the file is not found.
    """
    try:
        mat = spio.loadmat(filePath, squeeze_me=True)
    except FileNotFoundError:
        print(f"Error: File not found at {filePath}")
        return None, None, None

    rawData = mat['d']      
    if not isTraining:      
        return rawData
    
    # Handle cases where Index or Class might be missing in some raw files
    try:
        signalLocationIndex = mat['Index']      
        sampleClass = mat['Class']
    except KeyError:
        signalLocationIndex = []
        sampleClass = []

    return rawData, signalLocationIndex, sampleClass

def sortTupleLists(list1, list2, index=0):
    """
    Zips two lists together and sorts them based on the value at the specified tuple index.

    Args:
        list1 (list): First list (e.g., Indices).
        list2 (list): Second list (e.g., Classes).
        index (Optional[int]): The tuple index to sort by (0 for list1, 1 for list2). Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - sortedTupleList (list[tuple]): The combined list of sorted tuples.
            - sortedList1 (tuple): The sorted elements of the first list.
            - sortedList2 (tuple): The sorted elements of the second list.
    """
    combinedList = list(zip(list1, list2))
    sortedTupleList = sorted(combinedList, key=lambda x: x[index])
    sortedList1, sortedList2 = zip(*sortedTupleList)
    
    return sortedTupleList, sortedList1, sortedList2