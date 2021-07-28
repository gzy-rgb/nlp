from models import model
from dataLoader import loadData

def score():
	time = 5
	meanAcc = 0
	names = ["CORA", "CITESEER", "PUBMED"]
	for name in names:
		for i in range(time):
			#nodes are labeled from 0 to N - 1
			#trainData{node:numpy array(N, 1),
			#          edge:numpy array(M, 2),
			#          node_attr:numpy array(N, D),
			#		   ID: (N1, 1) numbers in range 0 to N - 1
			#          label:numpy array(N1,1)}
			
			#testData{node:numpy array(N, 1),
			#         edge:numpy array(M, 2),
			#         node_attr:numpy array(N, D),
			#         ID: (N2, 1)} numbers in range 0 to N - 1}
			
			#testLabel:numpy array(N2,1)
			
			#N1 + N2 = N

			#loadData will random split nodeID in train and test, the split rate is 2:8
	    	trainData, testData, testLabel = loadData(name)
	    	trainedModel = model.train(trainData)

	    	#return a numpy array of (N2, 1) contains the predicted label of test nodes 
	    	predictedLabel = model.test(testData)
	    	meanACC += accuracy(testLabel, predictedLabel)

    meanACC = meanAcc * 1.0 / time / len(names)

if __name__ == '__main__':
	score()
