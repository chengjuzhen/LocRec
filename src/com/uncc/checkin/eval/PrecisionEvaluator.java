package com.uncc.checkin.eval;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;

import com.uncc.checkin.CheckinException;
import com.uncc.checkin.util.Utils;

public abstract class PrecisionEvaluator {
	public static final double DEFAULT_MIN_RATING = -1.0 * (Double.MAX_VALUE - 1);

	public static enum MODE {
		ALL,
		CANDIDATES,
		DYNAMIC_CANDIDATES
	}

	protected int userNum;
	protected int itemNum;
	protected int[] testUserIDs              = null;
	protected int[][] testUserPlaceIDs       = null;
	protected float[][] testUserPlaceRatings = null;
	protected int[][] userTrainPlaceIDs      = null;

	// Mode All
	public abstract double[] getPlaceProb(int testUserID);

	// Mode CANDIDATES, note the IDs and probabilities must be mapped.
	// and not include train placeID
	public abstract int[] getCandidateIDs(int testUserID);
	public abstract double[] getCandidateProbs(int testUserID, int[] candidateIDs);
	
	// TO be override
	// Mode DYNAMIC_CANDIDATES
	public int[] getCandidateIDs(int testUserID, int testPlaceID)  {
		throw new RuntimeException("Need to override under mode DYNAMIC_CANDIDATES.");
	}
	public double[] getCandidateProbs(int testUserID, int testPlaceID,
							int[] candidateIDs) {
		throw new RuntimeException("Need to override under mode DYNAMIC_CANDIDATES.");
	}
	
	// TO be override
	public void init() throws CheckinException {};
	
	public PrecisionEvaluator(int userNum, int itemNum) {
		this.userNum = userNum;
		this.itemNum = itemNum;
	}

	private void init(File trainRatingFile, File testRatingFile,
			BufferedWriter infoWriter, MODE mode)
					throws IOException, CheckinException{
		Utils.writeAndPrint(infoWriter, new Date().toString(), true);
		Utils.writeAndPrint(infoWriter, "TrainRatingFile: " +
				trainRatingFile.getAbsolutePath(), true);
		Utils.writeAndPrint(infoWriter, "TestRatingFile : " +
				testRatingFile.getAbsolutePath(), true);
		Utils.writeAndPrint(infoWriter, "", true);
		
		testUserIDs          = null;
		testUserPlaceIDs     = null;
		testUserPlaceRatings = null;
		{
			Object[] userTestData = Utils.loadUserRatingData(
					testRatingFile, "Test", infoWriter);
			if (userTestData == null || userTestData.length == 0)
				throw new CheckinException("No user test data found.");
			testUserIDs          = (int[]) userTestData[0];
			testUserPlaceIDs     = (int[][]) userTestData[1];
			testUserPlaceRatings = (float[][]) userTestData[2];
			if (testUserIDs.length != testUserPlaceIDs.length ||
					testUserIDs.length != testUserPlaceIDs.length ||
					testUserIDs.length != testUserPlaceRatings.length)
				throw new CheckinException(
					"Inner Error: testUserIDSize=%s, testUserPlaceIDsSize=%s",
					testUserIDs.length, testUserPlaceIDs.length);
		}
		System.out.println("UserNum = " + userNum);
		System.out.println("ItemNum = " + itemNum);
		if (userTrainPlaceIDs == null) {
			System.out.println("Load userTrainPlaceIDs.");
			userTrainPlaceIDs = Utils.loadUserTrainPlaceIDs(
					trainRatingFile, userNum);
			if (userTrainPlaceIDs == null || userTrainPlaceIDs.length == 0) {
				throw new CheckinException("No train data loaded.");
			}
		} else {
			System.out.println("Have loaded userTrainPlaceIDs.");
		}
	}

	public void evaluatePrecisionRecallMapInMultiThread(File trainRatingFile,
			File testRatingFile, final int[] Ks, File infoFile,
			int threadNum, MODE mode) throws CheckinException,
						com.lhy.tool.ToolException {
		BufferedWriter infoWriter = null;
		try {
			if (! infoFile.getParentFile().exists())
				infoFile.getParentFile().mkdirs();
			infoWriter = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(infoFile, true)));

			init(trainRatingFile, testRatingFile, infoWriter, mode);
			init(); // called by additional code

			double[][] avgPrecision = new double[threadNum][Ks.length];
			double[][] avgRecall    = new double[threadNum][Ks.length];
			double[][] map          = new double[threadNum][1];
			double[][] mauc         = new double[threadNum][1];
			int[][] nUser4Precision = new int[threadNum][Ks.length];
			int[][] nUser4Recall    = new int[threadNum][Ks.length];
			int[][] nUser4MAP       = new int[threadNum][1];
			int[][] nUser4AUC       = new int[threadNum][1];
			int[][] testUserIndices = new int[threadNum][2];
			{
				int testUserNum = testUserIDs.length / threadNum;
				int startIndex  = 0;
				for (int i = 0; i < threadNum; i ++) {
					int endIndex = startIndex + testUserNum;
					if (i == threadNum - 1) {
						endIndex = testUserIDs.length - 1;
					}
					testUserIndices[i][0] = startIndex;
					testUserIndices[i][1] = endIndex;
					startIndex = endIndex + 1;
				}
				
			}
			System.out.println("EvaluateMode: " + mode.name());
			Thread[] threads  = new Thread[threadNum];
			for (int i = 0; i < threadNum; i ++) {
				if (mode == MODE.ALL) {
					threads[i] = new PrecisionThread(
						testUserIndices[i][0], testUserIndices[i][1],
						itemNum, Ks, avgPrecision[i], avgRecall[i],
						map[i], mauc[i], nUser4Precision[i], nUser4Recall[i],
						nUser4MAP[i], nUser4AUC[i]);
				} else
				if (mode == MODE.CANDIDATES){
					threads[i] = new PrecisionWithCandidatesThread(
						testUserIndices[i][0], testUserIndices[i][1],
						Ks, avgPrecision[i], avgRecall[i], map[i], mauc[i],
						nUser4Precision[i], nUser4Recall[i],
						nUser4MAP[i], nUser4AUC[i]);
				} else 
				if (mode == MODE.DYNAMIC_CANDIDATES) {
					threads[i] = new PrecisionWithDynamicCandidatesThread(
							testUserIndices[i][0], testUserIndices[i][1],
							Ks, avgPrecision[i], avgRecall[i], map[i], mauc[i],
							nUser4Precision[i], nUser4Recall[i],
							nUser4MAP[i], nUser4AUC[i]);
				} else {
					// never reaches
					throw new CheckinException(
						"No found mode: " + mode.name());
				}
				threads[i].start();
			}
			int count = 0;
			boolean status[] = new boolean[threadNum];
			while (true) {
				for (int i = 0; i < threads.length; i ++) {
					if (! status[i] && ! threads[i].isAlive()) {
						count ++;
						status[i] = true;
					}
				}
				if (count == threads.length) break;
			}
			double[] totalPrecision   = new double[Ks.length];
			double[] totalRecall      = new double[Ks.length];
			double totalMAP           = 0;
			double totalAUC           = 0;
			double[] totalnUser4Prec  = new double[Ks.length];
			double[] totalnUser4Recall= new double[Ks.length];
			double totalnUser4MAP     = 0;
			double totalnUser4AUC     = 0;
			boolean nUserChanged      = false;
			for (int i = 0; i < threadNum; i ++) {
				System.out.println("Thread-" + i);
				int currNUser = testUserIndices[i][1] - testUserIndices[i][0]+1; 
				for (int kIndex = 0; kIndex < Ks.length; kIndex ++) {
					totalPrecision[kIndex]    += avgPrecision[i][kIndex];
					totalRecall[kIndex]       += avgRecall[i][kIndex];
					totalnUser4Prec[kIndex]   += nUser4Precision[i][kIndex];
					totalnUser4Recall[kIndex] += nUser4Recall[i][kIndex];
					System.out.println(String.format("TotalPrecision : %s, nUser(%s, %s)",
							avgPrecision[i][kIndex], totalnUser4Prec[kIndex], currNUser));
					System.out.println(String.format("TotalRecall    : %s, nUser(%s, %s)",
							avgRecall[i][kIndex], nUser4Recall[i][kIndex], currNUser));
					// manual check
					if (totalnUser4Prec[kIndex] != totalnUser4Recall[kIndex])
						throw new CheckinException("nUser4Prec[%s] != nUser4Recall[%s]",
								nUser4Precision[i][kIndex], totalnUser4Recall[kIndex]);
				}
				totalMAP       += map[i][0];
				totalAUC       += mauc[i][0];
				totalnUser4MAP += nUser4MAP[i][0];
				totalnUser4AUC += nUser4AUC[i][0];
				System.out.println(String.format("TotalMAP       : %s, nUser(%s, %s)",
						map[i][0], nUser4MAP[i][0], currNUser));
				System.out.println(String.format("TotalAUC       : %s, nUser(%s, %s)", 
						mauc[i][0], nUser4AUC[i][0], currNUser));
				if (totalnUser4MAP != totalnUser4AUC)
					throw new CheckinException("totalnUser4MAP[%s] != totalnUser4AUC[%s]",
							totalnUser4MAP, totalnUser4AUC);
			}
			for (int kIndex = Ks.length - 1; kIndex >= 0; kIndex --) {
				totalPrecision[kIndex] /= totalnUser4Prec[kIndex];//testUserIDs.length + 0.0;
				totalRecall[kIndex]    /= totalnUser4Recall[kIndex];//testUserIDs.length + 0.0;
				if (!nUserChanged && totalnUser4Prec[kIndex] != testUserIDs.length) {
					nUserChanged = true;
				}
			}
			totalMAP /= totalnUser4MAP; //testUserIDs.length + 0.0
			totalAUC /= totalnUser4AUC; //testUserIDs.length + 0.0
			if (! nUserChanged && totalnUser4MAP != testUserIDs.length)
				nUserChanged = true;
	
			for (int kIndex = 0; kIndex < Ks.length; kIndex ++) {
				Utils.writeAndPrint(infoWriter, String.format(
				"K=%s\n===============\n" +
				"Precison=%s, Recall=%s\nMAP=%s, AUC=%s\n",
				Ks[kIndex], totalPrecision[kIndex],
				totalRecall[kIndex], totalMAP, totalAUC), true);
				infoWriter.flush();
			}
			String onelineInfo = "";
			String titleInfo   = "";
			String nUserCount  = nUserChanged ? "" : null;
			for (int kIndex = 0; kIndex < Ks.length; kIndex ++) {
				onelineInfo += String.format("%s\t", totalPrecision[kIndex]);
				titleInfo   += String.format("Precision@%s\t", Ks[kIndex]);
				if (nUserChanged) {
					nUserCount += String.format("(%s,%s)\t",
								totalnUser4Prec[kIndex], testUserIDs.length);
				}
			}
			for (int kIndex = 0; kIndex < Ks.length; kIndex ++) {
				onelineInfo += String.format("%s\t", totalRecall[kIndex]);
				titleInfo   += String.format("Recall@%s\t", Ks[kIndex]);
				if (nUserChanged) {
					nUserCount += String.format("(%s,%s)\t",
						totalnUser4Recall[kIndex], testUserIDs.length);
				}
			}
			onelineInfo += totalMAP + "\t" + totalAUC;
			titleInfo   += "MAP\tAUC";
			if (nUserChanged) {
				nUserCount += String.format("(%s,%s)\t",
							totalnUser4MAP, testUserIDs.length,
								totalnUser4AUC, testUserIDs.length);
			}
			
			Utils.writeAndPrint(infoWriter, titleInfo, true);
			Utils.writeAndPrint(infoWriter, onelineInfo, true);
			if (nUserChanged)
				Utils.writeAndPrint(infoWriter, nUserCount, true);
			
			infoWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(infoWriter);
		}
	}

	private class PrecisionThread extends Thread {
		private final int testUserStartIndex;
		private final int testUserEndIndex;
		private final int itemNum;
		private final int[] Ks;
		private final double[] avgPrecision;
		private final double[] avgRecall;
		private final double[] map;
		private final double[] mauc;
		private final int[] nUser4Precision;
		private final int[] nUser4Recall;
		private final int[] nUser4MAP;
		private final int[] nUser4AUC;

		public PrecisionThread(int testStartIndex, int testEndIndex,
				int itemNum, int[] Ks, double avgPrecision[], double[]avgRecall,
				double[] map, double[] mauc, int[] nUser4Precision,
				int[]nUser4Recall, int[] nUser4MAP, int[] nUser4AUC) {
			this.testUserStartIndex = testStartIndex;
			this.testUserEndIndex   = testEndIndex;
			this.itemNum           = itemNum;
			this.Ks                  = Ks;
			this.avgPrecision       = avgPrecision;
			this.avgRecall          = avgRecall;
			this.map                = map;
			this.mauc               = mauc;
			this.nUser4Precision    = nUser4Precision;
			this.nUser4Recall       = nUser4Recall;
			this.nUser4MAP          = nUser4MAP;
			this.nUser4AUC          = nUser4AUC;
			// check
			{
				int maxK = Ks[0];
				for (int i = 1; i < Ks.length; i ++)
					if (Ks[i] > maxK) maxK = Ks[i];
				for (int testIndex = testUserStartIndex; testIndex <=
						testUserEndIndex; testIndex ++) {
					int testUserID      = testUserIDs[testIndex];
					int[] testPlaceIDs  = testUserPlaceIDs[testIndex];
					int trainNum        = testUserID >= userTrainPlaceIDs.length ?
							0 : (userTrainPlaceIDs[testUserID] == null ?
							0:userTrainPlaceIDs[testUserID].length);
					if (maxK > itemNum - trainNum -
							testPlaceIDs.length)
						throw new RuntimeException(String.format(
							"TestUserID=%s, MaxK=%s, itemNum=%s, trainNum=%s, testNum=%s",
							testUserID, maxK, itemNum, trainNum, testPlaceIDs.length));
				}
			}
		}

		public void run() {
			System.out.println(String.format("TestStartIndex=%s, TestEndIndex=%s",
					testUserStartIndex, testUserEndIndex));
			for (int i = Ks.length - 1; i >= 0; i --) {
				avgPrecision[i] = 0;
				avgRecall[i]    = 0;
			}
			map[0] = 0;
			mauc[0]= 0;
			for (int testIndex = testUserStartIndex; testIndex <=testUserEndIndex; testIndex ++) {
				/*{
					int runNum = testIndex - testUserStartIndex;
					if (runNum % 1000 == 0) {
						if (runNum > 0) {
							System.out.println(String.format(
							"TestIndex=%s, precision_0=%s, recall_0=%s, map=%s, auc=%s",
							testIndex, (avgPrecision[0]/runNum),
							(avgRecall[0]/runNum), (map[0]/runNum),
							(mauc[0]/runNum)));
						} else {
							System.out.println("TestIndex=" + testIndex);
						}
					}
				}*/
				int testUserID    = testUserIDs[testIndex];
				int[] testPlaceIDs= testUserPlaceIDs[testIndex];
				int trainNum      = 0;
				final double[] placeProbs = getPlaceProb(testUserID);
				if (testUserID < userTrainPlaceIDs.length) {
					/*{
						// check if prob < DEFAULT_MIN_RATING
						for (int itemID = 0; itemID < itemNum; itemID ++) {
							if (placeProbs[itemID] < DEFAULT_MIN_RATING) {
								throw new RuntimeException (String.format(
									"placeProbs[userID][%s] < DEFAULT_MIN_RATING[%s]",
									placeProbs[itemID], DEFAULT_MIN_RATING));
							}
						}
					} */
					int[] trainPlaceIDs = userTrainPlaceIDs[testUserID];
					if (trainPlaceIDs != null) {
						trainNum  = trainPlaceIDs.length;
						for (int trainPlaceID : trainPlaceIDs) {
							placeProbs[trainPlaceID] = DEFAULT_MIN_RATING;
						}
					}
				}

				Integer[] placeIndices = new Integer[itemNum];
				for (int placeID = 0; placeID < itemNum; placeID ++) {
					placeIndices[placeID] = placeID;
				}
				
				Arrays.sort(placeIndices, new Comparator<Integer>(){
					@Override
					public int compare(Integer index1, Integer index2) {
						double diff = placeProbs[index1] - placeProbs[index2];
						if (diff > 0 ) return -1;
						if (diff < 0) return 1;
						return 0;
					}
				});
				final int[] placeIDOrder = new int[itemNum];
				for (int order = 0; order < placeIndices.length; order ++) {
					placeIDOrder[placeIndices[order]] = order;
				}
				
				//double thresholdProb = placeProbs[placeIndices[K - 1]];
				placeIndices         = null; //clear
				final Integer[] orderedTestPlaceIDs = new Integer[testPlaceIDs.length];
				for (int index = 0; index < testPlaceIDs.length; index ++) {
					orderedTestPlaceIDs[index] = testPlaceIDs[index];
				}
				Arrays.sort(orderedTestPlaceIDs, new Comparator<Integer>(){
					@Override
					public int compare(Integer index1, Integer index2) {
						int diff = placeIDOrder[index1] - placeIDOrder[index2];
						if (diff > 0 ) return 1;
						if (diff < 0) return -1;
						return 0;
					}
				});
				double ap    = 0;
				double auc   = 0;
				int[] hitNum = new int[Ks.length];
				int unrelNum = itemNum - trainNum - testPlaceIDs.length;
				for (int index = 0; index < orderedTestPlaceIDs.length; index ++) {
					int testPlaceID = orderedTestPlaceIDs[index];
					int order = placeIDOrder[testPlaceID];
					for (int i = Ks.length - 1; i >= 0; i --) {
						if (order <= Ks[i] - 1) { // order from 0
						//if (thresholdProb <= placeProbs[testPlaceID]) {
							hitNum[i] ++;
						}
					}
					
					ap  += (index + 1.0) / (order + 1.0);
					auc += unrelNum - order + index;
				}
				for (int i = Ks.length - 1; i >= 0; i --) {
					if (Ks[i] < unrelNum + testPlaceIDs.length) {
						avgPrecision[i] += (hitNum[i] + 0.0) / (Ks[i] + 0.0);
						avgRecall[i]    += (hitNum[i] + 0.0) /
											(testPlaceIDs.length + 0.0);
						nUser4Precision[i] ++;
						nUser4Recall[i] ++;
					}
				}
				if (unrelNum > 0) {
					mauc[0] += auc / (testPlaceIDs.length * unrelNum);
					map[0]  += ap / (testPlaceIDs.length + 0.0);
					nUser4AUC[0] ++;
					nUser4MAP[0] ++;
				}
			} // end for
		} // end run
	}

	private class PrecisionWithCandidatesThread extends Thread {
		private final int testUserStartIndex;
		private final int testUserEndIndex;
		private final int[] Ks;
		private final double[] avgPrecision;
		private final double[] avgRecall;
		private final double[] map;
		private final double[] mauc;
		private final int[] nUser4Precision;
		private final int[] nUser4Recall;
		private final int[] nUser4MAP;
		private final int[] nUser4AUC;

		public PrecisionWithCandidatesThread(int testStartIndex,
			int testEndIndex, int[] Ks, double avgPrecision[],
			double[] avgRecall, double[]map, double[]mauc, int[]nUser4Precision,
			int[]nUser4Recall, int[] nUser4MAP, int[] nUser4AUC){
			this.testUserStartIndex = testStartIndex;
			this.testUserEndIndex   = testEndIndex;
			this.Ks                  = Ks;
			this.avgPrecision       = avgPrecision;
			this.avgRecall          = avgRecall;
			this.map                = map;
			this.mauc               = mauc;
			this.nUser4Precision    = nUser4Precision;
			this.nUser4Recall       = nUser4Recall;
			this.nUser4MAP          = nUser4MAP;
			this.nUser4AUC          = nUser4AUC;
		}

		public void run() {
			System.out.println(String.format("TestStartIndex=%s, TestEndIndex=%s",
					testUserStartIndex, testUserEndIndex));
			for (int i = Ks.length - 1; i >= 0; i --) {
				avgPrecision[i] = 0;
				avgRecall[i]    = 0;
			}
			map[0] = 0;
			for (int testIndex = testUserStartIndex; testIndex <=testUserEndIndex; testIndex ++) {
				{
					int runNum = testIndex - testUserStartIndex;
					if (runNum % 1000 == 0) {
						if (runNum > 0) {
							System.out.println(String.format(
							"TestIndex=%s, precision_0=%s, recall_0=%s, map=%s",
							testIndex, (avgPrecision[0]/runNum),
							(avgRecall[0]/runNum), (map[0]/runNum)));
						} else {
							System.out.println("TestIndex=" + testIndex);
						}
					}
				}
				int testUserID    = testUserIDs[testIndex];
				int[] testPlaceIDs= testUserPlaceIDs[testIndex];
				final int[] candPlaceIDs      = getCandidateIDs(testUserID);
				final double[] candPlaceProbs =
						getCandidateProbs(testUserID, candPlaceIDs);
				if (candPlaceIDs.length == 0) {
					System.err.println(String.format(
						"No candidates found for testUserID[%s].",
							testUserID));
				}
				Integer[] candPlaceIndices = new Integer[candPlaceIDs.length];
				for (int i = 0; i < candPlaceIndices.length; i ++) {
					candPlaceIndices[i] = i;
				}
			
				Arrays.sort(candPlaceIndices, new Comparator<Integer>(){
					@Override
					public int compare(Integer index1, Integer index2) {
						double diff = candPlaceProbs[index1] -
							candPlaceProbs[index2];
						if (diff > 0 ) return -1;
						if (diff < 0) return 1;
						return 0;
					}
				});
				Set<Integer> testPlaceIDSet = new HashSet<Integer>();
				for (int placeID : testPlaceIDs) 
					testPlaceIDSet.add(placeID);
				
				int[] testPlaceIDOrders = new int[testPlaceIDs.length];
				int index               = 0;
				for (int order = 0; order < candPlaceIndices.length; order ++) {
					int placeID = candPlaceIDs[candPlaceIndices[order]];
					if (testPlaceIDSet.contains(placeID)) {
						testPlaceIDOrders[index ++] = order; 
						if (index >= testPlaceIDOrders.length)
							break;
					}
				}
				if (index < testPlaceIDOrders.length) {
					System.err.println(String.format(
						"Candidates not inlude some testings." +
						"index[%s] < testPlaceIDOrders.length[%s]. candPlaceIDsNum=%s." +
						"TestUserID=%s, candPlaceIndices[0]=%s, testPlaceIDs[0]=%s",
						index, testPlaceIDOrders.length,
						candPlaceIDs.length, testUserID,
						candPlaceIndices.length == 0 ? "null" :
							candPlaceIndices[0],
						testPlaceIDs.length == 0 ? "null" :
							testPlaceIDs[0]));
					if (index == 0) continue;
					for (; index < testPlaceIDOrders.length; index ++) {
						testPlaceIDOrders[index] = -1;
					}
				}
				double ap    = 0;
				double auc   = 0;
				int[] hitNum = new int[Ks.length];
				int unrelNum = candPlaceIDs.length - testPlaceIDs.length;
				for (index = 0; index < testPlaceIDOrders.length; index ++) {
					if (index >= testPlaceIDOrders.length ||
						testPlaceIDOrders[index] == -1) {
						break;
					}
					int order = testPlaceIDOrders[index];
					for (int i = Ks.length - 1; i >= 0; i --) {
						if (order <= Ks[i] - 1) { // order from 0
						//if (thresholdProb <= placeProbs[testPlaceID]) {
							hitNum[i] ++;
						}
					}
					
					ap += (index + 1.0) / (order + 1.0);
					auc+= unrelNum - order + index;
				}
				for (int i = Ks.length - 1; i >= 0; i --) {
					if (Ks[i] < unrelNum + testPlaceIDs.length) {
						avgPrecision[i] += (hitNum[i] + 0.0) / (Ks[i] + 0.0);
						avgRecall[i]    += (hitNum[i] + 0.0) /
											(testPlaceIDs.length + 0.0);
						nUser4Precision[i] ++;
						nUser4Recall[i] ++;
					}
				}
				if (unrelNum != 0) {
					mauc[0] += auc / (testPlaceIDs.length * unrelNum);
					map[0] += ap / (testPlaceIDs.length + 0.0);
					nUser4AUC[0] ++;
					nUser4MAP[0] ++;
				}
			} // end for
		} // end run
	}

	private class PrecisionWithDynamicCandidatesThread extends Thread {
		private final int testUserStartIndex;
		private final int testUserEndIndex;
		private final int[] Ks;
		private final double[] avgPrecision;
		private final double[] avgRecall;
		private final double[] map;
		private final double[] mauc;
		private final int[] nUser4Precision;
		private final int[] nUser4Recall;
		private final int[] nUser4MAP;
		private final int[] nUser4AUC;

		public PrecisionWithDynamicCandidatesThread(int testStartIndex,
			int testEndIndex, int[] Ks, double avgPrecision[],
			double[] avgRecall, double[]map, double[]mauc, int[]nUser4Precision,
			int[]nUser4Recall, int[] nUser4MAP, int[] nUser4AUC){
			this.testUserStartIndex = testStartIndex;
			this.testUserEndIndex   = testEndIndex;
			this.Ks                  = Ks;
			this.avgPrecision       = avgPrecision;
			this.avgRecall          = avgRecall;
			this.map                = map;
			this.mauc               = mauc;
			this.nUser4Precision    = nUser4Precision;
			this.nUser4Recall       = nUser4Recall;
			this.nUser4MAP          = nUser4MAP;
			this.nUser4AUC          = nUser4AUC;
		}

		public void run() {
			System.out.println(String.format("TestStartIndex=%s, TestEndIndex=%s",
					testUserStartIndex, testUserEndIndex));
			for (int i = Ks.length - 1; i >= 0; i --) {
				avgPrecision[i] = 0;
				avgRecall[i]    = 0;
			}
			map[0] = 0;
			for (int testIndex = testUserStartIndex; testIndex <=testUserEndIndex; testIndex ++) {
				{
					int runNum = testIndex - testUserStartIndex;
					if (runNum % 1000 == 0) {
						if (runNum > 0) {
							System.out.println(String.format(
							"TestIndex=%s, precision_0=%s, recall_0=%s, map=%s",
							testIndex, (avgPrecision[0]/runNum),
							(avgRecall[0]/runNum), (map[0]/runNum)));
						} else {
							System.out.println("TestIndex=" + testIndex);
						}
					}
				}
				int testUserID     = testUserIDs[testIndex];
				int[] testPlaceIDs = testUserPlaceIDs[testIndex];
				Set<Integer> testPlaceIDSet = new HashSet<Integer>();
				for (int placeID : testPlaceIDs) 
					testPlaceIDSet.add(placeID);
				int[] hitNum      = new int[Ks.length];
				int[] realTestNum = new int[Ks.length];
				int mapRealTestNum= 0;
				int aucPairNum    = 0;
				double ap         = 0;
				double auc        = 0;
				for (int pIndex = 0; pIndex < testPlaceIDs.length; pIndex ++) {
					final int testPlaceID         = testPlaceIDs[pIndex];
					final int[] candPlaceIDs      = getCandidateIDs(
								testUserID, testPlaceID);
					final double[] candPlaceProbs = getCandidateProbs(
								testUserID, testPlaceID, candPlaceIDs);
					if (candPlaceIDs.length == 0) {
						System.err.println(String.format(
							"No candidates found for testUserID[%s].",
								testUserID));
					}
					Integer[] candPlaceIndices = new Integer[candPlaceIDs.length];
					for (int i = 0; i < candPlaceIndices.length; i ++) {
						candPlaceIndices[i] = i;
					}
				
					Arrays.sort(candPlaceIndices, new Comparator<Integer>(){
						@Override
						public int compare(Integer index1, Integer index2) {
							double diff = candPlaceProbs[index1] -
								candPlaceProbs[index2];
							if (diff > 0 ) return -1;
							if (diff < 0) return 1;
							return 0;
						}
					});
					
					int testPlaceIDOrder = -1;
					int indexInTestSet   = 0;
					int unrelNum         = candPlaceIDs.length;
					for (int order = 0; order < candPlaceIndices.length; order ++) {
						int placeID = candPlaceIDs[candPlaceIndices[order]];
						if (testPlaceIDSet.contains(placeID)) {
							if (placeID == testPlaceID) {
								if (testPlaceIDOrder != -1) {
									throw new RuntimeException("Duplicate : "
											+ testUserID + ", " + testPlaceID);
								}
								testPlaceIDOrder = order;
							} else
							if (testPlaceIDOrder != -1) {
								indexInTestSet ++;
							}
							unrelNum --;
						}
					}
					if (testPlaceIDOrder == -1) {
						throw new RuntimeException(String.format(
								"Not include in candidate: %s, %s",
									testUserID, testPlaceID));
					}
					for (int i = Ks.length - 1; i >= 0; i --) {
						if (Ks[i] < candPlaceIDs.length) {
							realTestNum[i] ++;
							if (testPlaceIDOrder <= Ks[i] - 1) { // order from 0
								hitNum[i] ++;
							}
						}
					}
					if (unrelNum > 0) {
						mapRealTestNum ++;
						ap  += (indexInTestSet + 1.0) / (testPlaceIDOrder + 1.0);
						auc += unrelNum - testPlaceIDOrder + indexInTestSet;
						aucPairNum += unrelNum;
					}
				} // end for pIndex
				for (int i = Ks.length - 1; i >= 0; i --) {
					if (realTestNum[i] > 0) {
						avgPrecision[i] += (hitNum[i] + 0.0) / (Ks[i] + 0.0);
						avgRecall[i]    += (hitNum[i] + 0.0) /
											(realTestNum[i] + 0.0);
						nUser4Precision[i] ++;
						nUser4Recall[i] ++;
					}
				}
				if (mapRealTestNum != 0) {
					mauc[0] += auc / (aucPairNum + 0.0);
					map[0]  += ap / (mapRealTestNum + 0.0);
					nUser4AUC[0] ++;
					nUser4MAP[0] ++;
				}
			} // end for
		} // end run
	}
}
