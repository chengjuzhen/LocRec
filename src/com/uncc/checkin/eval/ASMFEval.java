package com.uncc.checkin.eval;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Properties;

import com.uncc.checkin.CheckinConstants;
import com.uncc.checkin.CheckinException;
import com.uncc.checkin.ParamManager;
import com.uncc.checkin.amf.AMF;
import com.uncc.checkin.amf.ASMF;
import com.uncc.checkin.util.Utils;

/*
 * Options:
 * ASMFEval: -HomeLatLng <file> -LocLatLng <file> -LocLatLng <file> 
 * 		[Optional]
 * 			-GeoM <0:Add, 1:Mult> If Add is selected, -GeoFactor <float>
 *			-NewLocN <file>
 *			-TotLocNum <int>
 */
public class ASMFEval extends GeneralPrecEval {
	private final String STRING_NEW_LOC_NEIGHBOR      = "NewLocN";
	private final String STRING_TOTAL_LOC_NUM_INC_NEW = "TotLocNum";
	private final String STRING_GEO_METHOD            = "GeoM";
	private final String STRING_GEO_FACTOR            = "GeoFactor";
	
	private enum GeoMethod {
		MULT,
		ADD
	}
	
	private GeoMethod GM    = GeoMethod.ADD;
	private float geoFactor = 0.003f;
	
	private class APMFPrecisionEvaluation extends PrecisionEvaluator {
		private final double meanTrainRating;
		private final double[] userFeature;
		private final double[] itemFeature;
		private final double[] userCategoryPref;
		private final double[] userHomePowerLawParam;
		private final int[] categoryIndexForItem;
		private final int categoryNum;
		private final double[] categoryPrefOffsets;
		private final double[][] locationLatLngArr;
		private final double[][] userHomeLatLngArr;

		// for new item
		private double[] pairwisePowerLawParam;
		private int[] candidateItemIDs;
		private int[][] newItemLocationNeighbors;
		private double[][] candidateItemNeighborSims;
	
		public APMFPrecisionEvaluation(int userNum, int itemNum,
				File userHomeLatLngFile, File locationLatLngFile,
				File locationCategoryFile, boolean newItem,
				int totalLocationNumWithNewLocation,
				File newItemLocationNeighborFile) throws CheckinException {
			super(userNum, itemNum);
			
			meanTrainRating = params.containsKey(
					CheckinConstants.STRING_MEAN_TRAIN_RATING) ?
							Double.parseDouble(params.getProperty(
								CheckinConstants.STRING_MEAN_TRAIN_RATING)) : 0;
			if (meanTrainRating != 0) {
				throw new CheckinException("meanTrainRating[%s] != 0",
						meanTrainRating);
			}
			categoryNum = Integer.parseInt(params.getProperty(
							ASMF.STRING_CATEGORY_NUM));
			featureNum  = Integer.parseInt(params.getProperty(
							CheckinConstants.STRING_FEATURE_NUM));
			categoryPrefOffsets = Utils.loadDoubleArray(new File(
							modelPath,ASMF.FILE_NAME_USER_CATPREF_OFFSET));
			userFeature = Utils.convert2DTo1D(Utils.load2DoubleArray(
					new File(modelPath, CheckinConstants.STRING_USER_FEATURE),
						CheckinConstants.DELIMITER));
			itemFeature = Utils.convert2DTo1D(Utils.load2DoubleArray(
					new File(modelPath, CheckinConstants.STRING_ITEM_FEATURE),
						CheckinConstants.DELIMITER));
			userCategoryPref = Utils.convert2DTo1D(Utils.load2DoubleArray(
					new File(modelPath, ASMF.FILE_NAME_USER_CATEGORY_PREF),
						CheckinConstants.DELIMITER));
			categoryIndexForItem = AMF.loadItemCategory(
					locationCategoryFile, itemNum, categoryNum);

			userHomePowerLawParam  = Utils.loadPowerLawParams(
					new File(modelPath, AMF.FILE_NAME_HOME_POWERLAW_PARAM));
			userHomeLatLngArr     =
					Utils.loadLatLngArr(userHomeLatLngFile,userNum);
			
			if (newItem) {
				if (totalLocationNumWithNewLocation <= itemNum) {
					throw new CheckinException(
						"totalLocationNumWithNewLocation[%s] <= itemNum[%s]",
						totalLocationNumWithNewLocation, itemNum);
				}
				int newItemStartID    = itemNum;
				int newItemEndID      = totalLocationNumWithNewLocation - 1;
				pairwisePowerLawParam = Utils.loadPowerLawParams(
					new File(modelPath, ASMF.FILE_NAME_PAIRWISE_POWERLAW_PARAM));
				candidateItemIDs  = new int[newItemEndID - newItemStartID + 1];
				for (int i = 0; i < candidateItemIDs.length; i ++) {
					candidateItemIDs[i] = newItemStartID + i;
				}
				locationLatLngArr       = Utils.loadLatLngArr(
						locationLatLngFile, newItemEndID + 1);
				newItemLocationNeighbors = loadNewLocationNeighbors(
						newItemLocationNeighborFile,
							itemNum, newItemStartID, newItemEndID);
				calItemNeighborSims();
			} else {
				locationLatLngArr = Utils.loadLatLngArr(
										locationLatLngFile, itemNum);
			}
			
		}
	
		public double[] getPlaceProb(int testUserID) {
			double[] probs    = new double[itemNum];
			double[] userHome = userHomeLatLngArr[testUserID];
			
			if (GM.equals(GeoMethod.MULT)) {
				for (int itemId = 0; itemId < itemNum; itemId ++) {
					probs[itemId] = calProb(testUserID, itemId, userHome);
				}
				return probs;
			} else {
				double[] uiprob  = new double[itemNum];
				double[] geoprob = new double[itemNum];
				double uisum     = 0;
				double geosum    = 0;
				for (int itemId = 0; itemId < itemNum; itemId ++) {
					uiprob[itemId] = Utils.logisticFunction(
							ASMF.predict(userNum, itemNum, categoryNum,
								featureNum, userFeature, itemFeature,
									userCategoryPref, testUserID, itemId,
										categoryIndexForItem[itemId],
											categoryPrefOffsets[testUserID]));
					double[] latlng = locationLatLngArr[itemId];
					double distance = Utils.calDistance(userHome[0],
										userHome[1], latlng[0], latlng[1]);
					geoprob[itemId] = Utils.calPowerLawProb(
							userHomePowerLawParam[0], userHomePowerLawParam[1],
								AMF.zeroDistanceDefaultValue, distance);
					uisum  += uiprob[itemId];
					geosum += geoprob[itemId];
				}
				for (int itemId = 0; itemId < itemNum; itemId ++) {
					uiprob[itemId]  /= uisum;
					geoprob[itemId] /= geosum;
				}
				for (int itemId = 0; itemId < itemNum; itemId ++) {
					probs[itemId] = (1 - geoFactor) * uiprob[itemId] +
									geoFactor * geoprob[itemId];
				}
			}
			
			return probs;
		}
		@Override
		public int[] getCandidateIDs(int testUserID) {
			return candidateItemIDs;
		}
		@Override
		public double[] getCandidateProbs(int testUserID, int[]candidateIDs) {
			double[] probs      = new double[candidateIDs.length];
			double[] userHome   = userHomeLatLngArr[testUserID];
			double[] trainProbs = getPlaceProb(testUserID);
			for (int i = 0; i < candidateIDs.length; i ++) {
				int candidateID     = candidateIDs[i];
				int[] neighbors     = newItemLocationNeighbors[candidateID];
				double[] currLatLng = locationLatLngArr[candidateID];
				double prefSum      = 0;
				for (int index = 0; index < neighbors.length; index ++) {
					int neighborID  = neighbors[index];
					double prob     = trainProbs[neighborID];
					double sim      = candidateItemNeighborSims[i][index];
					prefSum += sim * prob;
				}
				double distance = Utils.calDistance(userHome[0],
					userHome[1], currLatLng[0], currLatLng[1]);
				probs[i] = Utils.logisticFunction(prefSum) *
						Utils.calPowerLawProb(userHomePowerLawParam[0],
						userHomePowerLawParam[1], AMF.zeroDistanceDefaultValue,
							distance);
			}
			return probs;
		}
	
		private void calItemNeighborSims() {
			System.out.println("CalItemNeighborSims");
			candidateItemNeighborSims = new double[candidateItemIDs.length][];
			for (int i = 0; i < candidateItemIDs.length; i ++) {
				int candidateID     = candidateItemIDs[i];
				int[] neighbors     = newItemLocationNeighbors[candidateID];
				double[] currLatLng = locationLatLngArr[candidateID];
				candidateItemNeighborSims[i] = new double[neighbors.length];
				double simSum       = 0;
				for (int index = 0; index < neighbors.length; index ++) {
					int neighborID  = neighbors[index];
					double[] latlng = locationLatLngArr[neighborID];
					double distance = Utils.calDistance(currLatLng[0],
							currLatLng[1], latlng[0], latlng[1]);
					double sim      = Utils.calPowerLawProb(
							pairwisePowerLawParam[0], pairwisePowerLawParam[1],
								AMF.zeroDistanceDefaultValue, distance);
					simSum         += sim;
					candidateItemNeighborSims[i][index] = sim;
				}
				if (simSum != 0) {
					for (int index = 0; index < neighbors.length; index ++) {
						candidateItemNeighborSims[i][index] /= simSum;
					}
				}
			}
		}
	
		private double calProb(int testUserID, int itemId, double[] userHome) {
			double v = ASMF.predict(userNum, itemNum, categoryNum, featureNum,
						userFeature, itemFeature, userCategoryPref, testUserID, itemId,
						categoryIndexForItem[itemId], categoryPrefOffsets[testUserID]);
				
			double[] latlng = locationLatLngArr[itemId];
			double distance = Utils.calDistance(userHome[0],
								userHome[1], latlng[0], latlng[1]);
			return Utils.logisticFunction(v) * Utils.calPowerLawProb(
					userHomePowerLawParam[0], userHomePowerLawParam[1],
					AMF.zeroDistanceDefaultValue, distance);
			
			//return v;
		}

	}

	public static int[][] loadNewLocationNeighbors(File locationNeighborFile,
			int locationNumInTrain, int newItemStartID,
			int newItemEndID) throws CheckinException {
		BufferedReader reader = null;
		try {
			reader      = new BufferedReader(new InputStreamReader(
				new FileInputStream(locationNeighborFile)));
			String line = null;
			int[][] locationNeighbors = new int[locationNumInTrain +
			           newItemEndID - newItemStartID + 1][];
			while ((line = reader.readLine()) != null) {
				String[] array = line.split(CheckinConstants.DELIMITER);
				if (array.length <= 1) {
					reader.close();reader=null;
					throw new CheckinException("Format Incorrect:: " + line);
				}
				int locationID  = Integer.parseInt(array[0]);
				if (locationID > newItemEndID) {
					reader.close(); reader=null;
					throw new CheckinException(
						"LocationID[%s], newItemStartID[%s], newItemEndID[%s]",
						locationID, newItemStartID, newItemEndID);
				} else
				if (locationID < newItemStartID) {
					continue;
				}
				int[] neighbors = new int[array.length - 1];
				// skip the first element
				for (int i = 1; i < array.length; i ++) {
					int ID = Integer.parseInt(array[i]);
					if (ID >= locationNumInTrain) {
						reader.close();reader=null;
						throw new CheckinException(
							"LocationID[%s], ID[%s], locationNumInTrain[%s]",
							locationID, ID, locationNumInTrain);
					}
					neighbors[i - 1] = ID;
				}
				locationNeighbors[locationID] = neighbors;
			}
			reader.close();
			for (int locationID = 0; locationID < locationNeighbors.length; locationID ++) {
				if (locationID < newItemStartID) {
					locationNeighbors[locationID] = null;
				} else
				if (locationNeighbors[locationID] == null ||
					locationNeighbors[locationID].length == 0) {
					throw new CheckinException(
							"No found locationID[%s]", locationID);
				}
			}
			return locationNeighbors;
		} catch (IOException e) {
			e.printStackTrace();
			throw new CheckinException(e);
		} finally {
			Utils.cleanup(reader);
		}
	}
	
	public ASMFEval(ParamManager paramManager) throws CheckinException,
											com.lhy.tool.ToolException{
		super(paramManager);
		
		// parse options
		Properties options = Utils.parseCMD(paramManager.getOptions()
				.getProperty(paramManager.getModelName()));
		
		Utils.check(options, ASMF.STRING_LOCATION_CAT_FILE);
		File locationCategoryFile = new File(options.getProperty(
									ASMF.STRING_LOCATION_CAT_FILE));
		Utils.check(options, ASMF.STRING_LOCATION_LATLNG_FILE);
		File locationLatLngFile   = new File(options.getProperty(
									ASMF.STRING_LOCATION_LATLNG_FILE));
		Utils.check(options, ASMF.STRING_USER_HOME_LATLNG_FILE);
		File userHomeLatLngFile   = new File(options.getProperty(
									ASMF.STRING_USER_HOME_LATLNG_FILE));
		
		File newLocationNeighborFile        = null;
		boolean newItemEvaluation           = false;
		int totalLocationNumWithNewLocation = -1;
		if (options.get(STRING_NEW_LOC_NEIGHBOR) != null) {
			newItemEvaluation       = true;
			newLocationNeighborFile = new File(options.getProperty(
											STRING_NEW_LOC_NEIGHBOR));
			Utils.check(options, STRING_TOTAL_LOC_NUM_INC_NEW);
			totalLocationNumWithNewLocation = Integer.parseInt(
					options.getProperty(STRING_TOTAL_LOC_NUM_INC_NEW));
			System.out.println("\n+++++++++++++++++ Location Cold Start Evaluation +++++++++++");
			System.out.println("NewLocationNeighborFile         : " +
					newLocationNeighborFile.getAbsolutePath());
			System.out.println("TotalLocationNumWithNewLocation : " +
					totalLocationNumWithNewLocation);
			System.out.println("++++++++++++++++ End Location Cold Start Evaluation ++++++++++\n");
		}
		
		if (options.containsKey(STRING_GEO_METHOD)) {
			int choice = Integer.parseInt(options.getProperty(STRING_GEO_METHOD));
			switch (choice) {
				case 0: 
					GM = GeoMethod.ADD;
					break;
				case 1:
					GM = GeoMethod.MULT;
					break;
				default:
					throw new CheckinException("Unknown option for " + STRING_GEO_METHOD);
			}
		}
		if (options.containsKey(STRING_GEO_FACTOR)) {
			geoFactor = Float.parseFloat(options.getProperty(STRING_GEO_FACTOR));
		}
		
		System.out.println("UserHomeLatLngFile  : " +
							userHomeLatLngFile.getAbsolutePath());
		System.out.println("LocationLatLngFile  : " +
							locationLatLngFile.getAbsolutePath());
		System.out.println("LocationCategoryFile: " +
							locationCategoryFile.getAbsolutePath());
		System.out.println("GeoMethod: " + GM.toString());
		if (GM == GeoMethod.ADD) System.out.println("GeoFactor: " + geoFactor);
		
		
		// calculate Precision, Recall, MAP, AUC
		File precisionOutputFile = new File(outputPath, "Precision_Result");
		
		PrecisionEvaluator.MODE evalMode   = newItemEvaluation ?
				PrecisionEvaluator.MODE.CANDIDATES:PrecisionEvaluator.MODE.ALL;
		APMFPrecisionEvaluation evaluation = null;
		if (newItemEvaluation) {
			evaluation = new APMFPrecisionEvaluation(userNum, itemNum,
				userHomeLatLngFile, locationLatLngFile,
					locationCategoryFile, true, totalLocationNumWithNewLocation,
						newLocationNeighborFile);
		} else {
			evaluation = new APMFPrecisionEvaluation(userNum, itemNum,
							userHomeLatLngFile, locationLatLngFile,
								locationCategoryFile, false, -1, null);
		}
		evaluation.evaluatePrecisionRecallMapInMultiThread(
			trainRatingFile, testRatingFile, KArray, precisionOutputFile,
				threadNum, evalMode);
	}
	
	public static void main(String[] args) throws CheckinException,
							com.lhy.tool.ToolException{
		new ASMFEval(CheckinConstants.CONFIG_MANAGER);
	}
}
