package com.uncc.checkin.eval;

import java.io.File;

import com.lhy.tool.ToolException;
import com.uncc.checkin.CheckinConstants;
import com.uncc.checkin.CheckinException;
import com.uncc.checkin.ParamManager;
import com.uncc.checkin.eval.PrecisionEvaluator;
import com.uncc.checkin.util.Utils;
import com.uncc.checkin.eval.GeneralPrecEval;

public class MFEval extends GeneralPrecEval {
	private double meanTrainRating      = 0;
	private double[] userFeature       = null;
	private double[] itemFeature       = null;
	
	private double predict(int userId, int itemId) {
		int userStartIndex   = userId * featureNum;
		int itemStartIndex   = itemId * featureNum;
		double predictRating = 0;
		for (int featureId = 0; featureId < featureNum; featureId ++) {
			predictRating += userFeature[userStartIndex + featureId] *
					itemFeature[itemStartIndex + featureId];
		}
		if (meanTrainRating != 0) {
			predictRating += meanTrainRating;
		}
		return predictRating;
	}

	private class MFPrecisionEvaluation extends PrecisionEvaluator {
		
		public MFPrecisionEvaluation(int userNum, int itemNum) {
			super(userNum, itemNum);
		}
		
		public double[] getPlaceProb(int testUserID) {
			double[] probs = new double[itemNum];
			for (int itemId = 0; itemId < itemNum; itemId ++) {
				probs[itemId] = predict(testUserID, itemId);
			}
			return probs;
		}
		@Override
		public int[] getCandidateIDs(int testUserID) {
			// to do nothing
			throw new RuntimeException("Never reaches:: getCandidateIDs.");
		}
		@Override
		public double[] getCandidateProbs(int testUserID, int[]candidateIDs) {
			// to do nothing
			throw new RuntimeException("Never reaches:: getCandidateProbs.");
		}
	}

	public MFEval(ParamManager paramManager) throws CheckinException,
														ToolException{
		super(paramManager);
		
		// load values
		if (params.getProperty(CheckinConstants.STRING_MEAN_TRAIN_RATING) != null) {
			meanTrainRating = Float.parseFloat(params.getProperty(
								CheckinConstants.STRING_MEAN_TRAIN_RATING));
		}
		userFeature = Utils.convert2DTo1D(Utils.load2DoubleArray(
						new File(modelPath,CheckinConstants.STRING_USER_FEATURE),
								CheckinConstants.DELIMITER));
		itemFeature = Utils.convert2DTo1D(Utils.load2DoubleArray(
						new File(modelPath,CheckinConstants.STRING_ITEM_FEATURE),
								CheckinConstants.DELIMITER));
	
		// Precision, Recall, MAP, AUC
		File precisionOutputFile = new File(outputPath, "Precision_Result");
		new MFPrecisionEvaluation(userNum, itemNum)
			.evaluatePrecisionRecallMapInMultiThread(trainRatingFile,
					testRatingFile, KArray, precisionOutputFile,
						threadNum, PrecisionEvaluator.MODE.ALL);
	}
	
	public static void main(String[] args) throws CheckinException, ToolException{
		if (args.length > 0) {
			System.out.println("No options.");
			return;
		}
		new MFEval(CheckinConstants.CONFIG_MANAGER);
	}
}
