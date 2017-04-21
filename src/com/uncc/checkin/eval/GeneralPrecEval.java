package com.uncc.checkin.eval;

import java.io.File;
import java.util.Properties;

import com.uncc.checkin.CheckinConstants;
import com.uncc.checkin.CheckinException;
import com.uncc.checkin.ParamManager;
import com.uncc.checkin.util.Utils;

public abstract class GeneralPrecEval {
	protected File modelPath       = null;
	protected File outputPath      = null;
	protected File trainRatingFile = null;
	protected File testRatingFile  = null;
	// built-in with default value
	protected int threadNum        = 4;
	protected int[] KArray         = {5, 8, 10, 12, 15, 20};
	// param file
	protected Properties params = null;
	protected int userNum       = -1;
	protected int itemNum       = -1;
	protected int featureNum    = -1;
	
	public GeneralPrecEval(ParamManager paramManager) throws CheckinException {
		modelPath  = new File(new File(paramManager.getOutputPath(),
								paramManager.getModelName()),
								String.valueOf(paramManager.getFeatureNum()));
		outputPath = new File(new File(paramManager.getOutputPath(),
								paramManager.getModelName()),
								String.valueOf(paramManager.getFeatureNum()));
		
		trainRatingFile = paramManager.getTrainFile();
		testRatingFile  = paramManager.getTestFile();
		
		Utils.exists(modelPath);
		System.out.println("[Info] ModelName      : " +
							paramManager.getModelName());
		System.out.println("[Info] ModelPath      : " +
							modelPath.getAbsolutePath());
		System.out.println("[Info] OutputPath     : " +
							outputPath.getAbsolutePath());
		System.out.println("[Info] TrainRatingFile: " +
							trainRatingFile.getAbsolutePath());
		System.out.println("[Info] TestRatingFile : " +
							testRatingFile.getAbsolutePath());
		
		// parse properties
		parseProperties(paramManager);
		// parse param file
		parseParams(paramManager);
	}
	
	private void parseParams(ParamManager paramManager) throws CheckinException{
		params     = Utils.loadParams(new File(modelPath,
								CheckinConstants.STRING_PARAM));
		userNum    = Integer.parseInt(params.getProperty(
								CheckinConstants.STRING_USER_NUM));
		itemNum    = Integer.parseInt(params.getProperty(
								CheckinConstants.STRING_ITEM_NUM));
		featureNum = Integer.parseInt(params.getProperty(
								CheckinConstants.STRING_FEATURE_NUM));
		if (featureNum != paramManager.getFeatureNum()) {
			throw new CheckinException("Not match for featureNum: %s, %s",
					paramManager.getFeatureNum());
		}
	}
	
	
	private void parseProperties(ParamManager paramManager)
										throws CheckinException{
		try {
			String sprop = paramManager.getEvalKArrayStr();
			if (sprop != null) {
				KArray = Utils.parseIntArray(sprop, ",");
			}
			System.out.println("** Eval K Array **");
			print(KArray);
			
			sprop = paramManager.getEvalThreadNum();
			if (sprop != null) {
				threadNum = Integer.parseInt(sprop);
			}
			System.out.println("ThreadNum: " + threadNum);
		} catch (NumberFormatException e) {
			throw new CheckinException(e);
		}
	}
	
	private void print(int[] array) {
		for (int elem : array) {
			System.out.print(elem + "\t");
		}
		System.out.println();
	}
}
