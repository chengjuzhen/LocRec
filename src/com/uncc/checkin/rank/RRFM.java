package com.uncc.checkin.rank;

import com.uncc.checkin.CheckinConstants;
import com.uncc.checkin.CheckinException;
import com.uncc.checkin.ParamManager;
import com.uncc.checkin.fw.BasicRankMF;
import com.uncc.checkin.util.Utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

/*
 * Optional:
 * 	-ItemGroupFile <path> (which is conflict with -GroupNum)
 */
public class RRFM extends BasicRankMF {
	private static final String STRING_ENH_GROUP_NUM       = "EnhGroupNum";
	private static final String STRING_GROUP_NUM           = "GroupNum";
	private static final String STRING_ITEM_GROUP_FILE     = "ItemGroupFile";
	private static final String FILE_NAME_USER_PREF_MARGIN = "User_Pref_Margin";
	
	protected double[] itemFeature  = null;
	protected double[] userFeature  = null;
	protected double[] itemGroupSum = null;
	
	protected double[] observedMeanNegV = null;
	protected int[] itemNumInGroupOrder = null;
	protected double[] itemPopuSumWeightsInGroupOrder        = null;
	protected double[] itemPopuSumWeightsPerGroupInUserOrder = null;
	
	protected int[] itemMappedGroupIds             = null;
	protected int[] itemGroupIndexInUserOrder      = null;
	protected int[] itemGroupNumInUserOrder        = null;
	protected int[] itemNumPerGroupInUserOrder     = null;
	protected int[] userGroupStartIndexInUserOrder = null;
	protected int[] mappedUIndexInItemOrder        = null;
	protected boolean[] observedMeanNegVStatus     = null;
	protected double[] userPrefMargin              = null;
	
	protected int ITEM_GROUP_NUM;
	protected int ITEM_GROUP_RECORD_NUM;
	private int FEATURE_NUM_X_GROUP_NUM;
	
	private final float lambdaPrefMargin = 0.01f;
	protected float lambdaItem  = 0.01f;
	protected float lambdaUser  = 0.01f;
	protected float smoothAlpha = 5f;
	
	private float positiveRating   = 1;
	private float negativeRating   = 0;
	private float posNegRatingDiff = positiveRating - negativeRating;
	
	private int enhGroupNum		   = -1;
	private int[][] enhUserItemIDs = null;
	private int[][] enhItemUserIDs = null;
	
	protected Function smoothObjFunc  = new SmoothObjFunction();
	protected Function smoothGradFunc = new SmoothGradFunction();
	
	public RRFM(ParamManager paramManager) throws CheckinException {
		this.FORCE_CHECK = false;
		init(paramManager);
	}

	@Override
	protected Properties getParams() {
		Properties props = super.getParams();
		props.put(CheckinConstants.STRING_LAMBDA_ITEM,	String.valueOf(lambdaItem));
		props.put(CheckinConstants.STRING_LAMBDA_USER,	String.valueOf(lambdaUser));
		props.put(STRING_ENH_GROUP_NUM, 	String.valueOf(enhGroupNum));
		props.put(STRING_GROUP_NUM, 		String.valueOf(ITEM_GROUP_NUM));
		props.put("ItemGroupRecordNum", 	String.valueOf(ITEM_GROUP_RECORD_NUM));
		props.put("SmoothAlpha",			String.valueOf(smoothAlpha));
		props.put("LambdaUserPrefMargin",	String.valueOf(lambdaPrefMargin));
		return props;
	}
	
	@Override
	protected String getMethodName() {
		return CheckinConstants.STRING_MODEL_RRFM;
	}

	@Override
	protected void initModel(ParamManager paramManager) throws CheckinException  {
		lambdaUser = paramManager.getLambdaUser();
		lambdaItem = paramManager.getLambdaItem();
		
		// parse args
		Properties args = Utils.parseCMD(paramManager.getModelRRMF());
		
		// EnhGroupNum
		enhGroupNum = Integer.parseInt(args.getProperty(STRING_ENH_GROUP_NUM));
		
		// create group file
		File itemGroupFile = new File(outputPath, "ItemGroup");//new File("C:/Users/huayuli/Dropbox/code/librec/librec/Results/RRFMgi/ItemGroup");
		if (args.get(STRING_ITEM_GROUP_FILE) != null) {
			itemGroupFile = new File((String)args.get(STRING_ITEM_GROUP_FILE));
			System.out.println("Load ItemGroupFile : " + itemGroupFile.getAbsolutePath());
		} else {
			int specficGroupNum = Integer.parseInt(args.getProperty(STRING_GROUP_NUM));
			createItemGroup(paramManager.getTrainFile(), ITEM_NUM,
								specficGroupNum, itemGroupFile);
			System.out.println("Create ItemGroupFile : " + itemGroupFile.getAbsolutePath());
		}
		
		// load group info
		loadItemGroupFile(itemGroupFile);
		parseTrainRatingFile_RRMF(paramManager.getTrainFile());
		
		FEATURE_NUM_X_GROUP_NUM = FEATURE_NUM * ITEM_GROUP_NUM;
		
		userFeature    = initFeature(USER_NUM * FEATURE_NUM,
							paramManager.getUserFeatureInitFile());
				//new File("C:/Users/huayuli/Dropbox/code/librec/librec/Results/RRFMgi/UserFeature"));
		itemFeature    = initFeature(ITEM_NUM * FEATURE_NUM,
							paramManager.getItemFeatureInitFile());
				//new File("C:/Users/huayuli/Dropbox/code/librec/librec/Results/RRFMgi/ItemFeature"));
		userPrefMargin = initUserPrefMargin(USER_NUM * ITEM_GROUP_NUM,
							 null);
				//new File("C:/Users/huayuli/Dropbox/code/librec/librec/Results/RRFMgi/UserPrefMargin"));
		
		initMappedUIndexInItemOrder();

		prepareItemPopuWeights();
		updateItemGroupSum();
		updateObservedMeanNegV();
		
		//initEnhGroups();
	}
	
	@Override
	protected double predict(int userId, int itemId) {
		int userStartIndex   = userId * FEATURE_NUM;
		int itemStartIndex   = itemId * FEATURE_NUM;
		double predictRating = 0;
		for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
			predictRating += userFeature[userStartIndex + featureId] *
					itemFeature[itemStartIndex + featureId];
		}
		return predictRating;
	}

	@Override
	protected double calLoss() throws CheckinException {
		double lbvalue          = 0;
		int recordStartIndex    = 0;
		int userStartIndex      = 0;
		int userPrefMarginIndex = 0;
		int groupRecordIndex    = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int groupNum = itemGroupNumInUserOrder[userId];
			int itemNum  = itemNumInUserOrder[userId];
			lbvalue += getPosGradRatingPerUser(userStartIndex,
						recordStartIndex, itemNum, null, smoothObjFunc);
			lbvalue += getNegGradRatingPerUser(userId, userStartIndex,
						userPrefMarginIndex, groupRecordIndex, groupNum,
							null, smoothObjFunc);
			recordStartIndex    += itemNum;
			userStartIndex      += FEATURE_NUM;
			userPrefMarginIndex += ITEM_GROUP_NUM;
			groupRecordIndex    += groupNum;	
		}
		if (recordStartIndex != RATING_RECORD_NUM)
			throw new RuntimeException(String.format(
				"recordIndex[%s] != ratingsInUserOrder.length[%s]",
				recordStartIndex, RATING_RECORD_NUM));
		if (userStartIndex != userFeature.length)
			throw new RuntimeException(String.format(
				"userStartIndex[%s] != USER_NUM * FEATURE_NUM[%s]",
				userStartIndex, userFeature.length));
		if (userPrefMarginIndex != USER_NUM * ITEM_GROUP_NUM)
			throw new RuntimeException(String.format(
				"userPrefMarginIndex[%s] != userPrefMargin.length[%s]",
				userPrefMarginIndex, USER_NUM * ITEM_GROUP_NUM));
		if (groupRecordIndex != ITEM_GROUP_RECORD_NUM)
			throw new RuntimeException(String.format(
				"GroupRecordIndex[%s]!=ITEM_GROUP_RECORD_NUM[%s]",
				groupRecordIndex, ITEM_GROUP_RECORD_NUM));
		if (Double.isInfinite(lbvalue) || Double.isNaN(lbvalue))
			throw new CheckinException("RankObj:: lbvalue=" + lbvalue);
		
		/*lbvalue += 0.5 * lambdaUser * Utils.cal2norm(userFeature);
		if (Double.isInfinite(lbvalue) || Double.isNaN(lbvalue))
			throw new CheckinException("U'U:: " + lbvalue);
		lbvalue += 0.5 * lambdaItem * Utils.cal2norm(itemFeature);
		if (Double.isInfinite(lbvalue) || Double.isInfinite(lbvalue))
			throw new CheckinException("V'V:: " + lbvalue);
		
		double sum = 0;
		for (double v : userPrefMargin){
			sum += v - 1;
		}
		lbvalue += lambdaPrefMargin * sum; //Utils.cal1norm(userPrefMargin);
		if (Double.isInfinite(lbvalue) || Double.isInfinite(lbvalue))
			throw new CheckinException("prefMargin:: " + lbvalue);*/
		 
		return lbvalue;
	}

	@Override
	protected void update1Iter() throws CheckinException {
		initEnhGroups();
		
		updateItemFeature();
		updateItemGroupSum();
		updateObservedMeanNegV();
		
		updateUserFeature();
		
		updateUserPrefMargin();
	}

	private void updateUserFeature() {
		int recordStartIndex         = 0;
		int userStartIndex           = 0;
		int userPrefMarginStartIndex = 0;
		int groupRecordStartIndex    = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int groupNum = itemGroupNumInUserOrder[userId];
			int itemNum  = itemNumInUserOrder[userId];
			double[] V   = new double[FEATURE_NUM];
			
			getPosGradRatingPerUser(userStartIndex, recordStartIndex,
					itemNum, V, smoothGradFunc);
			getNegGradRatingPerUser(userId, userStartIndex,
					userPrefMarginStartIndex, groupRecordStartIndex,
					groupNum, V, smoothGradFunc);
			
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				double value = userFeature[userStartIndex];
				userFeature[userStartIndex ++] = value - learningRate*(
					V[featureId] + lambdaUser * value);
			}
			recordStartIndex         += itemNum;
			groupRecordStartIndex    += groupNum;
			userPrefMarginStartIndex += ITEM_GROUP_NUM;
		}
		if (recordStartIndex != RATING_RECORD_NUM)
			throw new RuntimeException(String.format(
				"recordIndex[%s] != ratingsInUserOrder.length[%s]",
				recordStartIndex, RATING_RECORD_NUM));
		if (userStartIndex != userFeature.length)
			throw new RuntimeException(String.format(
				"userStartIndex[%s] != USER_NUM * FEATURE_NUM[%s]",
				userStartIndex, userFeature.length));
		if (userPrefMarginStartIndex != USER_NUM * ITEM_GROUP_NUM)
			throw new RuntimeException(String.format(
				"userPrefMarginIndex[%s] != userPrefMargin.length[%s]",
				userPrefMarginStartIndex, USER_NUM * ITEM_GROUP_NUM));
		if (groupRecordStartIndex != ITEM_GROUP_RECORD_NUM)
			throw new RuntimeException(String.format(
				"GroupRecordIndex[%s]!=ITEM_GROUP_RECORD_NUM[%s]",
				groupRecordStartIndex, ITEM_GROUP_RECORD_NUM));
	}

	private void updateItemFeature() {
		double[] UPerUser = new double[ITEM_GROUP_RECORD_NUM * FEATURE_NUM];
		double[] USum     = new double[FEATURE_NUM_X_GROUP_NUM];
		
		getNegGradPerGroupUser(UPerUser, USum, smoothGradFunc);
			
		int recordIndex    = 0;
		int itemStartIndex = 0;
		boolean checkMappedIndex[] = new boolean[ITEM_GROUP_RECORD_NUM];
		for (int itemId = 0; itemId < ITEM_NUM; itemId ++) {
			int userNum = userNumInItemOrder[itemId];
			int groupId = itemMappedGroupIds[itemId];
			double U[]  = new double[FEATURE_NUM];
			for(int i = 0; i < userNum; i ++) {
				int userId      = userIndexInItemOrder[recordIndex];
				int mappedIndex = mappedUIndexInItemOrder[recordIndex];
				recordIndex ++;
				{
					if (mappedIndex % FEATURE_NUM != 0)
						throw new RuntimeException("mappedIndex=" + mappedIndex);
					checkMappedIndex[mappedIndex/FEATURE_NUM] = true;
				}
				int userStartIndex = userId * FEATURE_NUM;
				double rjh         = 0;
				for (int featureId=0;featureId<FEATURE_NUM; featureId ++) {
					rjh += userFeature[userStartIndex + featureId] *
						itemFeature[itemStartIndex + featureId];
				}
				rjh =  smoothGradFunc.getFunction(positiveRating - rjh);
				for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
					U[featureId] -= rjh * userFeature[userStartIndex + featureId]
							+ UPerUser[mappedIndex ++];
				}
			}
			
			///// Enhance groups //////////////////////////////////
			int[] enhUserIDs = enhItemUserIDs[itemId];
			if (enhUserIDs != null) {
				for (int i = 0; i < enhUserIDs.length; i ++) {
					int enhUserId      = enhUserIDs[i];
					int userStartIndex = enhUserId * FEATURE_NUM;
					double rij = 0;
					for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
						rij += userFeature[userStartIndex + featureId] *
							itemFeature[itemStartIndex + featureId];
					}
					int marginIndex = enhUserId * ITEM_GROUP_NUM + groupId;
					rij = smoothGradFunc.getFunction(rij -
							negativeRating - userPrefMargin[marginIndex]);
					for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
						U[featureId] += rij * userFeature[userStartIndex + featureId];
					}
				}
			}
			///////////////////////////////////////////////////////
			
			int index = groupId * FEATURE_NUM;
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				double value = U[featureId] + USum[index ++];
				double vf    = itemFeature[itemStartIndex];
				vf          -= learningRate* (value + lambdaItem * vf);
				//if (vf < 0) vf = 0;
				itemFeature[itemStartIndex ++] = vf;
			}
		}
		if (recordIndex != RATING_RECORD_NUM)
			throw new RuntimeException(String.format(
				"recordIndex[%s] != RATING_RECORD_NUM[%s]",
				recordIndex, RATING_RECORD_NUM));
		if (itemStartIndex != ITEM_NUM * FEATURE_NUM)
			throw new RuntimeException(String.format(
				"itemStartIndex[%s] != ITEM_NUM[%s] * FEATURE_NUM[%s]",
				itemStartIndex, ITEM_NUM, FEATURE_NUM));
		for (int i = 0; i < checkMappedIndex.length; i ++)
			if (!checkMappedIndex[i]) throw new RuntimeException(
				"do not use record " + i);
	}

	private void updateUserPrefMargin() {
		int userStartIndex      = 0;
		int groupRecordIndex    = 0;
		int userPrefMarginIndex = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int groupNum   = itemGroupNumInUserOrder[userId];
			int groupIndex = 0;
			double[] G     = new double[ITEM_GROUP_NUM];
			for (int groupId = 0; groupId < ITEM_GROUP_NUM; groupId ++) {
				double rih    = 0;
				double upm    = userPrefMargin[userPrefMarginIndex + groupId];
				boolean isCal = true;
				if (groupIndex < groupNum && groupId ==
						itemGroupIndexInUserOrder[groupRecordIndex]) {
					int groupFeatureIndex = FEATURE_NUM * groupRecordIndex;
					if (observedMeanNegVStatus[groupRecordIndex]) {
						groupIndex ++;
						groupRecordIndex ++;
						for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
							double value = observedMeanNegV[groupFeatureIndex++];
							rih += value * userFeature[userStartIndex + featureId];
						}
					} else {
						groupIndex ++;
						groupRecordIndex ++;
						isCal = false;
					}
				} else {
					double weightSum  = itemPopuSumWeightsInGroupOrder[groupId] + 0.0;
					if (weightSum != 0) {
						int gSumIndex  = groupId * FEATURE_NUM;
						for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
							double value = itemGroupSum[gSumIndex++] / weightSum;
							rih += value * userFeature[userStartIndex + featureId];
						}
					} else {
						isCal = false;
						throw new RuntimeException("Inner error: never reaches.");
					}
				}
				if (isCal) {
					rih = -1 * smoothGradFunc.getFunction(rih - negativeRating - upm);
				}
				G[groupId] = rih;
			}
			//////////////////Enhance groups /////////////////////////
			int[] enhItemIDs = enhUserItemIDs[userId];
			for (int i = 0; i < enhItemIDs.length; i ++) {
				int itemId         = enhItemIDs[i];
				int groupId        = itemMappedGroupIds[itemId];
				int itemStartIndex = itemId * FEATURE_NUM;
				double rij = 0;
				for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
					rij += userFeature[userStartIndex + featureId] *
						itemFeature[itemStartIndex + featureId];
				}
				double enhupm = userPrefMargin[userPrefMarginIndex + groupId];
				rij = -1 * smoothGradFunc.getFunction(rij - negativeRating - enhupm);
				G[groupId] += rij;
			} // end for (int groupId)
			//////////////////////////////////////////////////////////
			for (int groupId = 0; groupId < ITEM_GROUP_NUM; groupId ++) {
				double upm = userPrefMargin[userPrefMarginIndex + groupId];
				upm       -= learningRate *(G[groupId] + lambdaPrefMargin);
				if (upm < posNegRatingDiff) upm = posNegRatingDiff;
		
				userPrefMargin[userPrefMarginIndex + groupId] = upm;
			}
			userStartIndex      += FEATURE_NUM;
			userPrefMarginIndex += ITEM_GROUP_NUM;
			if (groupIndex != groupNum)
				throw new RuntimeException(String.format(
					"groupIndex[%s] != groupNum[%s]", groupIndex, groupNum));
		}
		if (userStartIndex != USER_NUM * FEATURE_NUM)
			throw new RuntimeException(String.format(
				"userStartIndex[%s] != USER_NUM[%s] * FEATURE_NUM[%s]",
				userStartIndex, USER_NUM, FEATURE_NUM));
		if (groupRecordIndex != ITEM_GROUP_RECORD_NUM)
			throw new RuntimeException(String.format(
				"groupRecordIndex[%s] != ITEM_GROUP_RECORD_NUM[%s]",
				groupRecordIndex, ITEM_GROUP_RECORD_NUM));
		if (userPrefMarginIndex != USER_NUM * ITEM_GROUP_NUM)
			throw new RuntimeException(String.format(
				"userPrefMarginIndex[%s] != USER_NUM[%s] * ITEM_GROUP_NUM[%s]",
				userPrefMarginIndex, USER_NUM, ITEM_GROUP_NUM));
	}
	
	protected void updateItemGroupSum() {
		itemGroupSum = new double[FEATURE_NUM_X_GROUP_NUM];
		int itemIndex = 0;
		for (int itemId = 0; itemId < ITEM_NUM; itemId ++) {
			int groupId = itemMappedGroupIds[itemId];
			int index   = groupId * FEATURE_NUM;
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				itemGroupSum[index++] += itemFeature[itemIndex ++];
			}
		}
		//Utils.print(itemGroupSum, 1, itemGroupSum.length, ",");
		if (itemIndex != ITEM_NUM * FEATURE_NUM) {
			throw new RuntimeException(String.format(
				"InnerError::itemIndex[%s]!=ITEM_NUM[%s]*FEATURE_NUM[%s]",
				itemIndex, ITEM_NUM, FEATURE_NUM));
		}
	}

	protected void updateObservedMeanNegV() {
		observedMeanNegV      = new double[ITEM_GROUP_RECORD_NUM * FEATURE_NUM];
		observedMeanNegVStatus= new boolean[ITEM_GROUP_RECORD_NUM];
		int recordIndex       = 0;
		int groupRecordIndex  = 0;
		int groupFeatureIndex = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int groupNum = itemGroupNumInUserOrder[userId];
			for (int i = 0; i < groupNum; i ++) {
				int itemNum = itemNumPerGroupInUserOrder[groupRecordIndex];
				int groupId = itemGroupIndexInUserOrder[groupRecordIndex];
				int allItemNumPerGroup = itemNumInGroupOrder[groupId];
				if (allItemNumPerGroup > itemNum) {
					double[] V  = new double[FEATURE_NUM];
					double wSum = 0;
					for (int jj = 0; jj < itemNum; jj ++) {
						int itemId    = itemIndexInUserOrder[recordIndex + jj];
						int itemIndex = FEATURE_NUM * itemId;
						wSum ++;
						for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
							V[featureId] += itemFeature[itemIndex ++];
						}
					}
					wSum = itemPopuSumWeightsInGroupOrder[groupId] - wSum + 0.0;
					if (wSum != 0) {
						int gVSumIndex = groupId * FEATURE_NUM;
						for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
							observedMeanNegV[groupFeatureIndex + featureId] =
								( itemGroupSum[gVSumIndex ++] -
									V[featureId] ) / wSum;
						}
						observedMeanNegVStatus[groupRecordIndex] = true;
					}
				} else
				if (allItemNumPerGroup == itemNum){
					/*System.err.println(String.format(
						"allItemNumPerGroup[%s] == itemNum[%s]",
						allItemNumPerGroup, itemNum));
					observedMeanNegVStatus[groupRecordIndex] = false;*/
				} else {
					// never reaches
					throw new RuntimeException(String.format(
						"UpdateObservedMeanNegV::UserId=%s,allItemPerGroup=%s, itemNum=%s",
						userId, allItemNumPerGroup, itemNum));
				}
				groupRecordIndex ++;
				recordIndex       += itemNum;
				groupFeatureIndex += FEATURE_NUM;
			} // end for (int i = 0; i < groupNum; i ++) {
		} // end for (int userId = 0; ...)
		if (recordIndex != RATING_RECORD_NUM) {
			throw new RuntimeException(String.format(
				"recordIndex[%s] != RATING_RECORD_NUM[%s]",
				recordIndex, RATING_RECORD_NUM));
		}
		if (groupRecordIndex != ITEM_GROUP_RECORD_NUM) {
			throw new RuntimeException(String.format(
				"groupRecordIndex[%s] != ITEM_GROUP_RECORD_NUM[%]",
				groupRecordIndex, ITEM_GROUP_RECORD_NUM));
		}
		if (groupFeatureIndex != ITEM_GROUP_RECORD_NUM * FEATURE_NUM) {
			throw new RuntimeException(String.format(
				"groupFeatureIndex[%s] != ITEM_GROUP_RECORD_NUM[%s] * FEATURE_NUM[%s]",
				groupFeatureIndex, ITEM_GROUP_RECORD_NUM, FEATURE_NUM));
		}
	}

	private double[][] getMeanNegRatingInUserGroupOrder() {
		int userStartIndex     = 0;
		int groupRecordIndex   = 0;
		double[][] meanRatings = new double[USER_NUM][ITEM_GROUP_NUM];
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int groupNum   = itemGroupNumInUserOrder[userId];
			int groupIndex = 0;
			for (int groupId = 0; groupId < ITEM_GROUP_NUM; groupId ++) {
				double meanRating = 0; // maybe set with a large value
				if (groupIndex < groupNum && groupId ==
					itemGroupIndexInUserOrder[groupRecordIndex]) {
					int groupFeatureIndex = FEATURE_NUM * groupRecordIndex;
					if (observedMeanNegVStatus[groupRecordIndex]) {
						groupIndex ++;
						groupRecordIndex ++;
						for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
							double value = observedMeanNegV[groupFeatureIndex++];
							meanRating += value * userFeature
								[userStartIndex + featureId];
						}
					} else {
						groupIndex ++;
						groupRecordIndex ++;
						continue;
					}
				} else {
					double weightSum  = itemPopuSumWeightsInGroupOrder[groupId] + 0.0;
					if (weightSum != 0) {
						int gSumIndex  = groupId * FEATURE_NUM;
						for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
							double value = itemGroupSum[gSumIndex++] / weightSum;
							meanRating += value * userFeature
								[userStartIndex + featureId];
						}
					} else {
						throw new RuntimeException("Inner error: never reaches.");
						// continue;
					}
				}
				meanRatings[userId][groupId] = meanRating;
			}
			userStartIndex += FEATURE_NUM;
			if (groupIndex != groupNum)
				throw new RuntimeException(String.format(
					"groupIndex[%s] != groupNum[%s]",
					groupIndex, groupNum));
		}
		if (userStartIndex != FEATURE_NUM * USER_NUM)
			throw new RuntimeException(String.format(
				"userStartIndex[%s] != FEATURE_NUM[%s]",
				userStartIndex, FEATURE_NUM * USER_NUM));
		if (groupRecordIndex != ITEM_GROUP_RECORD_NUM)
			throw new RuntimeException(String.format(
				"groupRecordIndex[%s] != ITEM_GROUP_RECORD_NUM[%s]",
				groupRecordIndex, ITEM_GROUP_RECORD_NUM));
		return meanRatings;
	}
	
	private double[][] getMeanFullRatingInUserGroupOrder() {
		int userStartIndex     = 0;
		double[][] meanRatings = new double[USER_NUM][ITEM_GROUP_NUM];
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int groupFeaIndex = 0;
			for (int groupId = 0; groupId < ITEM_GROUP_NUM; groupId ++) {
				double meanRating = 0; // maybe set with a large value
				double weightSum  = itemPopuSumWeightsInGroupOrder[groupId];
				if (weightSum != 0) {
					for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
						meanRating += itemGroupSum[groupFeaIndex++] *
								userFeature [userStartIndex + featureId];
					}
				} else {
					groupFeaIndex += FEATURE_NUM;
				}
				meanRatings[userId][groupId] = meanRating / weightSum;
			}
			userStartIndex += FEATURE_NUM;
			if (groupFeaIndex != itemGroupSum.length)
				throw new RuntimeException(String.format(
					"groupIndex[%s] != groupNum[%s]",
						groupFeaIndex, itemGroupSum.length));
		}
		if (userStartIndex != FEATURE_NUM * USER_NUM)
			throw new RuntimeException(String.format(
				"userStartIndex[%s] != FEATURE_NUM[%s]",
					userStartIndex, FEATURE_NUM * USER_NUM));
		return meanRatings;
	}

	private double getPosGradRatingPerUser(int userStartIndex,
			int recordStartIndex, int itemNum, double[] V, Function func) {
		double lloss = 0;
		for (int i = 0; i < itemNum; i ++) {
			int itemId    = itemIndexInUserOrder[recordStartIndex++];
			double rating = 0;
			int itemStartIndex = itemId * FEATURE_NUM;
			for (int featureId= 0; featureId < FEATURE_NUM; featureId ++){
				rating += userFeature[userStartIndex + featureId] *
						itemFeature[itemStartIndex + featureId];
			}
			rating = func.getFunction(positiveRating - rating);
			lloss += rating;
			if (V != null) {
				for (int featureId = 0; featureId < FEATURE_NUM; featureId ++){
					V[featureId] -= rating * itemFeature[itemStartIndex + featureId];
				}
			}
		}
		return lloss;
	}

	private double getNegGradRatingPerUser(int userId, int userStartIndex,
				int userPrefMarginStartIndex, int groupRecordStartIndex,
					int groupNum, double[] V, Function func) {
		double lloss   = 0;
		int groupIndex = 0;
		for (int groupId = 0; groupId < ITEM_GROUP_NUM; groupId ++) {
			double rih     = 0;
			double[] tempV = null;
			if (V != null) tempV = new double[FEATURE_NUM];
			if (groupIndex < groupNum && groupId ==
				itemGroupIndexInUserOrder[groupRecordStartIndex]) {
				if (observedMeanNegVStatus[groupRecordStartIndex]) {
					int groupFeatureIndex = FEATURE_NUM * groupRecordStartIndex;
					groupIndex ++;
					groupRecordStartIndex ++;
					for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
						double value = observedMeanNegV[groupFeatureIndex++];
						rih += value * userFeature[userStartIndex + featureId];
						if (V != null) tempV[featureId] = value;
					}
				} else {
					groupIndex ++;
					groupRecordStartIndex ++;
					continue;
				}
			} else {
				double weightSum = itemPopuSumWeightsInGroupOrder[groupId] + 0.0;
				if (weightSum != 0) {
					int gSumIndex = groupId * FEATURE_NUM;
					for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
						double value= itemGroupSum[gSumIndex++]/weightSum;
						rih += value * userFeature[userStartIndex + featureId];
						if (V != null) tempV[featureId] = value;
					}
				} else {
					throw new RuntimeException("Inner error: never reaches.");
					//continue;
				}
			}
			rih    = func.getFunction(rih - negativeRating -
						userPrefMargin[userPrefMarginStartIndex + groupId]);
			lloss += rih;
			if (V != null) {
				for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
					V[featureId] += rih * tempV[featureId];
				}
			}
		}
		if (groupIndex != groupNum) {
			throw new RuntimeException(String.format(
				"NegGradient::groupIndex=%s,groupNum=%s", groupIndex, groupNum));
		}
		
		////////////////// Enhance groups /////////////////////////
		int[] enhItemIDs = enhUserItemIDs[userId];
		for (int i = 0; i < enhItemIDs.length; i ++) {
			int itemId         = enhItemIDs[i];
			int itemStartIndex = itemId * FEATURE_NUM;
			double rij         = 0;
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				rij += userFeature[userStartIndex + featureId] *
					itemFeature[itemStartIndex + featureId];
			}
			int groupId = itemMappedGroupIds[itemId];
			rij         = func.getFunction(rij - negativeRating -
							userPrefMargin[userPrefMarginStartIndex + groupId]);
			lloss += rij;
			if (V != null) {
				for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
					V[featureId] += rij * itemFeature[itemStartIndex + featureId];
				}
			}
		}
		///////////////////////////////////////////////
		
		return lloss;
	}

	private void getNegGradPerGroupUser(double[] UPerUser, double[] USum,
						Function func) {
		int UIndex                   = 0;
		int userStartIndex           = 0;
		int groupRecordIndex         = 0;
		int userPrefMarginStartIndex = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int groupNum     = itemGroupNumInUserOrder[userId];
			int groupIndex   = 0;
			int featureIndex = 0;
			for (int groupId = 0; groupId < ITEM_GROUP_NUM; groupId ++) {
				double weightSum = itemPopuSumWeightsInGroupOrder[groupId] + 0.0;
				double rih       = 0;
				boolean record   = false;
				if (groupIndex < groupNum && groupId ==
					itemGroupIndexInUserOrder[groupRecordIndex]) {
					int groupFeatureIndex = FEATURE_NUM * groupRecordIndex;
					if (observedMeanNegVStatus[groupRecordIndex]) {
						for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
							double value = observedMeanNegV[groupFeatureIndex++];
							rih += value * userFeature[userStartIndex + featureId];
						}
						weightSum -= itemPopuSumWeightsPerGroupInUserOrder
											[groupRecordIndex];
						groupIndex ++;
						groupRecordIndex ++;
						record = true;
					} else {
						groupIndex ++;
						groupRecordIndex ++;
						UIndex       += FEATURE_NUM;
						featureIndex += FEATURE_NUM;
						continue;
					}
				} else {
					if (weightSum != 0) {
						int gSumIndex = groupId * FEATURE_NUM;
						for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
							double value = itemGroupSum[gSumIndex++]/weightSum;
							rih += value * userFeature[userStartIndex + featureId];
						}
					}
				}
				if (weightSum != 0) {
					rih = func.getFunction(rih - negativeRating -
							userPrefMargin[userPrefMarginStartIndex + groupId]) /
								weightSum;
					for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
						double value = rih * userFeature[userStartIndex + featureId];
						if (record)UPerUser[UIndex + featureId]= value;
						USum[featureIndex+ featureId] += value;
					}
				}
				if (record) UIndex+=FEATURE_NUM;
				featureIndex += FEATURE_NUM;
			} // end for (groupId )
			userStartIndex           += FEATURE_NUM;
			userPrefMarginStartIndex += ITEM_GROUP_NUM;
			if (groupIndex != groupNum) {
				throw new RuntimeException(String.format(
						"groupIndex[%s] != groupNum[%s]", groupIndex, groupNum));
			}
			if (featureIndex != USum.length) {
				throw new RuntimeException(String.format(
						"featureIndex[%s] != USum.length[%s]",
							featureIndex, USum.length));
			}
		} // end for (userId)
		if (UIndex != ITEM_GROUP_RECORD_NUM * FEATURE_NUM)
			throw new RuntimeException(String.format(
				"UIndex[%s] != ITEM_GROUP_NUM[%s] * FEATURE_NUM[%s]",
					UIndex, ITEM_GROUP_NUM, FEATURE_NUM));
		if (userStartIndex != USER_NUM * FEATURE_NUM)
			throw new RuntimeException(String.format(
				"userStartIndex[%s] != USER_NUM[%s] * FEATURE_NUM[%s]",
				userStartIndex, USER_NUM, FEATURE_NUM));
		if (groupRecordIndex != ITEM_GROUP_RECORD_NUM)
			throw new RuntimeException(String.format(
				"groupRecordIndex[%s] != ITEM_GROUP_RECORD_NUM[%s]",
				groupRecordIndex, ITEM_GROUP_RECORD_NUM));
		if (userPrefMarginStartIndex != USER_NUM * ITEM_GROUP_NUM)
			throw new RuntimeException(String.format(
				"userPrefMarginStartIndex[%s] != USER_NUM[%s] * ITEM_GROUP_NUM[%s]",
				userPrefMarginStartIndex, USER_NUM, ITEM_GROUP_NUM));
	}

	private double[] initUserPrefMargin(int feaNum, File inputFile) {
		double[] userPrefMargin = null;
		if (inputFile == null) {
			userPrefMargin = new double[feaNum];
			for (int i = userPrefMargin.length-1; i >= 0; i --) {
				userPrefMargin[i] = posNegRatingDiff + Math.random() * 0.1;
			}
		} else {
			System.out.println("[Info] Loading userPrefMargin from " +
					inputFile.getAbsolutePath());
			try {
				userPrefMargin = Utils.convert2DTo1D(Utils.load2DoubleArray(
						inputFile, CheckinConstants.DELIMITER));
			} catch (CheckinException e) {
				e.printStackTrace();
				throw new RuntimeException(e.toString());
			}
			if (userPrefMargin.length != USER_NUM * ITEM_GROUP_NUM) {
				throw new RuntimeException(String.format(
					"Failed to load feature: dimension = %s which should be %s.",
					userPrefMargin.length, ITEM_NUM * FEATURE_NUM));
			}
		}
		return userPrefMargin;
	}

	private void initEnhGroups() throws CheckinException {
		double[][] userMeanNegRatings = getMeanFullRatingInUserGroupOrder();// getMeanNegRatingInUserGroupOrder();
		
		enhUserItemIDs  = new int[USER_NUM][];
		int recordIndex = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int itemNum = itemNumInUserOrder[userId];
			Set<Integer> itemSet = new HashSet<Integer>();
			for (int i = 0; i < itemNum; i ++) {
				itemSet.add(itemIndexInUserOrder[recordIndex++]);
			}
			
			int sampledItemNum = ITEM_NUM - itemNum;
			if (enhGroupNum < sampledItemNum) sampledItemNum = enhGroupNum;
			int[] array = new int[sampledItemNum];
			for (int i = 0; i < sampledItemNum; i++) {
				int itemId = -1;
				while (true) {
					itemId = (int)(Math.random() * ITEM_NUM);
					if (! itemSet.contains(itemId)) {
						itemSet.add(itemId);
						if (predict(userId, itemId) > userMeanNegRatings
								[userId][itemMappedGroupIds[itemId]]) {
							break;
						}
					}
				}
				array[i] = itemId;
			}
			
			enhUserItemIDs[userId] = array;
		}
		if (recordIndex != RATING_RECORD_NUM) {
			throw new RuntimeException(String.format(
				"recordIndex[%s] != RATING_RECORD_NUM[%s]",
					recordIndex, RATING_RECORD_NUM));
		}
		
		Map<Integer, Set<Integer>> enhItemUserMap = new HashMap<Integer, Set<Integer>>();
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int[] enhItems = enhUserItemIDs[userId];
			for (int itemId: enhItems) {
				Set<Integer> userGroupSet = enhItemUserMap.get(itemId);
				if (userGroupSet == null) {
					userGroupSet = new HashSet<Integer>();
					enhItemUserMap.put(itemId, userGroupSet);
				}
				if (userGroupSet.contains(userId)) throw new
					RuntimeException("userId has existed:" + userId);
				userGroupSet.add(userId);
			}
		}
		
		enhItemUserIDs = new int[ITEM_NUM][];
		for (int itemID : enhItemUserMap.keySet()) {
			Set<Integer> userSet    = enhItemUserMap.get(itemID);
			enhItemUserIDs[itemID]  = Utils.getSortedKeys(userSet);
		}
	}

	protected void initMappedUIndexInItemOrder() {
		@SuppressWarnings("unchecked")
		Map<Integer, Integer>[] userItemMap = new HashMap[USER_NUM];
		int groupIndex  = 0;
		int recordIndex = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int groupNum = itemGroupNumInUserOrder[userId];
			Map<Integer, Integer> map = new HashMap<Integer, Integer>();
			for (int i = 0; i < groupNum; i ++) {
				//int groupId  = itemGroupIndexInUserOrder[groupIndex];
				int itemNum  = itemNumPerGroupInUserOrder[groupIndex];
				int groupFeatureStartIndex = groupIndex *
						FEATURE_NUM;
				groupIndex ++;
				for (int jj = 0; jj < itemNum; jj ++) {
					int itemId = itemIndexInUserOrder[recordIndex ++];
					if(map.containsKey(itemId))
						throw new RuntimeException(String.format(
						"initMappedUIndexInItemOrder::userId=%s, itemId=%s, exisited",
						userId, itemId));
					map.put(itemId, groupFeatureStartIndex);
				}
			}
			userItemMap[userId] = map;
		}
		if (groupIndex != ITEM_GROUP_RECORD_NUM) 
			throw new RuntimeException(String.format(
				"groupIndex[%s] != ITEM_GROUP_RECORD_NUM[%s]",
				groupIndex, ITEM_GROUP_RECORD_NUM));
		if (recordIndex != RATING_RECORD_NUM)
			throw new RuntimeException(String.format(
				"recordIndex[%s] != RATING_RECORD_NUM[%s]",
				recordIndex, RATING_RECORD_NUM));
		mappedUIndexInItemOrder = new int[RATING_RECORD_NUM];
		recordIndex = 0;
		for (int itemId = 0; itemId < ITEM_NUM; itemId ++) {
			int userNum = userNumInItemOrder[itemId];
			for (int i = 0; i < userNum; i ++) {
				int userId=userIndexInItemOrder[recordIndex];
				Map<Integer, Integer> map = userItemMap[userId];
				int mappedIndex = map.get(itemId);
				mappedUIndexInItemOrder[recordIndex]=mappedIndex;
				recordIndex ++;
			}
		}
		if (recordIndex != RATING_RECORD_NUM)
			throw new RuntimeException(String.format(
				"recordIndex[%s] != RATING_RECORD_NUM[%s]",
				recordIndex, RATING_RECORD_NUM));
	}
	
	protected void loadItemGroupFile(File itemGroupFile)
						throws CheckinException {
		Utils.exists(itemGroupFile);
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(itemGroupFile)));
			String line          = null;
			Set<Integer>groupSet = new HashSet<Integer>();
			int maxGroupID       = -1;
			int minGroupID       = Integer.MAX_VALUE;
			itemMappedGroupIds   = new int[ITEM_NUM];
			for (int i = 0; i < ITEM_NUM; i ++)
				itemMappedGroupIds[i] = -1;
			while ((line = reader.readLine()) != null) {
				String[] array = Utils.parseLine(line, 2);
				int itemID     = Integer.parseInt(array[0]);
				int groupID    = Integer.parseInt(array[1]);
				if (groupID < 0 || itemID < 0 ||
					itemID >= ITEM_NUM ||
							itemMappedGroupIds[itemID] != -1) {
					reader.close(); reader=null;
					throw new CheckinException("ItemID=%s,GroupID=%s,ITEM_NUM=%s,itemGroupIndices[itemID]=%s",
						itemID, groupID, ITEM_NUM,
						itemMappedGroupIds[itemID]);
				}
				itemMappedGroupIds[itemID] = groupID;
				if (! groupSet.contains(groupID)) 
					groupSet.add(groupID);
				if (groupID > maxGroupID) maxGroupID = groupID;
				if (groupID < minGroupID) minGroupID = groupID;
			}
			reader.close();
			if (minGroupID != 0 && maxGroupID != groupSet.size() - 1) {
				throw new CheckinException("MinGroupID=%s,MaxGroupID=%s,GroupNum=%s",
					minGroupID, maxGroupID, groupSet.size());
			}
			ITEM_GROUP_NUM      = groupSet.size();
			itemNumInGroupOrder = new int[ITEM_GROUP_NUM];
			for (int i = 0; i < ITEM_NUM; i ++) {
				if (itemMappedGroupIds[i] < minGroupID ||
					itemMappedGroupIds[i] > maxGroupID)
					throw new CheckinException("Cannot find group for item %s .", i);
				itemNumInGroupOrder[itemMappedGroupIds[i]] ++;
			}
		} catch (IOException e) {
			e.printStackTrace();
			throw new CheckinException(e);
		} finally {
			Utils.cleanup(reader);
		}
	}


	@Override
	protected void storeLatentVariables() {
		// store item feature
		Utils.store2DArray(itemFeature, FEATURE_NUM, new File(outputPath, CheckinConstants.STRING_ITEM_FEATURE));
		// store user feature
		Utils.store2DArray(userFeature, FEATURE_NUM, new File(outputPath, CheckinConstants.STRING_USER_FEATURE));
				
		// store user pref margin
		Utils.store2DArray(userPrefMargin, ITEM_GROUP_NUM,
			new File(outputPath, FILE_NAME_USER_PREF_MARGIN));
	}

	@Override
	protected void parseTrainRatingFile(File trainRatingFile) {
		//  donothing, just want to cancel this
	}
	
	/*
	 * The line format is as follows:
	 * 	UserId '\t' ItemId '\t' RawRating
	 * Before calling this method, must call parseTrainClusterFile()
	 */
	protected void parseTrainRatingFile_RRMF(File trainRatingFile) throws CheckinException {
		BufferedReader reader = null;
		try {
			System.out.println(String.format("[Info] FORCE_CHECK = %s", FORCE_CHECK));
			// load rating in item order
			Map<Integer, Map<Integer, Float>> itemUserRatingMap =
					new HashMap<Integer, Map<Integer, Float>>();
			String line = null;
			reader      = new BufferedReader(new InputStreamReader(
					new FileInputStream(trainRatingFile)));
			int minItemId = Integer.MAX_VALUE;
			int maxItemId = -1;
			RATING_RECORD_NUM = 0;
			while ((line = reader.readLine()) != null) {
				if ("".equals(line = line.trim())) continue;
				String[] array = line.split("\t");
				if (array.length < 3) {
					reader.close();
					reader = null;
					throw new CheckinException("Failed to parse TrainRatingFile: %s", line);
				}
				int userId   = Integer.parseInt(array[0]);
				int itemId   = Integer.parseInt(array[1]);
				float rating = Float.parseFloat(array[2]);
				Map<Integer, Float> userRatingMap =
					itemUserRatingMap.get(itemId);
				if (userRatingMap == null) {
					userRatingMap = new HashMap<Integer, Float>();
					userRatingMap.put(userId, rating);
					itemUserRatingMap.put(itemId,
							userRatingMap);
				} else {
					if (userRatingMap.containsKey(userId)) {
						reader.close();
						reader = null;
						throw new CheckinException("User %s has existed. Line : %s",
							userId, line);
					}
					userRatingMap.put(userId, rating);
				}
				if (itemId < minItemId) minItemId = itemId;
				if (itemId > maxItemId) maxItemId = itemId;
				RATING_RECORD_NUM ++;
			}
			reader.close();
			if (itemUserRatingMap.size() != ITEM_NUM) {
				String errMsg = String.format("Failed to parse TrainRatingFile: item number does not match with predefined number." +
						"[PredefinedItemNum=%s][ParsedItemNum=%s]",
						ITEM_NUM, itemUserRatingMap.size());
				if (FORCE_CHECK) {
					throw new CheckinException(errMsg);
				} else {
					System.err.println(errMsg);
				}
			}
			if (minItemId != 0) {
				String errMsg = String.format("MinItemId=%s, instead of 0.",
						minItemId);
				if (FORCE_CHECK) {
					throw new CheckinException(errMsg);
				} else {
					System.err.println(errMsg);
				}
			}
			if (maxItemId != ITEM_NUM - 1) {
				String errMsg = String.format("MaxItemId=%s, instead of %s.",
						maxItemId, ITEM_NUM - 1);
				if (FORCE_CHECK) {
					throw new CheckinException(errMsg);
				} else {
					System.err.println(errMsg);
				}
			}
			{
				ratingsInItemOrder   = new float[RATING_RECORD_NUM];
				userIndexInItemOrder = new int[RATING_RECORD_NUM];
				userNumInItemOrder   = new int[ITEM_NUM];
				int[] itemIds = Utils.getSortedKeys(
						itemUserRatingMap.keySet());
				int index     = 0;
				for (int itemId : itemIds) {
					Map<Integer, Float> userRatingMap =
						itemUserRatingMap.get(itemId);
					int[] userIds = Utils.getSortedKeys(
						userRatingMap.keySet());
					for (int userId : userIds) {
						ratingsInItemOrder[index] =
							userRatingMap.get(userId);
						userIndexInItemOrder[index] =
							userId;
						index ++;
					}
					userNumInItemOrder[itemId] =
						userRatingMap.size();
				}
			}
			// clear
			itemUserRatingMap = null;
	
			// load rating in User order
			reader = new BufferedReader(new InputStreamReader(
						new FileInputStream(trainRatingFile)));
			Map<Integer, Map<Integer, Float>> userItemRatingMap =
				new HashMap<Integer, Map<Integer, Float>>();
			int minUserId = Integer.MAX_VALUE;
			int maxUserId = -1;
			while ((line = reader.readLine()) != null) {
				if ("".equals(line = line.trim())) continue;
				String[] array = line.split("\t");
				if (array.length < 3) {
					reader.close();
					reader = null;
					throw new CheckinException("Failed to parse TrainRatingFile: %s", line);
				}
				int userId    = Integer.parseInt(array[0]);
				int itemId    = Integer.parseInt(array[1]);
				float rating  = Float.parseFloat(array[2]);
				Map<Integer, Float> itemRatingMap =
					userItemRatingMap.get(userId);
				if (itemRatingMap == null) {
					itemRatingMap = new HashMap<Integer, Float>();
					itemRatingMap.put(itemId, rating);
					userItemRatingMap.put(userId, itemRatingMap);
				} else {
					if (itemRatingMap.containsKey(itemId)) {
						throw new CheckinException("Item %s has existed. Line:%s",
							itemId, line);
					}
					itemRatingMap.put(itemId, rating);
				}
				if (userId < minUserId)
					minUserId = userId;
				if (userId > maxUserId)
					maxUserId = userId;
			} // end
			reader.close();
			if (userItemRatingMap.size() != USER_NUM) {
				String errMsg = String.format("Failed to parse TrainRatingFile : User number does not match with the predefined number." +
						"[PredefinedUserNum=%s][LoadedUserNum=%s]",
						USER_NUM, userItemRatingMap.size());
				if (FORCE_CHECK) {
					throw new CheckinException(errMsg);
				} else {
					System.err.println(errMsg);
				}
			}
			if (minUserId != 0) {
				String errMsg = String.format("MinUserNum=%s, instead of 0.",
						minUserId);
				if (FORCE_CHECK) {
					throw new CheckinException(errMsg);
				} else {
					System.err.println(errMsg);
				}
			}
			if (maxUserId != USER_NUM - 1) {
				String errMsg = String.format("MaxUserNum=%s, instead of %s.",
						maxUserId, USER_NUM - 1);
				if (FORCE_CHECK) {
					throw new CheckinException(errMsg);
				} else {
					System.err.println(errMsg);
				}
			}
			{
				final int[] userIds     = Utils.getSortedKeys(
						userItemRatingMap.keySet());
				ITEM_GROUP_RECORD_NUM   = 0;
				itemGroupNumInUserOrder = new int[USER_NUM];
				for (int userId : userIds) {
					Map<Integer, Float> itemRatingMap =
						userItemRatingMap.get(userId);
					Set<Integer> groupSet = new HashSet<Integer>();
					for (int itemId : itemRatingMap.keySet()) {
						int groupID = itemMappedGroupIds[itemId];
						if (! groupSet.contains(groupID))
							groupSet.add(groupID);
					}
					ITEM_GROUP_RECORD_NUM          += groupSet.size();
					itemGroupNumInUserOrder[userId] = groupSet.size();
				}
				ratingsInUserOrder   = new float[RATING_RECORD_NUM];
				itemIndexInUserOrder = new int[RATING_RECORD_NUM]; 
				itemNumInUserOrder   = new int[USER_NUM];
				itemGroupIndexInUserOrder      = new int[ITEM_GROUP_RECORD_NUM];
				itemNumPerGroupInUserOrder     = new int[ITEM_GROUP_RECORD_NUM];
				userGroupStartIndexInUserOrder = new int[USER_NUM];
				int index          = 0;
				int itemGroupIndex = 0;
				for (int userId : userIds) {
					Map<Integer, Float> itemRatingMap      =
						userItemRatingMap.get(userId);
					Map<Integer, Set<Integer>> groupItemMap=
						new HashMap<Integer, Set<Integer>>();
					for (int itemId : itemRatingMap.keySet()) {
						int groupId = itemMappedGroupIds[itemId];
						Set<Integer> itemSet = groupItemMap.get(groupId);
						if (itemSet == null) {
							itemSet = new HashSet<Integer>();
							groupItemMap.put(groupId, itemSet);
						}
						itemSet.add(itemId);
					}
					if (groupItemMap.size() != itemGroupNumInUserOrder[userId]) {
						throw new CheckinException("InnerError:: userID=%s,formmerGroupNum=%s,latterGroupNum=%s",
							userId, itemGroupNumInUserOrder[userId], groupItemMap.size());
					}
					int[] sortedGroupIds = Utils.getSortedKeys(
						groupItemMap.keySet());
					userGroupStartIndexInUserOrder[userId]=
							itemGroupIndex;
					for (int groupId : sortedGroupIds) {
						Set<Integer> itemSet = groupItemMap.get(groupId);
						int[] sortedItemIds  = Utils.getSortedKeys(itemSet);
						itemGroupIndexInUserOrder[itemGroupIndex]  = groupId;
						itemNumPerGroupInUserOrder[itemGroupIndex] = sortedItemIds.length;
						itemGroupIndex ++;
						for (int i = 0; i < sortedItemIds.length; i ++) {
							int itemId = sortedItemIds[i];
							ratingsInUserOrder[index] =
								itemRatingMap.get(itemId);
							itemIndexInUserOrder[index] =
								itemId;
							index ++;
						}
					}
					itemNumInUserOrder[userId] =
						itemRatingMap.size();
				} // end for (int userId : userIds)
				if (index !=  RATING_RECORD_NUM) {
					throw new CheckinException(
						"ParseTrain::index[%s]!=RATING_RECORD_NUM[%s]",
						index, RATING_RECORD_NUM);
				}
				if (itemGroupIndex != ITEM_GROUP_RECORD_NUM) {
					throw new CheckinException(
						"ParseTrain::itemGroupIndex[%s]!=ITEM_GROUP_NUM[%s]",
						itemGroupIndex, ITEM_GROUP_NUM);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
			throw new CheckinException(e);
		} finally {
			Utils.cleanup(reader);
		}
	}


	protected void prepareItemPopuWeights() {
		int recordIndex = 0;
		int groupIndex  = 0;
		itemPopuSumWeightsInGroupOrder        = new double[ITEM_GROUP_NUM];
		itemPopuSumWeightsPerGroupInUserOrder = new double[ITEM_GROUP_RECORD_NUM];
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int groupNum = itemGroupNumInUserOrder[userId];
			for (int i = 0; i < groupNum; i ++) {
				int itemNum = itemNumPerGroupInUserOrder[groupIndex];
				double sum  = 0;
				for (int jj = 0; jj < itemNum; jj ++) {
					@SuppressWarnings("unused")
					int itemId = itemIndexInUserOrder[recordIndex ++];
					sum ++;
				}
				itemPopuSumWeightsPerGroupInUserOrder[groupIndex++] = sum;
			}
		}
		for (int itemId = 0; itemId < ITEM_NUM; itemId ++) {
			int groupId = itemMappedGroupIds[itemId];
			itemPopuSumWeightsInGroupOrder[groupId] ++;
		}
		//Utils.print(itemPopuSumWeightsInGroupOrder, 1, ITEM_GROUP_NUM, ",");
		//Utils.print(itemPopuSumWeightsPerGroupInUserOrder, 1, itemPopuSumWeightsPerGroupInUserOrder.length, ",");
		if (recordIndex != RATING_RECORD_NUM)
			throw new RuntimeException(String.format(
				"RecordIndex[%s] != RATING_RECORD_NUM[%s]",
				recordIndex, RATING_RECORD_NUM));
		if (groupIndex !=  ITEM_GROUP_RECORD_NUM)
			throw new RuntimeException(String.format(
				"GroupIndex[%s] != ITEM_GROUP_RECORD_NUM",
				groupIndex, ITEM_GROUP_NUM));
	}
	private static void createItemGroup(File inputFile, int itemNum,
						int groupNum, File outputFile) throws CheckinException {
		BufferedWriter writer = null;
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new InputStreamReader(
							new FileInputStream(inputFile)));
			String line = null;
			Map<Integer, Set<Integer>> itemUserMap =
					new HashMap<Integer, Set<Integer>>();
			int maxItemID = -1;
			while ((line = reader.readLine()) != null) {
				String[] array = Utils.parseLine(line, 3);
				int userID     = Integer.parseInt(array[0]);
				int itemID     = Integer.parseInt(array[1]);
				Set<Integer> userSet = itemUserMap.get(itemID);
				if (userSet == null) {
					userSet = new HashSet<Integer>();
					itemUserMap.put(itemID, userSet);
				}
				userSet.add(userID);
				if (itemID > maxItemID) maxItemID = itemID;
			}
			reader.close();
			final Integer[] rank  = new Integer[itemNum];
			final int[] viewerNum = new int[itemNum];
			double viewerNumSum   = 0;
			for (int i = 0; i < itemNum; i ++) {
				rank[i]      = i;
				viewerNum[i] = itemUserMap.containsKey(i) ?
					itemUserMap.get(i).size() : 0;
				//if (viewerNum[i] == 0) System.out.println(i);
				viewerNumSum += viewerNum[i];
			}
			Arrays.sort(rank, new Comparator<Integer>() {
				@Override
				public int compare(Integer item1, Integer item2){
					int num1 = viewerNum[item1];
					int num2 = viewerNum[item2];
					if (num1 > num2) return -1;
					else if (num1 < num2) return 1;
					else return 0;
				}
			});
			
			int itemNumPerGroup = Math.round(itemNum / groupNum);
			System.out.println("itemNumPerGroup=" + itemNumPerGroup);
			int[] groupIDs = new int[itemNum];
			for (int itemId = 0; itemId < itemNum; itemId ++)
				groupIDs[itemId] = -1;
			int groupIndex = 0;
			int[] itemNumsPerGroup = new int[groupNum];
			int randomRank[] = new int[itemNum];
			for (int i = 0; i < itemNum; i ++)
				randomRank[i] = i;
			Utils.shuffle(randomRank);
			
			for (int order = 0; order < itemNum; order ++) {
				//int itemId = rank[order];
				int itemId = randomRank[order];
				groupIDs[itemId] = groupIndex;
				itemNumsPerGroup[groupIndex] ++;
				groupIndex ++;
				if (groupIndex == groupNum) {
					groupIndex = 0;
				}
			}
			int minItemNumPerGroup = Integer.MAX_VALUE;
			int maxItemNumPerGroup = -1;
			for (int groupId = 0; groupId < groupNum; groupId ++) {
				int num = itemNumsPerGroup[groupId];
				if (num < minItemNumPerGroup)
					minItemNumPerGroup = num;
				if (num > maxItemNumPerGroup)
					maxItemNumPerGroup = num;
			}
			System.out.println("MinItemNumPerGroup=" + minItemNumPerGroup);
			System.out.println("MaxItemNumPerGroup=" + maxItemNumPerGroup);
	
			if (! outputFile.getParentFile().exists())
				outputFile.getParentFile().mkdirs();
			writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(outputFile)));
			for (int itemId = 0; itemId < itemNum; itemId ++) {
				int groupId = groupIDs[itemId];
				if (groupId < 0 || groupId >= groupNum) {
					writer.close(); writer = null;
					throw new CheckinException("itemNumPerGroup=%s, itemId=%s, groupID=%s",
						itemNumPerGroup, itemId, groupId);
				}
				writer.write(String.format("%s\t%s", itemId, groupId));
				writer.newLine();
				//System.out.println(String.format("%s\t%s\t%s", itemId, groupId, itemRanks[itemId]));
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(writer);
			Utils.cleanup(reader);
		}
	}

	protected class SmoothObjFunction implements Function {
		// x is the u_i^T x v_h - x_i^T x v_j + r_ih
		public double getFunction(double x) {
			// smooth
			/*if (x >= 0) {
				return x + Math.log(1 + Math.exp(-1 *
						smoothAlpha * x)) / smoothAlpha; 
			} else {
				return Math.log(1 + Math.exp(smoothAlpha * x)) /
						smoothAlpha;
			}*/
			//return  x + Math.log(1 + Math.exp(-1 * smoothAlpha * x)) / smoothAlpha; 
			
			// max{0, x}
			return Math.max(0, x);
			
			// mle log (1 + e(-x))
			//return Math.log(1 + Math.exp(x));
		}
		
	}

	private class SmoothGradFunction implements Function {
		// x is the u_i^T x v_h - x_i^T x v_j + r_ih
		public double getFunction(double x) {
			// smooth
			/*if (x >= 0) {
				return 1 / (1 + Math.exp(-1 * smoothAlpha * x));
			} else {
				return 1 - 1.0 / (1.0 + Math.exp(smoothAlpha * x));
			}*/
			
			return 1 - 1 / (1 + Math.exp(smoothAlpha * x));
		
			// max{0, x}
			//if (x > 0) return 1; else return 0;
			
			// mle log (1 + e(-x))
			//return Math.exp(x ) / (1 + Math.exp(x));
		}
	}

	protected interface Function {
		public double getFunction(double x);
	}
	
	private void checkGradient() throws CheckinException {
		lambdaUser          = 0.03f;
		lambdaItem          = 0.02f;
	
		learningRate          = 1;
		double loopNumPerIter = 100;
		double changeRateEta  = 1e-4;
		for (int iter = 0; iter < loopNumPerIter; iter ++) {
	
			// sample userId
			int userId = (int) (Math.random() * USER_NUM);
			int itemId = (int) (Math.random() * ITEM_NUM);
			
			System.out.println(String.format("userId:%s, itemId:%s", userId, itemId));
			
			double[] origUserFeature    = Utils.copy(userFeature);
			double[] origItemFeature    = Utils.copy(itemFeature);
			double[] origUserPrefMargin = Utils.copy(userPrefMargin);
			
			double[] gradUserFeature    = new double[FEATURE_NUM];
			double[] gradItemFeature    = new double[FEATURE_NUM];
			double[] gradUserPrefMargin = new double[FEATURE_NUM];
			
			updateUserFeature();
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				int index = userId * FEATURE_NUM + featureId;
				gradUserFeature[featureId] = origUserFeature[index] - userFeature[index];
			}
			userFeature = Utils.copy(origUserFeature);
			
			updateUserPrefMargin();
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				int index = userId * FEATURE_NUM + featureId;
				gradUserPrefMargin[featureId] = origUserPrefMargin[index] - userPrefMargin[index];
			}
			userPrefMargin = Utils.copy(origUserPrefMargin);
			
			updateItemFeature();
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				int index = itemId * FEATURE_NUM + featureId;
				gradItemFeature[featureId] = origItemFeature[index] - itemFeature[index];
			}
			itemFeature = Utils.copy(origItemFeature);
			
			// User Feature
			for (int featureId = 0; featureId < FEATURE_NUM ; featureId ++) {
				int ufIndex = userId * FEATURE_NUM + featureId;
				
				userFeature           = Utils.copy(origUserFeature);
				userFeature[ufIndex] += changeRateEta;
				double loss_1 = calLoss();
				
				userFeature           = Utils.copy(origUserFeature);
				userFeature[ufIndex] -= changeRateEta;
				double loss_2 = calLoss();
				
				double approximatedGrad = (loss_1 - loss_2) / changeRateEta / 2;
				
				double diff = approximatedGrad - gradUserFeature[featureId];
				String str  = String.format("UserFeature: calGrad:%s, approximatedGrad:%s, absDiff:%s, relDiff:%s",
						gradUserFeature[featureId], approximatedGrad,
						Math.abs(diff), Math.abs(diff / approximatedGrad ));
				if (diff > 1.0e-4) {
					System.err.println(str);
					System.exit(-1);
				} else {
					System.out.println(str);
				}
			}
			userFeature = Utils.copy(origUserFeature);
			
			// UserPrefMargin
			for (int featureId = 0; featureId < FEATURE_NUM ; featureId ++) {
				int ufIndex = userId * FEATURE_NUM + featureId;
				
				userPrefMargin           = Utils.copy(origUserPrefMargin);
				userPrefMargin[ufIndex] += changeRateEta;
				double loss_1 = calLoss();
				
				userPrefMargin           = Utils.copy(origUserPrefMargin);
				userPrefMargin[ufIndex] -= changeRateEta;
				double loss_2 = calLoss();
				
				double approximatedGrad = (loss_1 - loss_2) / changeRateEta / 2;
				
				double diff = approximatedGrad - gradUserPrefMargin[featureId];
				String str  = String.format("UserPrefMargin: calGrad:%s, approximatedGrad:%s, absDiff:%s, relDiff:%s",
						gradUserPrefMargin[featureId], approximatedGrad,
						Math.abs(diff), Math.abs(diff / approximatedGrad ));
				if (diff > 1.0e-4) {
					System.err.println(str);
				} else {
					System.out.println(str);
				}
			}
			userPrefMargin = Utils.copy(origUserPrefMargin);
			
			// Item Feature
			for (int featureId = 0; featureId < FEATURE_NUM ; featureId ++) {
				int ifIndex = itemId * FEATURE_NUM + featureId;
				
				itemFeature           = Utils.copy(origItemFeature);
				itemFeature[ifIndex] += changeRateEta;
				updateItemGroupSum();
				updateObservedMeanNegV();
				double loss_1 = calLoss();
				
				itemFeature           = Utils.copy(origItemFeature);
				itemFeature[ifIndex] -= changeRateEta;
				updateItemGroupSum();
				updateObservedMeanNegV();
				double loss_2 = calLoss();
				
				double approximatedGrad = (loss_1 - loss_2) / changeRateEta / 2;
				
				double diff = approximatedGrad - gradItemFeature[featureId];
				String str  = String.format("ItemFeature: calGrad:%s, approximatedGrad:%s, absDiff:%s, relDiff:%s",
						gradItemFeature[featureId], approximatedGrad,
						Math.abs(diff), Math.abs(diff / approximatedGrad ));
				if (diff > 1.0e-4) {
					System.err.println(str);
					System.exit(-1);
				} else {
					System.out.println(str);
				}
			}
			itemFeature = Utils.copy(origItemFeature);
			updateItemGroupSum();
			updateObservedMeanNegV();
		
			System.out.println();
		} // end for iteration loop
	}

	public static void main(String[] args) throws CheckinException {
		new RRFM(CheckinConstants.CONFIG_MANAGER).checkGradient();
	}
}


