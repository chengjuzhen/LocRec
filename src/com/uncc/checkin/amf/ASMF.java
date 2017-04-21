package com.uncc.checkin.amf;

import com.uncc.checkin.CheckinConstants;
import com.uncc.checkin.CheckinException;
import com.uncc.checkin.ParamManager;
import com.uncc.checkin.util.Utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

/*
 * Options:
 * ASMF: -CatNum <int> -LocCat <file> -AugRating <file>
 * 		[Optional]
 * 			-PotRating <float>
 *			-LambdaCat <float>
 *			-CatOffset <float>
 *			-AugRatingTopNum <int>
 *			-UserSimRatio <float>
 * 			-HomeLatLng <file>
 * 			-LocLatLng <file> 
 * 			-DFriend <file>
 * 			-LFreind <file>
 * 			-NFriend <file>
 * 
 *			for weight schema:
 * 				-WM	<int> (0: MULT, 1: SQRT, 2: LOG)
 *				-Alpha <float>
 *				-Epsilon <double>
 *				-Delta <float>
 */
public class ASMF extends AMF {
	public static final String FILE_NAME_PAIRWISE_POWERLAW_PARAM =
											"Pairwise_PowerLawParam";
	public  static final String METHOD_NAME                   = "ASMF";
	public  static final String STRING_CATEGORY_NUM           = "CatNum";
	public  static final String FILE_NAME_USER_CATEGORY_PREF  = "UserCategoryPref";
	public  static final String FILE_NAME_USER_CATPREF_OFFSET = "UserCatPrefOffset";
	private static final String STRING_DIRECT_FRIEND_FILE     = "DFriend";
	private static final String STRING_LOCATION_FRIEND_FILE   = "LFreind";
	private static final String STRING_NEIGHBOR_FRIEND_FILE   = "NFriend";
	public  static final String STRING_USER_HOME_LATLNG_FILE  = "HomeLatLng";
	public  static final String STRING_LOCATION_LATLNG_FILE   = "LocLatLng";
	public  static final String STRING_LOCATION_CAT_FILE      = "LocCat";
	private static final String STRING_AUG_TRAING_RATING_FILE = "AugRating";
	public  static final String STRING_LAMBDA_CAT             = "LambdaCat";
	public  static final String STRING_CAT_OFFSET             = "CatOffset";
	
	private double[] itemFeature                = null;
	private double[] userFeature                = null;
	private double[] userCategoryPref           = null;
	private double[] categoryPrefOffsets        = null;
	
	private double[] userTuserSumInCategory     = null;
	private double[] itemTitemSumInCategory     = null;
	private double[] itemTitemSumAll            = null;
	
	private int[] categoryIndexForItem          = null;
	private int[] itemCategoryIndexInUserOrder  = null;
	private int[] itemCategoryNumInUserOrder    = null;
	private int[] categoryNumInUserOrder        = null;

	private int FEATURE_NUM_SQUARE;
	private int CATEGORY_NUM;
	private int CATEGORY_RECORD_NUM;

	private float lambdaItem            = 0.015f; 
	private float lambdaUser            = 0.015f;
	private float lambdaCat             = 500;
	private float lambdaPrefOffset      = 0.015f;
	private float catPrefOffSetConstant = 0.1f;

	private final boolean augmentRating = true;

	public ASMF(ParamManager paramManager) throws CheckinException{
		super();
		
		init(paramManager);
	}

	private void updateUserFeature() {
		int recordIndex = 0;
		int userIndex   = 0;
		int catIndex    = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			double[][] VV = new double[FEATURE_NUM][FEATURE_NUM];
			double[] T    = new double[FEATURE_NUM];
			int catNum    = categoryNumInUserOrder[userId];
			int catPrefStartIndex = userId * CATEGORY_NUM;
			for (int i = 0; i < catNum; i++) {
				int catId      = itemCategoryIndexInUserOrder[catIndex];
				int itemNum    = itemCategoryNumInUserOrder[catIndex];
				double catPref = userCategoryPref[catPrefStartIndex + catId] +
									categoryPrefOffsets[userId];
				catIndex ++;
				for (int j = 0; j < itemNum; j ++) {
					int itemId         = itemIndexInUserOrder[recordIndex];
					int itemStartIndex = itemId * FEATURE_NUM;
					double w_minus_1   = weightsMinusOneInUserOrder[recordIndex];
					double w           = weightsInUserOrder[recordIndex];
					double pref        = prefsInUserOrder[recordIndex];
					for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
						double currentValue = catPref * itemFeature[itemStartIndex + featureId];
						T[featureId]       += pref * w * currentValue;
						for (int f = 0; f < FEATURE_NUM; f ++) {
							VV[featureId][f] += catPref * w_minus_1 *
								currentValue * itemFeature[itemStartIndex + f];
						}
					}
					recordIndex ++;
				}
			}
			int index = 0;
			for (int catId = 0; catId < CATEGORY_NUM; catId ++) {
				double catPref = userCategoryPref[catPrefStartIndex + catId] +
									categoryPrefOffsets[userId];
				for (int i = 0; i < FEATURE_NUM; i ++) {
					for (int j = 0; j < FEATURE_NUM; j ++) {
						VV[i][j] += catPref * catPref *
									itemTitemSumInCategory[index ++];
					}
				}
			}
			for (int i = 0; i < FEATURE_NUM; i ++) {
				VV[i][i] += lambdaUser;
			}
			VV = Utils.inverse(VV);
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				double value = 0;
				for (int i = 0; i < FEATURE_NUM; i ++) {
					value += VV[featureId][i] * T[i];
				}
				userFeature[userIndex ++] = value;
			}
		}
		if (recordIndex != ratingsInUserOrder.length) {
			throw new RuntimeException(String.format(
				"recordIndex[%s] != ratingsInUserOrder.length[%s]",
				recordIndex, ratingsInUserOrder.length));
		}
		if (userIndex != USER_NUM * FEATURE_NUM) {
			throw new RuntimeException(String.format(
				"userIndex[%s] != USER_NUM * FEATURE_NUM[%s]",
				userIndex, USER_NUM * FEATURE_NUM));
		}
		if (catIndex != CATEGORY_RECORD_NUM) {
			throw new RuntimeException(String.format(
				"catIndex[%s] != CATEGORY_RECORD_NUM[%s]",
				catIndex, CATEGORY_RECORD_NUM));
		}
	}

	private void updateItemFeature() {
		int recordIndex = 0;
		int itemIndex   = 0;
		for (int itemId = 0; itemId < ITEM_NUM; itemId ++) {
			double[][] TT = new double[FEATURE_NUM][FEATURE_NUM];
			double[] V    = new double[FEATURE_NUM];
			int userNum   = userNumInItemOrder[itemId];
			int catId     = categoryIndexForItem[itemId];
			for (int i = 0; i < userNum; i++) {
				int userId = userIndexInItemOrder[recordIndex];
				int userStartIndex = userId * FEATURE_NUM;
				double w_minus_1   = weightsMinusOneInItemOrder[recordIndex];
				double w           = weightsInItemOrder[recordIndex];
				double pref        = prefsInItemOrder[recordIndex];
				double catPref     = userCategoryPref[userId * CATEGORY_NUM + catId] +
						categoryPrefOffsets[userId];
				for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
					double currentValue = catPref * userFeature[userStartIndex + featureId];
					V[featureId]       += pref * w * currentValue;
					for (int f = 0; f < FEATURE_NUM; f ++) {
						TT[featureId][f] +=  catPref *
							w_minus_1 * currentValue *
							userFeature[userStartIndex + f];
					}
				}
				recordIndex ++;
			}
			int index = catId * FEATURE_NUM_SQUARE;
			for (int i = 0; i < TT.length; i ++) {
				for (int j = 0; j < TT[0].length; j ++) {
					TT[i][j] += userTuserSumInCategory[index ++];
				}
				TT[i][i] += lambdaItem;
				
			}
			TT = Utils.inverse(TT);
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				double value = 0;
				for (int i = 0; i < FEATURE_NUM; i ++) {
					value += TT[featureId][i] * V[i];
				}
				itemFeature[itemIndex ++] = value;
			}
		}
		if (recordIndex != ratingsInItemOrder.length) {
			throw new RuntimeException(String.format(
				"recordIndex[%s] != ratingsInItemOrder.length[%s]",
				recordIndex, ratingsInItemOrder.length));
		}
		if (itemIndex != ITEM_NUM * FEATURE_NUM) {
			throw new RuntimeException(String.format(
				"itemIndex[%s] != ITEM_NUM * FEATURE_NUM[%s]",
				itemIndex, ITEM_NUM * FEATURE_NUM));
		}
	}

	private void updateUserCategoryPref() {
		int recordIndex      = 0;
		int userIndex        = 0;
		int catIndex         = 0;
		int userCatPrefIndex = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			//int catNum          = categoryNumInUserOrder[userId];
			int existedCatIndex = itemCategoryIndexInUserOrder[catIndex];
			int catNum          = categoryNumInUserOrder[userId];
			int catCount        = 0;
			for (int catId = 0; catId < CATEGORY_NUM; catId ++) {
				double pnumerator   = 0;
				double pdenominator = 0;
				if (catCount < catNum && catId == existedCatIndex) {
					int itemNum = itemCategoryNumInUserOrder[catIndex];
					for (int index = 0; index < itemNum; index ++) {
						int itemId = itemIndexInUserOrder[recordIndex];
						int itemStartIndex = itemId * FEATURE_NUM;
						double w_minus_1   = weightsMinusOneInUserOrder[recordIndex];
						double w           = weightsInUserOrder[recordIndex];
						double pref        = prefsInUserOrder[recordIndex];
						double uTv         = 0;
						for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
							uTv += userFeature[userIndex + featureId] *
								itemFeature[itemStartIndex + featureId];
						}
						pnumerator   += pref * w * uTv;
						pdenominator += w_minus_1 * uTv * uTv;
						recordIndex ++;
					}
					catIndex ++;
					catCount ++;
					if (catIndex < CATEGORY_RECORD_NUM) {
						existedCatIndex = itemCategoryIndexInUserOrder[catIndex];
					}
				}
				int catStartIndex = catId * FEATURE_NUM_SQUARE;
				for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
					double fvalue = userFeature[userIndex + featureId];
					for (int f = 0; f < FEATURE_NUM; f ++) {
						pdenominator += userFeature[userIndex + f] *
							itemTitemSumInCategory[catStartIndex ++] *
							fvalue;
					}
				}
				double v = (pnumerator - categoryPrefOffsets[userId] * pdenominator) /
						(lambdaCat + pdenominator);
				userCategoryPref[userCatPrefIndex ++] = v > 0 ? v : 0;
			}
			userIndex += FEATURE_NUM;
			if (catCount != catNum) throw new RuntimeException(String.format(
				"Cannot match for userId[%s]: count[%s] != catNum[%s]",
				userId, catCount, catNum));
		}
		if (recordIndex != ratingsInUserOrder.length) {
			throw new RuntimeException(String.format(
				"recordIndex[%s] != ratingsInUserOrder.length[%s]",
				recordIndex, ratingsInUserOrder.length));
		}
		if (userIndex != USER_NUM * FEATURE_NUM) {
			throw new RuntimeException(String.format(
				"userIndex[%s] != USER_NUM * FEATURE_NUM[%s]",
				userIndex, USER_NUM * FEATURE_NUM));
		}
		if (catIndex != CATEGORY_RECORD_NUM) {
			throw new RuntimeException(String.format(
				"catIndex[%s] != CATEGORY_RECORD_NUM[%s]",
				catIndex, CATEGORY_RECORD_NUM));
		}
		if (userCatPrefIndex !=  USER_NUM * CATEGORY_NUM) {
			throw new RuntimeException(String.format(
				"userCatPrefIndex[%s] !=  USER_NUM * CATEGORY_NUM[%s]",
				userCatPrefIndex, USER_NUM * CATEGORY_NUM));
		}
	}

	@SuppressWarnings("unused")
	private void updateCatPrefOffset() {
		int recordIndex = 0;
		int userIndex   = 0;
		int catIndex    = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int catNum            = categoryNumInUserOrder[userId];
			int catPrefStartIndex = userId * CATEGORY_NUM;
			double numerator      = 0;
			double denominator    = 0;
			for (int i = 0; i < catNum; i++) {
				int catId      = itemCategoryIndexInUserOrder[catIndex];
				int itemNum    = itemCategoryNumInUserOrder[catIndex];
				double catPref = userCategoryPref[catPrefStartIndex + catId];
				catIndex ++;
				for (int j = 0; j < itemNum; j ++) {
					int itemId         = itemIndexInUserOrder[recordIndex];
					int itemStartIndex = itemId * FEATURE_NUM;
					double w_minus_1   = weightsMinusOneInUserOrder[recordIndex];
					double w           = weightsInUserOrder[recordIndex];
					double pref        = prefsInUserOrder[recordIndex];
					double v           = 0;
					for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
						v += itemFeature[itemStartIndex + featureId] *
								userFeature[userIndex + featureId];
					}
					denominator += w_minus_1 * v* v;
					numerator   += pref * w * v - w_minus_1 * catPref * v* v;
					recordIndex ++;
				}
				int catStartIndex = catId * FEATURE_NUM_SQUARE;
				for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
					double fvalue = userFeature[userIndex + featureId];
					for (int f = 0; f < FEATURE_NUM; f ++) {
						numerator -= userFeature[userIndex + f] *
							itemTitemSumInCategory[catStartIndex ++] * fvalue;
					}
				}
			}
			int sumIndex = 0;
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				double fvalue = userFeature[userIndex + featureId];
				for (int f = 0; f < FEATURE_NUM; f ++) {
					denominator += userFeature[userIndex + f] *
									itemTitemSumAll[sumIndex ++] * fvalue;
				}
			}
			double value = numerator / (denominator + lambdaPrefOffset);
			//if (value < 0) value = 0; 
			categoryPrefOffsets[userId] = value;
			userIndex += FEATURE_NUM;
		}
		
		if (userIndex != USER_NUM * FEATURE_NUM) {
			throw new RuntimeException(String.format(
				"userIndex[%s] != USER_NUM * FEATURE_NUM[%s]",
				userIndex, USER_NUM * FEATURE_NUM));
		}
		if (recordIndex != ratingsInUserOrder.length) {
			throw new RuntimeException(String.format(
				"recordIndex[%s] != ratingsInUserOrder.length[%s]",
				recordIndex, ratingsInUserOrder.length));
		}
		if (catIndex != CATEGORY_RECORD_NUM) {
			throw new RuntimeException(String.format(
				"catIndex[%s] != CATEGORY_RECORD_NUM[%s]",
				catIndex, CATEGORY_RECORD_NUM));
		}
	}

	private void updateUserTUserSumInCategory() {
		userTuserSumInCategory = new double[CATEGORY_NUM * FEATURE_NUM_SQUARE];
		int startIndex         = 0;
		int userCatIndex       = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			double[] userTuserSum = new double[FEATURE_NUM_SQUARE];
			int index             = 0;
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				double v = userFeature[startIndex + featureId];
				for (int f = 0; f < FEATURE_NUM; f ++) {
					userTuserSum[index ++] = v * userFeature[startIndex + f];
				}
			}
			int catIndex = 0;
			for (int catId = 0; catId < CATEGORY_NUM; catId ++) {
				double catPref = userCategoryPref[userCatIndex ++] +
						categoryPrefOffsets[userId];
				for (int i = 0; i < FEATURE_NUM_SQUARE; i ++) {
					userTuserSumInCategory[catIndex ++] +=
						catPref * catPref * userTuserSum[i];
				}
			}
			startIndex += FEATURE_NUM;
		}
		if (startIndex != USER_NUM * FEATURE_NUM) {
			throw new RuntimeException(String.format(
				"startIndex[%s] != USER_NUM * FEATURE_NUM[%s]",
				startIndex, USER_NUM * FEATURE_NUM));
		}
		if (userCatIndex  != USER_NUM * CATEGORY_NUM) {
			throw new RuntimeException(String.format(
				"userCatIndex[%s]  != USER_NUM * CATEGORY_NUM[%s]",
				userCatIndex, USER_NUM * CATEGORY_NUM));
		}
	}

	private void updateItemTItemSumInCategory() {
		itemTitemSumInCategory = new double[CATEGORY_NUM * FEATURE_NUM_SQUARE];
		int startIndex         = 0;
		for (int itemId = 0; itemId < ITEM_NUM; itemId ++) {
			int catId = categoryIndexForItem[itemId];
			int index = catId * FEATURE_NUM_SQUARE;
			for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
				double v = itemFeature[startIndex + featureId];
				for (int f = 0; f < FEATURE_NUM; f ++) {
					itemTitemSumInCategory[index ++] +=
						v * itemFeature[startIndex + f];
				}
			}
			startIndex += FEATURE_NUM;
		}
		if (startIndex != ITEM_NUM * FEATURE_NUM) {
			throw new RuntimeException(String.format(
				"startIndex[%s] != ITEM_NUM * FEATURE_NUM[%s]",
				startIndex, ITEM_NUM * FEATURE_NUM));
		}
		itemTitemSumAll = new double[FEATURE_NUM_SQUARE];
		startIndex      = 0;
		for (int catId = 0; catId < CATEGORY_NUM; catId ++) {
			for (int i = 0; i < FEATURE_NUM_SQUARE; i ++) {
				itemTitemSumAll[i] += itemTitemSumInCategory[startIndex++];
			}
		}
		if (startIndex != itemTitemSumInCategory.length) {
			throw new RuntimeException(String.format(
				"startIndex[%s] != CATEGORY_NUM * FEATURE_NUM_SQUARE[%s]",
				startIndex, itemTitemSumInCategory.length));
		}
	}

	@Override
	protected void update1Iter() {
		updateItemFeature();
		updateItemTItemSumInCategory();
		
		updateUserFeature();
		updateUserCategoryPref();
		updateUserTUserSumInCategory();
	}

	@Override
	protected double calLoss() throws CheckinException {
		double lbvalue  = 0;
		// (R - U'V)_2
		int recordIndex = 0;
		double tmpValue = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int itemNum  = itemNumInUserOrder[userId];
			for (int i = 0; i < itemNum; i ++) {
				int itemId  = itemIndexInUserOrder[recordIndex];
				double pref = prefsInUserOrder[recordIndex];
				int catId   = categoryIndexForItem[itemId]; 
				double v    = predict(USER_NUM, ITEM_NUM,
						CATEGORY_NUM, FEATURE_NUM, userFeature,
							itemFeature, userCategoryPref, userId,
								itemId, catId, categoryPrefOffsets[userId]);
				tmpValue += weightsMinusOneInUserOrder[recordIndex] *
						(pref - v) * (pref- v) + pref * pref - 2 * v * pref;
				recordIndex ++;
			}
		}
		lbvalue += 0.5 * tmpValue;

		tmpValue = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			for (int itemId = 0; itemId < ITEM_NUM; itemId ++) {
				int catId  = categoryIndexForItem[itemId];
				double v   = predict(USER_NUM, ITEM_NUM,
							CATEGORY_NUM, FEATURE_NUM, userFeature,
								itemFeature, userCategoryPref, userId,
									itemId, catId, categoryPrefOffsets[userId]);
				tmpValue += v * v;
			}
		}
		lbvalue += 0.5 * tmpValue;

		// V'V
		lbvalue += 0.5 * lambdaItem * Utils.cal2norm(itemFeature);
		if (Double.isNaN(lbvalue) || Double.isInfinite(lbvalue)) {
			System.err.println("V'V : lbvalue = " + lbvalue);
		}

		// U'U
		lbvalue += 0.5 * lambdaUser * Utils.cal2norm(userFeature);
		if (Double.isNaN(lbvalue) || Double.isInfinite(lbvalue)) {
			System.err.println("U'U : lbvalue = " + lbvalue);
		}

		// C'*C
		lbvalue += 0.5 * lambdaCat * Utils.cal2norm(userCategoryPref);
		if (Double.isNaN(lbvalue) || Double.isInfinite(lbvalue)) {
			System.err.println("C'C : lbvalue = " + lbvalue);
		}
		
		return lbvalue;
	}

	@Override
	protected void initModel(ParamManager paramManager) throws CheckinException{
		lambdaUser         = paramManager.getLambdaUser();
		lambdaItem         = paramManager.getLambdaItem();
		FEATURE_NUM_SQUARE = FEATURE_NUM * FEATURE_NUM;
		
		// parse options
		Properties options = Utils.parseCMD(paramManager.getOptions()
									.getProperty(getMethodName()));
		Utils.check(options, STRING_CATEGORY_NUM);
		CATEGORY_NUM = Integer.parseInt(options.getProperty(STRING_CATEGORY_NUM));
		if (options.containsKey(STRING_LAMBDA_CAT)) {
			lambdaCat = Float.parseFloat(options.getProperty(STRING_LAMBDA_CAT));
		}
		if (options.containsKey(STRING_CAT_OFFSET)) {
			catPrefOffSetConstant = Float.parseFloat(
					options.getProperty(STRING_CAT_OFFSET));
		}
		
		File augmentedRatingFile = null;
		if (augmentRating) {
			augmentedRatingFile = parseAugOption(USER_NUM, augRatingTopNum,
					userSimRatio, paramManager.getTrainFile(), outputPath,
						options, params);
		}
		
		Utils.check(options, STRING_LOCATION_CAT_FILE);
		File locationCategoryFile = new File(options.getProperty(
										STRING_LOCATION_CAT_FILE));
		params.setProperty(STRING_LOCATION_CAT_FILE,
										locationCategoryFile.getAbsolutePath());
		
		parseOptions(options);
		
		loadItemCategory(locationCategoryFile);
		if (augmentRating) {
			System.out.println("Load train aug rating file : " +
							augmentedRatingFile.getAbsolutePath());
			parseTrainRatingFileInCatOrder(augmentedRatingFile);
		} else {
			System.out.println("Load train rating file : " + paramManager.getTrainFile());
			parseTrainRatingFileInCatOrder(paramManager.getTrainFile());
		}
		initWeight(options);
		
		checkArray();

		itemFeature      = initFeature(ITEM_NUM * FEATURE_NUM,
								paramManager.getItemFeatureInitFile());
		userFeature      = initFeature(USER_NUM * FEATURE_NUM,
								paramManager.getUserFeatureInitFile());
		userCategoryPref = initFeature(USER_NUM * CATEGORY_NUM, null);
		initCategoryPrefOffsets();
		
		updateItemTItemSumInCategory();
		updateUserCategoryPref();
		updateUserTUserSumInCategory();
	}
	
	@Override
	protected double calRatingDiff(double predRating, double groundtruthRating) {
		return predRating - 1;
	}
	
	@Override
	protected String getMethodName() {
		return METHOD_NAME;
	}
	
	protected Properties getParams() {
		Properties props = super.getParams();
		
		props.put(CheckinConstants.STRING_LAMBDA_ITEM,	String.valueOf(lambdaItem));
		props.put(CheckinConstants.STRING_LAMBDA_USER,	String.valueOf(lambdaUser));
		props.put(STRING_CATEGORY_NUM,	String.valueOf(CATEGORY_NUM));
		props.put(STRING_LAMBDA_CAT, 	String.valueOf(lambdaCat));
		props.put("CATEGORY_RECORD_NUM",String.valueOf(CATEGORY_RECORD_NUM));
		props.put(STRING_CAT_OFFSET,    String.valueOf(catPrefOffSetConstant));
		
		props.putAll(params);
		return props;
	}

	@Override
	protected void storeLatentVariables() {
		// store item feature
		Utils.store2DArray(itemFeature, FEATURE_NUM,
			new File(outputPath, CheckinConstants.STRING_ITEM_FEATURE));
		// store user feature
		Utils.store2DArray(userFeature, FEATURE_NUM,
			new File(outputPath, CheckinConstants.STRING_USER_FEATURE));
		// store user category pref
		Utils.store2DArray(userCategoryPref, CATEGORY_NUM,
			new File(outputPath, FILE_NAME_USER_CATEGORY_PREF));
		Utils.save(categoryPrefOffsets, "\n",
			new File(outputPath, FILE_NAME_USER_CATPREF_OFFSET));
	}
	
	@Override
	protected double predict(int userId, int itemId) {
		return predict(USER_NUM, ITEM_NUM, CATEGORY_NUM, FEATURE_NUM,
				userFeature, itemFeature, userCategoryPref, userId, itemId,
					categoryIndexForItem[itemId], categoryPrefOffsets[userId]);
	}
	
	public static double predict(final int USER_NUM, final int ITEM_NUM,
			final int CATEGORY_NUM, final int FEATURE_NUM, double[] userFeature,
				double[] itemFeature, double[] userCategoryPref, int userId,
					int itemId, int categoryId, double categoryPrefOffset) {
		int userStartIndex   = userId * FEATURE_NUM;
		int itemStartIndex   = itemId * FEATURE_NUM;
		double predictRating = 0;
		for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
			predictRating += userFeature[userStartIndex + featureId] *
					itemFeature[itemStartIndex + featureId];
		}
		return predictRating * (
			userCategoryPref[userId * CATEGORY_NUM + categoryId] +
			categoryPrefOffset);
	}

	@Override
	protected void parseTrainRatingFile(File trainRatingFile)
									throws CheckinException {
		// do do nothing
	}
	
	/*
	 * The line format is as follows:
	 * 	UserId '\t' ItemId '\t' RawRating
	 */
	private void parseTrainRatingFileInCatOrder(File trainRatingFile)
											throws CheckinException {
		BufferedReader reader = null;
		BufferedWriter writer = null;
		try {
			System.out.println(String.format("[Info] FORCE_CHECK = %s", FORCE_CHECK));
			// load rating in item order
			Map<Integer, Map<Integer, Float>> itemUserRatingMap =
					new HashMap<Integer, Map<Integer, Float>>();
			String line       = null;
			reader            = new BufferedReader(new InputStreamReader(
									new FileInputStream(trainRatingFile)));
			int minItemId     = Integer.MAX_VALUE;
			int maxItemId     = -1;
			RATING_RECORD_NUM = 0;
			while ((line = reader.readLine()) != null) {
				if ("".equals(line = line.trim())) continue;
				String[] array = line.split("\t");
				if (array.length < 3) {
					reader.close(); reader = null;
					throw new CheckinException("Failed to parse TrainRatingFile: %s", line);
				}
				int userId   = Integer.parseInt(array[0]);
				int itemId   = Integer.parseInt(array[1]);
				float rating = Float.parseFloat(array[2]);
				Map<Integer,Float>userRatingMap = itemUserRatingMap.get(itemId);
				if (userRatingMap == null) {
					userRatingMap = new HashMap<Integer, Float>();
					userRatingMap.put(userId, rating);
					itemUserRatingMap.put(itemId, userRatingMap);
				} else {
					if (userRatingMap.containsKey(userId)) {
						reader.close(); reader = null;
						throw new CheckinException(
								"User %s has existed. Line : %s", userId, line);
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
				int[] itemIds = Utils.getSortedKeys(itemUserRatingMap.keySet());
				int index     = 0;
				for (int itemId : itemIds) {
					Map<Integer, Float> userRatingMap =
									itemUserRatingMap.get(itemId);
					int[] userIds = Utils.getSortedKeys(userRatingMap.keySet());
					for (int userId : userIds) {
						ratingsInItemOrder[index]   = userRatingMap.get(userId);
						userIndexInItemOrder[index] = userId;
						index ++;
					}
					userNumInItemOrder[itemId] = userRatingMap.size();
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
					reader.close(); reader = null;
					throw new CheckinException("Failed to parse TrainRatingFile: %s", line);
				}
				int userId   = Integer.parseInt(array[0]);
				int itemId   = Integer.parseInt(array[1]);
				float rating = Float.parseFloat(array[2]);
				Map<Integer,Float>itemRatingMap = userItemRatingMap.get(userId);
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
				if (userId < minUserId) minUserId = userId;
				if (userId > maxUserId) maxUserId = userId;
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
				int[] userIds = Utils.getSortedKeys(userItemRatingMap.keySet());
				CATEGORY_RECORD_NUM = 0;
				for (int userId : userIds) {
					Map<Integer, Float> itemRatingMap = 
								userItemRatingMap.get(userId);
					int[] itemIds = Utils.getSortedKeys(itemRatingMap.keySet());
					Set<Integer> categorySet = new HashSet<Integer>();
					for (int itemId : itemIds) {
						int catId = categoryIndexForItem[itemId];
						if (! categorySet.contains(catId))
							categorySet.add(catId);
					}
					CATEGORY_RECORD_NUM += categorySet.size();
				}
				ratingsInUserOrder   = new float[RATING_RECORD_NUM];
				itemIndexInUserOrder = new int[RATING_RECORD_NUM]; 
				itemNumInUserOrder   = new int[USER_NUM];
				itemCategoryIndexInUserOrder = new int[CATEGORY_RECORD_NUM];
				itemCategoryNumInUserOrder   = new int[CATEGORY_RECORD_NUM];
				categoryNumInUserOrder       = new int[USER_NUM];
				int index         = 0;
				int categoryIndex = 0;
				System.out.println("CATEGORY_RECORD_NUM= " +CATEGORY_RECORD_NUM);
				for (int userId = 0; userId < USER_NUM; userId ++) {
					Map<Integer, Float> itemRatingMap =
										userItemRatingMap.get(userId);
					if (itemRatingMap == null) {
						throw new CheckinException(
							"No found training record for userId: " + userId);
					}
					int[] itemIds = Utils.getSortedKeys(itemRatingMap.keySet());
					Map<Integer, Set<Integer>> catItemMap =
									new HashMap<Integer, Set<Integer>>();
					for (int itemId : itemIds) {
						int catId = categoryIndexForItem[itemId];
						Set<Integer> itemSet = catItemMap.get(catId);
						if (itemSet == null) {
							itemSet = new HashSet<Integer>();
							catItemMap.put(catId, itemSet);
						}
						itemSet.add(itemId);
					}
					int[] catIds = Utils.getSortedKeys(catItemMap.keySet());
					if (catIds.length == 0) {
						throw new CheckinException("Not category ofr userId:" + userId);
					}
					for (int catId : catIds) {
						itemIds = Utils.getSortedKeys(catItemMap.get(catId));
						for (int itemId : itemIds) {
							ratingsInUserOrder[index]   = itemRatingMap.get(itemId);
							itemIndexInUserOrder[index] = itemId;
							index ++;
						}
						itemCategoryIndexInUserOrder[categoryIndex] = catId;
						itemCategoryNumInUserOrder[categoryIndex]   = itemIds.length;
						categoryIndex ++;
					}
					itemNumInUserOrder[userId]     = itemRatingMap.size();
					categoryNumInUserOrder[userId] = catIds.length;
				}
				if (categoryIndex != CATEGORY_RECORD_NUM) {
					throw new CheckinException(
						"categoryIndex[%s] != CATEGORY_RECORD_NUM[%s]",
							categoryIndex, CATEGORY_RECORD_NUM);
				}
				if (index != RATING_RECORD_NUM) {
					throw new CheckinException("index[%s] != RATING_RECORD_NUM[%s]",
							index, RATING_RECORD_NUM);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader);
			Utils.cleanup(writer);
		}
	}

	private void initCategoryPrefOffsets() {
		categoryPrefOffsets = new double[USER_NUM];
		for (int i = categoryPrefOffsets.length-1; i >= 0; i --) {
			//categoryPrefOffsets[i] = Math.random() * 0.1;
			categoryPrefOffsets[i] = catPrefOffSetConstant;
		}
	}

	protected void loadItemCategory(File locationCategoryFile)
						throws CheckinException {
		categoryIndexForItem = loadItemCategory(locationCategoryFile,
								ITEM_NUM, CATEGORY_NUM);
	}
	
	public static File parseAugOption(int userNum, int augRatingTopNum,
			float userSimRatio, File trainRatingFile, File outputPath,
				Properties options, Properties params) throws CheckinException {
		File augmentedRatingFile = null;
		if (options.getProperty(STRING_AUG_TRAING_RATING_FILE) != null) {
			augmentedRatingFile = new File(options.getProperty(
										STRING_AUG_TRAING_RATING_FILE));
			Utils.exists(augmentedRatingFile);
		} else {
			augmentedRatingFile = new File(outputPath, "Augmented_Train_Rating");
		}
		params.setProperty(STRING_AUG_TRAING_RATING_FILE,
						augmentedRatingFile.getAbsolutePath());
		if (! augmentedRatingFile.exists()) {
			Utils.check(options, STRING_DIRECT_FRIEND_FILE);
			Utils.check(options, STRING_LOCATION_FRIEND_FILE);
			Utils.check(options, STRING_NEIGHBOR_FRIEND_FILE);
			Utils.check(options, STRING_USER_HOME_LATLNG_FILE);
			Utils.check(options, STRING_LOCATION_LATLNG_FILE);
			
			File directFriendFile   = new File(options.getProperty(
										STRING_DIRECT_FRIEND_FILE));
			File locationFriendFile = new File(options.getProperty(
										STRING_LOCATION_FRIEND_FILE));
			File neighborFriendFile = new File(options.getProperty(
										STRING_NEIGHBOR_FRIEND_FILE));
			File userHomeLatLngFile = new File(options.getProperty(
										STRING_USER_HOME_LATLNG_FILE));
			File locationLatLngFile = new File(options.getProperty(
										STRING_LOCATION_LATLNG_FILE));
			Utils.exists(userHomeLatLngFile);
			Utils.exists(locationLatLngFile);
			
			createPotCheckins8LA(userNum, augRatingTopNum, userSimRatio,
				trainRatingFile, directFriendFile, locationFriendFile,
					neighborFriendFile, userHomeLatLngFile, locationLatLngFile,
							augmentedRatingFile);
			
			params.setProperty(STRING_DIRECT_FRIEND_FILE,   
								directFriendFile.getAbsolutePath());
			params.setProperty(STRING_LOCATION_FRIEND_FILE,
								locationFriendFile.getAbsolutePath());
			params.setProperty(STRING_NEIGHBOR_FRIEND_FILE,
								neighborFriendFile.getAbsolutePath());
			params.setProperty(STRING_USER_HOME_LATLNG_FILE,
								userHomeLatLngFile.getAbsolutePath());
			params.setProperty(STRING_LOCATION_LATLNG_FILE,
								locationLatLngFile.getAbsolutePath());
		}
		return augmentedRatingFile;
	}

	// only for debug
	private void checkArray() throws CheckinException {
		int recordIndex = 0;
		int catIndex    = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int catNum = categoryNumInUserOrder[userId];
			for (int index = 0; index < catNum; index ++) {
				int catId   = itemCategoryIndexInUserOrder[catIndex];
				if (index > 0 && catId <= itemCategoryIndexInUserOrder[catIndex - 1]) {
					throw new CheckinException("CategoryIndex is not sorted in increasing order for userId:" + userId);
				}
				int itemNum = itemCategoryNumInUserOrder[catIndex];
				catIndex ++;
				for (int i = 0; i < itemNum; i ++) {
					int itemId = itemIndexInUserOrder[recordIndex];
					if (catId != categoryIndexForItem[itemId]) {
						throw new CheckinException("userID=%s, itemId=%s, itemIdCatId=%s, catId=%s",
							userId, itemId, categoryIndexForItem[itemId], catId);
					}
					recordIndex ++;
				}
			}
		}
		catIndex = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int existedCatIndex = itemCategoryIndexInUserOrder[catIndex];
			int catNum          = categoryNumInUserOrder[userId];
			int catCount        = 0;
			for (int catId = 0; catId < CATEGORY_NUM; catId ++) {
				if (catCount < catNum && catId == existedCatIndex) {
					catIndex ++;
					catCount ++;
					if (catIndex < CATEGORY_RECORD_NUM) {
						existedCatIndex = itemCategoryIndexInUserOrder[catIndex];
					}
				}
			}
			if (catCount != catNum) throw new CheckinException(
				"Cannot match for userId[%s]: count[%s] != catNum[%s]",
				userId, catCount, catNum);
		}
		if (catIndex != CATEGORY_RECORD_NUM) {
			throw new RuntimeException(String.format(
				"recordIndex[%s] != ratingsInUserOrder.length[%s]",
				recordIndex, ratingsInUserOrder.length));
		}
	}

	public static void main(String[] args) throws CheckinException{

		new ASMF(CheckinConstants.CONFIG_MANAGER) .estimate();
	}
}


