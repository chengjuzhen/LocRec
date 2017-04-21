package com.uncc.checkin.amf;

import com.uncc.checkin.CheckinConstants;
import com.uncc.checkin.CheckinException;
import com.uncc.checkin.ParamManager;
import com.uncc.checkin.util.Utils;
import com.uncc.checkin.fw.BasicRankMF;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

/*
 * Options:
 * ARMF: -CatNum <int> -LocCat <file> -AugRating <file>
 * 		[Optional]
 *			-LambdaCat <float>
 *			-CatOffset <float>
 *			-AugRatingTopNum <int>
 *			-UserSimRatio <float>
 * 			-HomeLatLng <file>
 * 			-LocLatLng <file> 
 * 			-DFriend <file>
 * 			-LFreind <file>
 * 			-NFriend <file>
 */
public class ARMF extends BasicRankMF {
	public static final String METHOD_NAME = "ARMF";
	
	private double[] userFeature         = null;
	private double[] itemFeature         = null;
	private double[] userCategoryPref    = null;
	private double[] categoryPrefOffsets = null;
	
	protected int[] categoryIndexForItem                = null;
	private int[] potentialItemNumInUserOrder           = null;
	private int[] potentialItemIndexInUserOrder         = null;
	private int[] potentialItemRecStartIndexInUserOrder = null;

	private int CATEGORY_NUM;
	private int POTENTIAL_RATING_RECORD_NUM;

	private float lambdaItem = 0.01f;
	private float lambdaUser = 0.01f;
	private float lambdaCat  = 50f;
	private float catPrefOffSetConstant = 0.1f;
	
	private float userSimRatio  = 0.2f;  // 0.5
	private int augRatingTopNum = 1000; // 500;
	
	private Properties params = new Properties();
	private double loss       = 0;
	
	public ARMF(ParamManager paramManager) throws CheckinException{
		init(paramManager);
	}
	
	// return loss
	private double update(double learningRate, int loopNumPerIter) {
		double _lloss = 0;
		for (int iter = 0; iter < loopNumPerIter; iter ++) {
			int userId           = -1;
			int observedItemId   = -1;
			int potentialItemId  = -1;
			int obsIemStartIndex = -1;
			int obsItemEndIndex  = -1;
			int potIemStartIndex = -1;
			int potItemEndIndex  = -1;
			while (true){
				userId         = (int)(Math.random() * USER_NUM);
				int obsItemNum = itemNumInUserOrder[userId];
				int potItemNum = potentialItemNumInUserOrder[userId];
		
				if (obsItemNum != 0) {
					int nth          = (int)(Math.random() * obsItemNum);
					obsIemStartIndex = itemRecStartIndexInUserOrder[userId];
					obsItemEndIndex  = obsIemStartIndex + obsItemNum - 1;
					observedItemId   = itemIndexInUserOrder[obsIemStartIndex + nth];
				}

				if (potItemNum == 0) {
					if (observedItemId == -1) {
						throw new RuntimeException(String.format(
							"UserID:%s, PotentialItemNum:%s",
							userId, potItemNum));
					}
					break;
				}
				
				int nth          = (int)(Math.random() * potItemNum);
				potIemStartIndex = potentialItemRecStartIndexInUserOrder[userId];
				potItemEndIndex  = potIemStartIndex + potItemNum - 1;
				potentialItemId  = potentialItemIndexInUserOrder[potIemStartIndex + nth];
				break;
			}
			
	
			if (observedItemId != -1) {
				int unobservedItemId = -1;
				do {
					unobservedItemId = (int)(Math.random() * ITEM_NUM);
				} while (exists(itemIndexInUserOrder, obsIemStartIndex,
						obsItemEndIndex, unobservedItemId) ||
					(potentialItemId != -1 && exists(
							potentialItemIndexInUserOrder, potIemStartIndex,
								potItemEndIndex, unobservedItemId)) );
				_lloss += potentialItemId != -1 ? updateOneSample(
						userId, observedItemId, potentialItemId, 
							unobservedItemId, learningRate) :
								updateOneSample(userId, observedItemId, 
										unobservedItemId, learningRate);
			} else {
				int unobservedItemId = -1;
				do {
					unobservedItemId = (int)(Math.random() * ITEM_NUM);
				} while (exists(potentialItemIndexInUserOrder, potIemStartIndex,
						potItemEndIndex, unobservedItemId));
				_lloss += updateOneSample(userId, potentialItemId, 
						unobservedItemId, learningRate);
			}
		}
		
		return _lloss;
	}

	// return loss
	private double updateOneSample(int userId,  int observedItemId,
			int potentialItemId, int unobservedItemId,
					double learningRate) {
		double _lloss                 = 0;
		final double catOffset        = categoryPrefOffsets[userId];
		final int userStartIndex      = userId * FEATURE_NUM;
		final int userCatStartIndex   = userId * CATEGORY_NUM;
		final int obsItemStartIndex   = observedItemId * FEATURE_NUM;
		final int potItemStartIndex   = potentialItemId * FEATURE_NUM;
		final int unobsItemStartIndex = unobservedItemId * FEATURE_NUM;
		final int obsCatIndex   = userCatStartIndex +
									categoryIndexForItem[observedItemId];
		final int potCatIndex   = userCatStartIndex +
									categoryIndexForItem[potentialItemId];
		final int unobsCatIndex = userCatStartIndex +
									categoryIndexForItem[unobservedItemId];
		double r_ij = 0;
		double r_ik = 0;
		double r_ih = 0;
		for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
			double uf = userFeature[userStartIndex + featureId];
			r_ij += uf * itemFeature[obsItemStartIndex + featureId];
			r_ik += uf * itemFeature[potItemStartIndex + featureId];
			r_ih += uf * itemFeature[unobsItemStartIndex + featureId];
		}
		double e_r_ijk = 1 + Math.exp(r_ik - r_ij);
		double e_r_ikh = 1 + Math.exp(r_ih - r_ik);
		double mainGjk   = -1 + 1.0 / e_r_ijk;
		double mainGkh   = -1 + 1.0 / e_r_ikh;
	
		double weight_ijk = 1;
		double weight_ikh = 1;
		mainGjk *= weight_ijk;
		mainGkh *= weight_ikh;
		
		_lloss += weight_ijk * Math.log(e_r_ijk) + 
					weight_ikh * Math.log(e_r_ikh);
		double obsTotCatPref   = userCategoryPref[obsCatIndex] + catOffset;
		double potTotCatPref   = userCategoryPref[potCatIndex] + catOffset;
		double unobsTotCatPref = userCategoryPref[unobsCatIndex]+ catOffset;
		for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
			int ufIndex     = userStartIndex      + featureId;
			int obsfIndex   = obsItemStartIndex   + featureId;
			int potfIndex   = potItemStartIndex   + featureId;
			int unobsfIndex = unobsItemStartIndex + featureId;
			double uf       = userFeature[ufIndex];
			double obsf     = itemFeature[obsfIndex];
			double potf     = itemFeature[potfIndex]; 
			double unobsf   = itemFeature[unobsfIndex];
	
			// update user latent feature
			userFeature[ufIndex]     -= learningRate *
				(mainGjk * (obsf * obsTotCatPref - potf * potTotCatPref) +
				mainGkh * (potf * potTotCatPref - unobsf * unobsTotCatPref) +
				lambdaUser * uf);
			
			// update observed item latent feature
			itemFeature[obsfIndex]   -= learningRate *
				(mainGjk * uf * obsTotCatPref +
						lambdaItem * obsf);
			// update potential item latent feature
			itemFeature[potfIndex]   -= learningRate *
				( (mainGkh - mainGjk) * uf * potTotCatPref + 
						lambdaItem * potf);
			// update unobserved item latent feature
			itemFeature[unobsfIndex] -=  learningRate *
				(-1 * mainGkh * uf * unobsTotCatPref +
						lambdaItem * unobsf);
			
			// update loss
			_lloss += 0.5 *(lambdaUser * uf * uf + lambdaItem * 
					(obsf * obsf + potf * potf + unobsf * unobsf) );
		} 
		
		// update user catPref
		double ocp = userCategoryPref[obsCatIndex];
		double pcp = userCategoryPref[potCatIndex];
		double ucp = userCategoryPref[unobsCatIndex];
		userCategoryPref[obsCatIndex] -= learningRate * (
						mainGjk * r_ij + lambdaCat * ocp);
		userCategoryPref[potCatIndex] -= learningRate * (
						(mainGkh - mainGjk) * r_ik + lambdaCat * pcp);
		userCategoryPref[unobsCatIndex] -= learningRate * (
						-1 * mainGkh * r_ih + lambdaCat * ucp);
		
		_lloss += 0.5 * lambdaCat * (ocp * ocp + pcp * pcp + ucp * ucp);
		
		return _lloss;
	}

	private double updateOneSample(int userId, int potentialItemId,
			int unobservedItemId, double learningRate) {
		double loss                   = 0;
		final double catOffset        = categoryPrefOffsets[userId];
		final int userStartIndex      = userId * FEATURE_NUM;
		final int userCatStartIndex   = userId * CATEGORY_NUM;
		
		final int potItemStartIndex   = potentialItemId * FEATURE_NUM;
		final int unobsItemStartIndex = unobservedItemId * FEATURE_NUM;
		
		final int potCatIndex   = userCatStartIndex +
									categoryIndexForItem[potentialItemId];
		final int unobsCatIndex = userCatStartIndex +
									categoryIndexForItem[unobservedItemId];
		double r_ik = 0;
		double r_ih = 0;
		for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
			double uf = userFeature[userStartIndex + featureId];
			r_ik += uf * itemFeature[potItemStartIndex + featureId];
			r_ih += uf * itemFeature[unobsItemStartIndex + featureId];
		}
		double e_r_ikh = 1 + Math.exp(r_ih - r_ik);
		double mainGkh = -1 + 1.0 / e_r_ikh;
	
		loss +=  Math.log(e_r_ikh);
		double potTotCatPref   = userCategoryPref
			[potCatIndex] + catOffset;
		double unobsTotCatPref = userCategoryPref
			[unobsCatIndex]+ catOffset;
		for (int featureId = 0; featureId < FEATURE_NUM; featureId ++) {
			int ufIndex     = userStartIndex      + featureId;
			int potfIndex   = potItemStartIndex   + featureId;
			int unobsfIndex = unobsItemStartIndex + featureId;
			double uf       = userFeature[ufIndex];
			double potf     = itemFeature[potfIndex]; 
			double unobsf   = itemFeature[unobsfIndex];
	
			// update user latent feature
			userFeature[ufIndex]     -= learningRate *
				(mainGkh * (potf * potTotCatPref - unobsf * unobsTotCatPref) +
						lambdaUser * uf);
			
			// update potential item latent feature
			itemFeature[potfIndex]   -= learningRate *
				( mainGkh * uf * potTotCatPref +
						lambdaItem * potf);
			// update unobserved item latent feature
			itemFeature[unobsfIndex] -=  learningRate *
				(-1 * mainGkh * uf * unobsTotCatPref +
						lambdaItem * unobsf);
			
			// update loss
			loss += 0.5 *(lambdaUser * uf * uf +
						lambdaItem * ( potf * potf + unobsf * unobsf) );
		} 
		
		// update user catPref
		double pcp = userCategoryPref[potCatIndex];
		double ucp = userCategoryPref[unobsCatIndex];
		userCategoryPref[potCatIndex] -= learningRate * (
						mainGkh * r_ik + lambdaCat * pcp);
		userCategoryPref[unobsCatIndex] -= learningRate * (
						-1 * mainGkh * r_ih + lambdaCat * ucp);
		
		loss += 0.5 * lambdaCat * (ucp * ucp + pcp * pcp);
		
		return loss;
	}

	@Override
	protected double calLoss() {
		return loss;
	}
	
	@Override
	protected void update1Iter() throws CheckinException {
		loss = update(learningRate, USER_NUM * 100);
	}

	@Override
	protected void initModel(ParamManager paramManager) throws CheckinException {
		lambdaUser      = paramManager.getLambdaUser();
		lambdaItem      = paramManager.getLambdaItem();
		
		// parse options
		Properties options = Utils.parseCMD(paramManager.getOptions()
											.getProperty(getMethodName()));
		Utils.check(options, ASMF.STRING_CATEGORY_NUM);
		CATEGORY_NUM = Integer.parseInt(options
						.getProperty(ASMF.STRING_CATEGORY_NUM));
		if (options.containsKey(ASMF.STRING_LAMBDA_CAT)) {
			lambdaCat = Float.parseFloat(options
						.getProperty(ASMF.STRING_LAMBDA_CAT));
		}
		if (options.containsKey(ASMF.STRING_CAT_OFFSET)) {
			catPrefOffSetConstant = Float.parseFloat(
					options.getProperty(ASMF.STRING_CAT_OFFSET));
		}
		if (options.containsKey(AMF.STRING_USER_SIM_RATIO)) {
			userSimRatio = Float.parseFloat(options
						.getProperty(AMF.STRING_USER_SIM_RATIO));
		}
		if (options.containsKey(AMF.STRING_AUG_RATING_TOP_NUM)) {
			augRatingTopNum = Integer.parseInt(options
						.getProperty(AMF.STRING_AUG_RATING_TOP_NUM));
		}
		
		// all rating file including potential check-ins
		File augmentedRatingFile = ASMF.parseAugOption(USER_NUM,augRatingTopNum,
									userSimRatio, paramManager.getTrainFile(),
										outputPath, options, params);
		File potentialRatingFile = new File(outputPath, "Potential_Rating");
		
		selectPotentialRating(augmentedRatingFile, paramManager.getTrainFile(),
								potentialRatingFile);
		parsePotentialRatingFile(potentialRatingFile);
		
		Utils.check(options, ASMF.STRING_LOCATION_CAT_FILE);
		File locationCategoryFile =
				new File(options.getProperty(ASMF.STRING_LOCATION_CAT_FILE));
		params.setProperty(ASMF.STRING_LOCATION_CAT_FILE,
				locationCategoryFile.getAbsolutePath());
		
		categoryIndexForItem = AMF.loadItemCategory(locationCategoryFile,
				ITEM_NUM, CATEGORY_NUM); 
		
		//check();

		itemFeature      = initFeature(ITEM_NUM * FEATURE_NUM,
								paramManager.getItemFeatureInitFile());
		userFeature      = initFeature(USER_NUM * FEATURE_NUM,
								paramManager.getUserFeatureInitFile());
		userCategoryPref = initFeature(USER_NUM * CATEGORY_NUM, null);
		initCategoryPrefOffsets();
	}
	
	protected String getMethodName() {
		return METHOD_NAME;
	}
	
	@Override
	protected Properties getParams() {
		Properties props = super.getParams();
		props.put(CheckinConstants.STRING_LAMBDA_ITEM,	String.valueOf(lambdaItem));
		props.put(CheckinConstants.STRING_LAMBDA_USER,	String.valueOf(lambdaUser));
		props.put(ASMF.STRING_CATEGORY_NUM, 	String.valueOf(CATEGORY_NUM));
		props.put(ASMF.STRING_CAT_OFFSET,   	String.valueOf(catPrefOffSetConstant));
		props.put(ASMF.STRING_LAMBDA_CAT, 		String.valueOf(lambdaCat));
		props.put(AMF.STRING_AUG_RATING_TOP_NUM,String.valueOf(augRatingTopNum));
		props.put(AMF.STRING_USER_SIM_RATIO, 	String.valueOf(userSimRatio));
		props.put("PotentialRatingRecordNum", 	String.valueOf(POTENTIAL_RATING_RECORD_NUM));
		props.putAll(params);
		return props;
	}

	@Override
	protected double predict(int userId, int itemId) {
		return ASMF.predict(USER_NUM, ITEM_NUM, CATEGORY_NUM, FEATURE_NUM,
				userFeature, itemFeature, userCategoryPref, userId, itemId,
					categoryIndexForItem[itemId], categoryPrefOffsets[userId]);
	}
	
	@Override
	protected void storeLatentVariables() {
		// store item feature
		Utils.store2DArray(itemFeature, FEATURE_NUM,
				new File(outputPath, CheckinConstants.STRING_ITEM_FEATURE));
		// store user feature
		Utils.store2DArray(userFeature, FEATURE_NUM,
				new File(outputPath, CheckinConstants.STRING_USER_FEATURE));
		Utils.store2DArray(userCategoryPref, CATEGORY_NUM,
				new File(outputPath, ASMF.FILE_NAME_USER_CATEGORY_PREF));
		Utils.save(categoryPrefOffsets, "\n",
				new File(outputPath, ASMF.FILE_NAME_USER_CATPREF_OFFSET));
	}
	

	private void parsePotentialRatingFile(File potentialRatingFile)
						throws CheckinException {
		BufferedReader reader = null;
		try {
			// load rating in item order
			Map<Integer, Map<Integer, Float>> userItemRatingMap =
						new HashMap<Integer, Map<Integer, Float>>();
			reader = new BufferedReader(new InputStreamReader(
						new FileInputStream(potentialRatingFile)));
			boolean potCheck = false;
			String line      = null;
			int minUserId    = Integer.MAX_VALUE;
			int maxUserId    = -1;
			POTENTIAL_RATING_RECORD_NUM = 0;
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
				Map<Integer, Float> itemMap = userItemRatingMap.get(userId);
				if (itemMap == null) {
					itemMap = new HashMap<Integer, Float>();
					userItemRatingMap.put(userId, itemMap);
				}
				if (itemMap.containsKey(itemId)) {
					reader.close();reader=null;
					throw new CheckinException("PotentialRating:: userId:%s, itemId:%s, existed",
						userId, itemId);
				}
				itemMap.put(itemId, rating);
				if (userId < minUserId) minUserId = userId;
				if (userId > maxUserId) maxUserId = userId;
				POTENTIAL_RATING_RECORD_NUM ++;
			}
			reader.close();
			if (userItemRatingMap.size() != USER_NUM) {
				String errMsg = String.format("Failed to parse TrainRatingFile: user number does not match with predefined number." +
						"[PredefinedUserNum=%s][ParsedUserNum=%s]",
						USER_NUM, userItemRatingMap.size());
				if (potCheck) {
					throw new CheckinException(errMsg);
				} else {
					System.err.println(errMsg);
				}
			}
			if (minUserId != 0) {
				String errMsg = String.format("MinUserId=%s, instead of 0.",
												minUserId);
				if (potCheck) {
					throw new CheckinException(errMsg);
				} else {
					System.err.println(errMsg);
				}
			}
			if (maxUserId != USER_NUM - 1) {
				String errMsg = String.format("MaxUserId=%s, instead of %s.",
												maxUserId, USER_NUM - 1);
				if (potCheck) {
					throw new CheckinException(errMsg);
				} else {
					System.err.println(errMsg);
				}
			}

			potentialItemIndexInUserOrder= new int[POTENTIAL_RATING_RECORD_NUM]; 
			potentialItemNumInUserOrder  = new int[USER_NUM];
			potentialItemRecStartIndexInUserOrder = new int[USER_NUM];
			int[] userIds = Utils.getSortedKeys(userItemRatingMap.keySet());
			int index     = 0;
			for (int userId : userIds) {
				Map<Integer, Float> itemRatingMap = 
								userItemRatingMap.get(userId);
				int[] ItemIds = Utils.getSortedKeys(itemRatingMap.keySet());
				potentialItemRecStartIndexInUserOrder[userId] = index;
				potentialItemNumInUserOrder[userId]           = itemRatingMap.size();
				for (int itemId : ItemIds) {
					//double rating = itemRatingMap.get(itemId);
					potentialItemIndexInUserOrder[index] = itemId;
					index ++;
				}
			}
			if (index != POTENTIAL_RATING_RECORD_NUM)
				throw new CheckinException("index[%s] != POTENTIAL_RATING_RECORD_NUM[%s]",
						index, POTENTIAL_RATING_RECORD_NUM);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader);
		}
	}
	
	public static void selectPotentialRating(File allRatingFile,
			File trainRatingFile, File outputFile) throws CheckinException {
		 Map<Integer, Map<Integer, Float>> allUserItemRatingMap =
				 loadUserItemRatingMap(allRatingFile);
		 Map<Integer, Map<Integer, Float>> trainUserItemRatingMap =
				 loadUserItemRatingMap(trainRatingFile);
		 Map<Integer, Map<Integer, Float>> potentialUserItemRatingMap=
				 new HashMap<Integer, Map<Integer, Float>>();
		 for (int userId : allUserItemRatingMap.keySet()) {
			 Map<Integer, Float> allItemMap   =
					 allUserItemRatingMap.get(userId);
			 Map<Integer, Float> trainItemMap =
					 trainUserItemRatingMap.get(userId);
			 int potentialCount = 0;
			 if (trainItemMap != null) {
				 Map<Integer, Float> potentialItemMap =
						 new HashMap<Integer, Float>();
				 potentialUserItemRatingMap.put(userId, potentialItemMap);
				 for (int itemId : allItemMap.keySet()) {
					 if (trainItemMap.containsKey(itemId)) continue;
					 float rating = allItemMap.get(itemId);
					 potentialItemMap.put(itemId, rating);
				 }
				 potentialCount = potentialItemMap.size();
			 } else {
				 potentialUserItemRatingMap.put(userId, allItemMap);
				 potentialCount = allItemMap.size();
			 }
			 if (potentialCount == 0) {
				 /*throw new CheckinException(
					"UserID:%s, PotentialCount==0", userId);*/
			 }
		 }
		 allUserItemRatingMap   = null;
		 trainUserItemRatingMap = null;
		 BufferedWriter writer  = null;
		 try {
			 if (! outputFile.getParentFile().exists())
				 outputFile.getParentFile().mkdirs();
			 writer = new BufferedWriter(new OutputStreamWriter(
					 	new FileOutputStream(outputFile)));
			 int[] userIdArr = Utils.getSortedKeys(
					 	potentialUserItemRatingMap.keySet());
			 for (int userId : userIdArr) {
				 Map<Integer, Float> itemMap =
						 potentialUserItemRatingMap.get(userId);
				 int[] itemIdArr = Utils.getSortedKeys(itemMap.keySet());
				 for (int itemId : itemIdArr) {
					 float rating = itemMap.get(itemId);
					 writer.write(String.format("%s\t%s\t%s\n",
							 		userId, itemId, rating));
				 }
			 }
			 writer.close();
		 } catch (IOException e) {
			 e.printStackTrace();
		 }
		
	}
	
	@SuppressWarnings("unused")
	private static void mergePotentialRating(File trainRatingFile,
			File potentialRatingFile, File outputFile) throws CheckinException {
		 Map<Integer, Map<Integer, Float>> potentialUserItemRatingMap =
				 loadUserItemRatingMap(potentialRatingFile);
		 Map<Integer, Map<Integer, Float>> trainUserItemRatingMap =
				 loadUserItemRatingMap(trainRatingFile);
		 for (int userId : potentialUserItemRatingMap.keySet()) {
			 Map<Integer, Float> potItemMap   = potentialUserItemRatingMap.get(userId);
			 Map<Integer, Float> trainItemMap = trainUserItemRatingMap.get(userId);
			 if (trainItemMap != null) {
				 for (int itemId : potItemMap.keySet()) {
					 if (trainItemMap.containsKey(itemId)) {
						 throw new CheckinException(
							"UserId:%s, ItemId:%s, pot-train exist.",
							userId, itemId);
					 }
					 trainItemMap.put(itemId, potItemMap.get(itemId));
				 }
			 } else {
				 trainUserItemRatingMap.put(userId, potItemMap);
			 }
		 }
		 
		 BufferedWriter writer  = null;
		 try {
			 if (! outputFile.getParentFile().exists())
				 outputFile.getParentFile().mkdirs();
			 writer = new BufferedWriter(new OutputStreamWriter(
					 new FileOutputStream(outputFile)));
			 int[] userIdArr = Utils.getSortedKeys(
					 trainUserItemRatingMap.keySet());
			 for (int userId : userIdArr) {
				 Map<Integer, Float> itemMap =
					trainUserItemRatingMap.get(userId);
				 int[] itemIdArr = Utils.getSortedKeys(
						 itemMap.keySet());
				 for (int itemId : itemIdArr) {
					 float rating = itemMap.get(itemId);
					 writer.write(String.format("%s\t%s\t%s\n",
						userId, itemId, rating));
				 }
			 }
			 writer.close();
		 } catch (IOException e) {
			 e.printStackTrace();
		 }
		
	}

	private static Map<Integer, Map<Integer, Float>> loadUserItemRatingMap(
			File inputFile) throws CheckinException {
		Utils.exists(inputFile);
		BufferedReader reader = null;
		try {
			Map<Integer, Map<Integer, Float>> userItemMap =
				new HashMap<Integer, Map<Integer, Float>>();
			reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(inputFile)));
			String line = null;
			while ((line = reader.readLine()) != null) {
				String[] array = Utils.parseLine(line, 3);
				int userId     = Integer.parseInt(array[0]);
				int itemId     = Integer.parseInt(array[1]);
				float rating   = Float.parseFloat(array[2]);
				Map<Integer, Float> itemMap =
						userItemMap.get(userId);
				if (itemMap == null) {
					itemMap = new HashMap<Integer, Float>();
					userItemMap.put(userId, itemMap);
				}
				if (itemMap.containsKey(itemId)) {
					reader.close();reader=null;
					throw new CheckinException("userID:%s, ItemID:%s, exsited",
						userId, itemId);
				}
				itemMap.put(itemId, rating);
			}
			reader.close();
			return userItemMap;
		} catch (IOException e) {
			e.printStackTrace();throw new CheckinException(e);
		} finally {
			Utils.cleanup(reader);
		}
	}
	
	private void initCategoryPrefOffsets() {
		categoryPrefOffsets = new double[USER_NUM];
		for (int i = categoryPrefOffsets.length-1; i >= 0; i --) {
			categoryPrefOffsets[i] = catPrefOffSetConstant;
		}
	}
	
	private void check() throws CheckinException {
		Map<Integer, Set<Integer>> trainUserItemMap =
				new HashMap<Integer, Set<Integer>>();
		int index = 0;
		for(int userId = 0; userId < USER_NUM; userId ++) {
			int itemNum = itemNumInUserOrder[userId];
			Set<Integer> set = new HashSet<Integer>();
			for (int i = 0; i < itemNum; i ++) {
				int itemId = itemIndexInUserOrder[index++];
				if (set.contains(itemId)) {
					throw new RuntimeException(String.format(
						"UserId:%s, itemId:%s, existed",
						userId, itemId));
				}
				set.add(itemId);
				if (i > 0 && itemId <= itemIndexInUserOrder[index-2]) {
					throw new RuntimeException(String.format(
						"UserId:%s, itemId[%s] <= itemIndexInUserOrder[index-2][%s]",
						userId, itemId,itemIndexInUserOrder[index-2]));
				}
			}
			if (set.size() > 0) {
				trainUserItemMap.put(userId, set);
			}
		}
		if (index != RATING_RECORD_NUM)
			throw new CheckinException("index[%s] != RATING_RECORD_NUM[%s]",
					index, RATING_RECORD_NUM);
		index = 0;
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int itemNum = potentialItemNumInUserOrder[userId];
			Set<Integer> trainSet = trainUserItemMap.get(userId);
			for (int i = 0; i < itemNum; i ++) {
				int itemId = potentialItemIndexInUserOrder[index++];
				if (trainSet.contains(itemId)) {
					throw new CheckinException("userId:$s, potentialItemId:%s, in train.",
						userId, itemId);
				}
				if (i > 0 && itemId <= potentialItemIndexInUserOrder[index-2])
					throw new CheckinException("UserId:%s,itemId[%s] <= potentialItemIndexInUserOrder[index-2][%s]",
						userId, itemId, potentialItemIndexInUserOrder[index-2]);
			}
		}
	}
}


