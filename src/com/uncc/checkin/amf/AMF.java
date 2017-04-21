package com.uncc.checkin.amf;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import com.uncc.checkin.CheckinConstants;
import com.uncc.checkin.CheckinException;
import com.uncc.checkin.fw.BasicRatingMF;
import com.uncc.checkin.util.Utils;
import com.uncc.checkin.util.WeightSchema;

/*
 * Optional:
 * 			-AugRatingTopNum <int>
 * 			-PotRating <float>
 * 			-UserSimRatio <float>
 */
public abstract class AMF extends BasicRatingMF {
	public static final String FILE_NAME_HOME_POWERLAW_PARAM = "UserHome_PowerLawParam";
	public  static final String STRING_AUG_RATING_TOP_NUM  = "AugRatingTopNum";
	public  static final String STRING_USER_SIM_RATIO      = "UserSimRatio";
	private static final String STRING_POT_RATING          = "PotRating";
	public  static final float zeroDistanceDefaultValue    = 1f;
	
	protected double[] weightsInItemOrder         = null;
	protected double[] weightsInUserOrder         = null;
	protected double[] weightsMinusOneInItemOrder = null;
	protected double[] weightsMinusOneInUserOrder = null;
	protected double[] prefsInUserOrder           = null;
	protected double[] prefsInItemOrder           = null;
	
	protected float potRating        = 0.1f;
	protected float userSimRatio     = 0.5f; // 0.2
	protected int augRatingTopNum    = 500; // 1000;
	private boolean useLowWeight4Pot = false;
	
	protected final Properties params;
	
	public AMF() {
		this.params = new Properties();
	}

	@Override
	protected Properties getParams() {
		Properties props = super.getParams();
		props.put(STRING_AUG_RATING_TOP_NUM, String.valueOf(augRatingTopNum));
		props.put(STRING_POT_RATING, 		String.valueOf(potRating));
		props.put(STRING_USER_SIM_RATIO, 	String.valueOf(userSimRatio));
		return props;
	}
	
	@Override
	protected double calRatingDiff(double predRating, double groundtruthRating) {
		return predRating - 1;
	}
	
	protected void parseOptions(Properties options) {
		if (options.getProperty(STRING_POT_RATING) != null) {
			potRating = Float.parseFloat(
							options.getProperty(STRING_POT_RATING));
		}
		if (options.getProperty(STRING_USER_SIM_RATIO) != null) {
			userSimRatio = Float.parseFloat(
							options.getProperty(STRING_USER_SIM_RATIO));
		}
		if (options.getProperty(STRING_AUG_RATING_TOP_NUM) != null) {
			augRatingTopNum = Integer.parseInt(
							options.getProperty(STRING_AUG_RATING_TOP_NUM));
		}
	}

	protected void initWeight(Properties args) throws CheckinException {
		WeightSchema ws = new WeightSchema(args);
		params.putAll(ws.getParams());
		
		Map<Integer, Set<Integer>> origTrainUserLocationMap =
							Utils.loadUserLocationMap(trainFile, 3, 0, 1);
		int recordIndex            = 0;
		prefsInUserOrder           = new double[ratingsInUserOrder.length];
		weightsInUserOrder         = new double[ratingsInUserOrder.length];
		weightsMinusOneInUserOrder = new double[ratingsInUserOrder.length];
		Map<Integer, Map<Integer, Integer>> userItemIndexMap =
				new HashMap<Integer, Map<Integer, Integer>>();
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int itemNum = itemNumInUserOrder[userId];
			Set<Integer>trainLocationSet = origTrainUserLocationMap.get(userId);
			for (int index = 0; index < itemNum; index ++) {
				int itemId    = itemIndexInUserOrder[recordIndex];
				double rating = ratingsInUserOrder[recordIndex];
				double v      = ws.calWeightMinusOne(rating);
				if (trainLocationSet != null &&
						trainLocationSet.contains(itemId)) {
					prefsInUserOrder[recordIndex] = 1;
				} else {
					prefsInUserOrder[recordIndex] = potRating;
					if (useLowWeight4Pot) v = 0;
				}
				double w = v + 1;
				weightsInUserOrder[recordIndex]         = w;
				weightsMinusOneInUserOrder[recordIndex] = v;

				// store result in case of value's precision
				Map<Integer, Integer> itemIndexMap =
					userItemIndexMap.get(userId);
				if (itemIndexMap == null) {
					itemIndexMap = new HashMap<Integer, Integer>();
					userItemIndexMap.put(userId, itemIndexMap);
				}
				if (itemIndexMap.containsKey(itemId)) {
					throw new CheckinException(
						"ItemId[%s] has repeated for userId[%s].",
						itemId, userId);
				} else {
					itemIndexMap.put(itemId, recordIndex);
				}
				recordIndex ++;
			}
		}
		if (recordIndex != ratingsInUserOrder.length) {
			throw new CheckinException("recordIndex[%s] != ratingsInUserOrder.length[%s]",
				recordIndex, ratingsInUserOrder.length);
		}

		recordIndex                = 0;
		prefsInItemOrder           = new double[ratingsInItemOrder.length];
		weightsInItemOrder         = new double[ratingsInItemOrder.length];
		weightsMinusOneInItemOrder = new double[ratingsInItemOrder.length];
		boolean[] visitedStatus    = new boolean[ratingsInItemOrder.length];
		for (int itemId = 0; itemId < ITEM_NUM; itemId ++) {
			int userNum = userNumInItemOrder[itemId];
			for (int index = 0; index< userNum; index ++) {
				int userId = userIndexInItemOrder[recordIndex];
				Map<Integer, Integer> itemIndexMap =
						userItemIndexMap.get(userId);
				if (itemIndexMap == null ||
					! itemIndexMap.containsKey(itemId)) {
					throw new CheckinException(
							"Cannot find userId[%s], ItemId[%s]",userId,itemId);
				}
				int mappedIndex = itemIndexMap.get(itemId);
				prefsInItemOrder[recordIndex]           =
					prefsInUserOrder[mappedIndex];
				weightsInItemOrder[recordIndex]         =
					weightsInUserOrder[mappedIndex];
				weightsMinusOneInItemOrder[recordIndex] =
					weightsMinusOneInUserOrder[mappedIndex];
				visitedStatus[mappedIndex] = true;
				recordIndex ++;
			}
		}
		if (recordIndex != ratingsInUserOrder.length) {
			throw new CheckinException("recordIndex[%s] != ratingsInItemOrder.length[%s]",
				recordIndex, ratingsInItemOrder.length);
		}
		for (int i = visitedStatus.length - 1; i >= 0; i --) {
			if (! visitedStatus[i]) {
				throw new CheckinException("Error initWeight:: i = %s, do not have maps.", i);
			}
		}
	}

	/*
	 * trainRatingFile   : userId \t placeId \t rating
	 * locationLatLngFile: locationId \t lat \t lng
	 * userHomeLatLngFile: userId \t lat \t lng
	 * directFriendFile, locationFriendFile, neighboringFriendFile:
	 * 		userId \t friend_1 \t friend_2 ...
	 * 
	 * Output: joint potential checkins with observed checkins together
	 */
	public static void createPotCheckins8LA(int userNum, int augRatingTopNum,
			float userSimRatio, File trainRatingFile, File directFriendFile,
				File locationFriendFile, File neighboringFriendFile,
					File userHomeLatLngFile, File locationLatLngFile,
						File outputFile) throws CheckinException{
		System.out.println("Create AugRating File for  " +
						outputFile.getAbsolutePath());
		
		double[] userHomePowerLawParams = loadUserHomePowerLawParams(
							userHomeLatLngFile, trainRatingFile,
								locationLatLngFile, outputFile.getParentFile());
		BufferedReader reader = null;
		try {
			// loading data
			reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(trainRatingFile)));
			Map<Integer, Map<Integer, Integer>> trainUserPlaceRatingMap =
				new HashMap<Integer, Map<Integer, Integer>>();
			String line = null;
			while ((line = reader.readLine()) != null) {
				String[] array = Utils.parseLine(line, 3);
				int userID     = Integer.parseInt(array[0]);
				int placeID    = Integer.parseInt(array[1]);
				int rating     = Integer.parseInt(array[2]);
				Map<Integer, Integer> placeRatingMap =
					trainUserPlaceRatingMap.get(userID);
				if (placeRatingMap == null) {
					placeRatingMap = new HashMap<Integer, Integer>();
					trainUserPlaceRatingMap.put(userID, placeRatingMap);
				}
				if (placeRatingMap.containsKey(placeID)) {
					reader.close(); reader=null;
					throw new CheckinException(
						"UserID[%s], placeID[%s] has repeated.",userID,placeID);
				} else {
					placeRatingMap.put(placeID, rating);
				}
			}
			reader.close();
			
			int[][] trainUserPlaceArr       = new int[userNum][];
			int[][] trainUserPlaceRatingArr = new int[userNum][];
			for (int userID = 0; userID < userNum; userID ++) {
				Map<Integer, Integer> trainPlaceRatingMap =
						trainUserPlaceRatingMap.get(userID);
				if (trainPlaceRatingMap != null) {
					int[] placeIDArr = Utils.getSortedKeys(
										trainPlaceRatingMap.keySet());
					int[] ratingArr  = new int[placeIDArr.length];
					for (int i = 0; i < placeIDArr.length; i ++) {
						ratingArr[i] = trainPlaceRatingMap.get(placeIDArr[i]);
					}
					trainUserPlaceArr[userID]       = placeIDArr;
					trainUserPlaceRatingArr[userID] = ratingArr;
				} else {
					trainUserPlaceArr[userID]       = null;
					trainUserPlaceRatingArr[userID] = null;
				}
			}
			
			// augmented rating map
			Map<Integer, Map<Integer, Float>> augUserPlaceRatingMap =
								new HashMap<Integer, Map<Integer, Float>>();
			Map<Integer, double[]> userHomeLatLngMap =
								Utils.loadLocationDic(userHomeLatLngFile,
										CheckinConstants.DELIMITER, 0, 1, 2);
			Map<Integer, double[]> locationLatLngMap = Utils.loadLocationDic(
								locationLatLngFile, CheckinConstants.DELIMITER,
										0, 1, 2);
			int[][] userFriends = loadUserFriends(directFriendFile,
								locationFriendFile, neighboringFriendFile,
								userNum);
			int origTrainNum  = 0;
			int augTrainNum   = 0;
			double maxGeoProb = Utils.calPowerLawProb(userHomePowerLawParams[0],
							userHomePowerLawParams[1], zeroDistanceDefaultValue,
									zeroDistanceDefaultValue);
			if (maxGeoProb <= 0) {
				throw new CheckinException("maxGeoProb[%s] <=0", maxGeoProb);
			} else {
				System.out.println("maxGeoProb=" + maxGeoProb);
			}
			for (int userID = 0; userID < userNum; userID ++) {
				if (userID % 10000 == 0) System.out.println("UserID=" + userID);
				int[] friendArr = userFriends[userID];
				Map<Integer, Integer> trainPlaceRatingMap =
					trainUserPlaceRatingMap.get(userID);
				Map<Integer, Map<Integer, Integer>> placeFriendRatingMap =
					new HashMap<Integer, Map<Integer, Integer>>();
				for (int friendID : friendArr) {
					Map<Integer, Integer> friendPlaceRatingMap
						= trainUserPlaceRatingMap.get(friendID);
					//// new added
					if (friendPlaceRatingMap == null) continue;
					for (int placeID : friendPlaceRatingMap.keySet()) {
						if (trainPlaceRatingMap != null &&
							trainPlaceRatingMap.containsKey(placeID)) {
							continue;
						}
						Map<Integer, Integer> friendRatingMap =
							placeFriendRatingMap.get(placeID);
						if (friendRatingMap == null) {
							friendRatingMap = new HashMap<Integer, Integer>();
							placeFriendRatingMap.put(placeID, friendRatingMap);
						}
						friendRatingMap.put(friendID,friendPlaceRatingMap.get(placeID));
					}
				}
				Map<Integer,Float>augPlaceRatingMap=new HashMap<Integer,Float>();
				
				if (placeFriendRatingMap != null && 
						placeFriendRatingMap.size()!=0) {
				/////////////////////////////////////////////
				double[] homeLatLng = userHomeLatLngMap.get(userID);
				
				final Map<Integer, Float> placeRatingMap =
										new HashMap<Integer, Float>();
				for (int placeID : placeFriendRatingMap.keySet()) {
					if (augPlaceRatingMap.containsKey(placeID))
						continue;
					// geo
					double[] placeLatLng = locationLatLngMap.get(placeID);
					double distance      = Utils.calDistance(
											homeLatLng[0], homeLatLng[1],
											placeLatLng[0], placeLatLng[1]);
					// normalize geo
					double geoProb = Utils.calPowerLawProb(
							userHomePowerLawParams[0],userHomePowerLawParams[1],
								zeroDistanceDefaultValue, distance) /maxGeoProb;
					
					// user sim
					int[] userPlaceArr       = trainUserPlaceArr[userID];
					int[] userPlaceRatingArr = trainUserPlaceRatingArr[userID];
					Map<Integer, Integer> friendRatingMap =
						placeFriendRatingMap.get(placeID);
					float totalRating        = 0;
					float totalSim           = 0;
					Float maxRating          = 0f;
					float pref               = 0;
					for (int friendID : friendRatingMap.keySet()) {
						int rating    = friendRatingMap.get(friendID);
						double cosine = Utils.cosineSimilarity(
										userPlaceArr, userPlaceRatingArr,
											trainUserPlaceArr[friendID],
											trainUserPlaceRatingArr[friendID]);
					
						double sim   = userSimRatio * cosine +
										(1 - userSimRatio) * geoProb;
						totalRating += rating * sim;
						
						totalSim += sim;
						float v  = (float)(rating * sim);
						if (maxRating == null || v > maxRating) {
							maxRating = v;
							pref      = maxRating;
						}
					}
					// average rating
					//totalRating /= friendRatingMap.size() + 0.0f;
					
					// max rating
					//totalRating = maxRating * augmentRatingRatio;
					
					// pref
					totalRating = pref;
					/*if (totalRating >= ratingThreshold) {
						augPlaceRatingMap.put(placeID, totalRating);
					}*/
					placeRatingMap.put(placeID, totalRating);
				} // end placeID
				Integer[] placeIDArr = placeRatingMap.keySet().toArray(new Integer[0]);
				if (placeIDArr.length > augRatingTopNum) {
					Arrays.sort(placeIDArr, new java.util.Comparator<Integer>(){
						@Override
						public int compare(Integer placeID_1,Integer placeID_2){
							Float rating_1 = placeRatingMap.get(placeID_1);
							Float rating_2 = placeRatingMap.get(placeID_2);
							return rating_2.compareTo(rating_1);
						}
					});
				}
				int num = placeIDArr.length > augRatingTopNum ?
								augRatingTopNum : placeIDArr.length;
				for (int i = 0; i < num; i ++) {
					int placeID  = placeIDArr[i];
					float rating = placeRatingMap.get(placeID);
					augPlaceRatingMap.put(placeID, rating);
				}
				///////////////////////////////////////////////
				} // end if
				if (trainPlaceRatingMap != null) {
					for (int placeID : trainPlaceRatingMap.keySet()) {
						augPlaceRatingMap.put(placeID,
							trainPlaceRatingMap.get(placeID) + 0.0f);
					}
					origTrainNum += trainPlaceRatingMap.size();
				}
				if (augPlaceRatingMap.size() > 0) {
					augUserPlaceRatingMap.put(userID, augPlaceRatingMap);
					augTrainNum += augPlaceRatingMap.size();
				} else {
					System.err.println("No Training: UserID = " + userID);
				}
			} // end userID
			System.out.println("MeanOrigTrainNum="+ (origTrainNum+0.0)/userNum);
			System.out.println("MeanAugTrainNum ="+ (augTrainNum+0.0)/userNum);
			
			// output
			storePotCheckins(userNum, augUserPlaceRatingMap, outputFile);
		} catch (IOException e) {
			e.printStackTrace();
			throw new CheckinException(e);
		} finally {
			Utils.cleanup(reader);
		}
	}
	
	private static double[] loadUserHomePowerLawParams(File userHomeLatLngFile,
			File trainRatingFile, File locationLatLngFile, File outputPath)
										throws CheckinException {
		return Utils.loadUserHomePowerLawParams(userHomeLatLngFile,
			trainRatingFile, locationLatLngFile, zeroDistanceDefaultValue,
				outputPath, FILE_NAME_HOME_POWERLAW_PARAM);
	}

	public static int[][] loadUserFriends(File directFriendFile,
						File locationFriendFile, File neighboringFriendFile,
							int userNum) throws CheckinException {
		BufferedReader reader = null;
		try {
			File[] friendFiles = {directFriendFile,
						locationFriendFile, neighboringFriendFile};
			String line        = null;
			Map<Integer, Set<Integer>> userFriendMap =
									new HashMap<Integer, Set<Integer>>();
			for (File file : friendFiles) {
				if (file == null) continue;
				
				reader = new BufferedReader(new InputStreamReader(
							new FileInputStream(file)));
				Set<Integer> existedUserIDSet = new HashSet<Integer>();
				int totalFriendNum            = 0;
				while ((line = reader.readLine()) != null) {
					String[] array = line.split("\t");
					int userID     = Integer.parseInt(array[0]);
					if (existedUserIDSet.contains(userID)) {
						reader.close(); reader=null;
						throw new CheckinException("UserID[%s] has existed in %s.",
							userID, file.getAbsolutePath());
					} else {
						existedUserIDSet.add(userID);
					}
					Set<Integer> friendSet = userFriendMap
								.get(userID);
					if (friendSet == null) {
						friendSet = new HashSet<Integer>();
						userFriendMap.put(userID, friendSet);
					}
					// from the second element, they are friends
					for (int index = 1; index < array.length; index ++) {
						int friendID = Integer.parseInt(array[index]);
						if (! friendSet.contains(friendID)) {
							friendSet.add(friendID);
						}
					}
					totalFriendNum += array.length -1;
				}
				reader.close();
				System.out.println(String.format("%s: MeanFriendNum = %s",
					file.getName(), (totalFriendNum + 0.0)
						/ existedUserIDSet.size()));
			}
			if (userFriendMap.size() == 0) {
				throw new RuntimeException("No user friend loaded.");
			}
			if (userFriendMap.size() != userNum) {
				System.err.println(String.format(
					"UserNum=%s, UserNumWhoHaveFriends=%s",
						userNum, userFriendMap.size()));
			}
			int[][] userFriendArr = new int[userNum][];
			int totalFriendNum    = 0;
			for (int userID = 0; userID < userNum; userID ++) {
				if (userFriendMap.containsKey(userID)) {
					userFriendArr[userID] =
						Utils.getSortedKeys(
						userFriendMap.get(userID));
					totalFriendNum += userFriendArr[userID].length;
				} else {
					userFriendArr[userID] = null;
				}
			}
			System.out.println("Total: MeanFriendNum = " +
					(totalFriendNum + 0.0) / userNum);
			return userFriendArr;
		} catch (IOException e) {
			e.printStackTrace();
			throw new CheckinException(e);
		} finally {
			Utils.cleanup(reader);
		}
	}
	
	public static int[] loadItemCategory(File locationCategoryFile,int ITEM_NUM,
							int CATEGORY_NUM) throws CheckinException {
		Utils.exists(locationCategoryFile);
		
		Map<Integer, Integer> locationCatMap = Utils.loadMap(
				locationCategoryFile, CheckinConstants.DELIMITER);
		int[] categoryIndexForItem = new int[ITEM_NUM];
		for (int itemID = 0; itemID < ITEM_NUM; itemID ++) {
			if (! locationCatMap.containsKey(itemID)) {
				throw new CheckinException(
					"Cannot find category for item[%s].", itemID);
			}
			int catID = locationCatMap.get(itemID);
			if (catID >= CATEGORY_NUM) {
				throw new CheckinException(
					"ItemID[%s], categoryID[%s] >= CATEGORY_NUM[%s]",
					itemID, catID, CATEGORY_NUM);
			}
			categoryIndexForItem[itemID] = catID;
		}
		return categoryIndexForItem;
	}

	private static void storePotCheckins(int userNum,
			Map<Integer, Map<Integer, Float>> augUserPlaceRatingMap,
				File outputFile) throws CheckinException {
		BufferedWriter writer = null;
		try {
			if (! outputFile.getParentFile().exists())
				outputFile.getParentFile().mkdirs();
			writer = new BufferedWriter(new OutputStreamWriter(
						new FileOutputStream(outputFile)));
			int noTrainingUserNum = 0;
			for (int userID = 0; userID < userNum; userID ++) {
				if (! augUserPlaceRatingMap.containsKey(userID)) {
					noTrainingUserNum ++;
					continue;
				}
				Map<Integer, Float> placeRatingMap =
									augUserPlaceRatingMap.get(userID);
				int[] placeIDArr = Utils.getSortedKeys(placeRatingMap.keySet());
				for (int placeID : placeIDArr) {
					writer.write(String.format("%s\t%s\t%s", userID, placeID, 
						placeRatingMap.get(placeID)));
					writer.newLine();
				}
			}
			writer.close();
			System.out.println("NoTrainingUserNum = " + noTrainingUserNum);
		} catch (IOException e) {
			throw new CheckinException(e);
		} finally {
			Utils.cleanup(writer);
		}
	}
}
