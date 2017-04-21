package com.uncc.checkin.fw;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
//import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.uncc.checkin.CheckinConstants;
import com.uncc.checkin.CheckinException;
import com.uncc.checkin.ParamManager;
import com.uncc.checkin.util.Utils;

public abstract class BasicRatingMF extends GeneralMF {
	protected float ERROR_THRESHOLD      = 1.0e-5f;
	protected boolean MINUS_MEAN_RATING  = false;
	
	protected float[] ratingsInItemOrder = null;
	protected float[] ratingsInUserOrder = null;
	protected int[] userNumInItemOrder    = null;
	protected int[] userIndexInItemOrder  = null;
	protected int[] itemNumInUserOrder    = null;
	protected int[] itemIndexInUserOrder  = null;
	
	protected double meanRating = 0;
	
	protected int FEATURE_NUM       = -1;
	protected int ITEM_NUM          = -1;
	protected int USER_NUM          = -1;
	protected int RATING_RECORD_NUM = -1;
	protected int MAX_ITER_NUM      = -1;
	
	protected abstract void initModel(ParamManager paramManager) throws CheckinException;
	protected abstract void update1Iter();
	protected abstract double calLoss() throws CheckinException;
	protected abstract double predict(int userId, int itemId);
	protected abstract void storeLatentVariables();
	
	public void estimate() throws CheckinException {
		if (! resultFile.getParentFile().exists())
			resultFile.getParentFile().mkdirs();
		
		final File testPredictedFile = new File(outputPath, 
				CheckinConstants.STRING_PREDECT_RATING);
		if (testPredictedFile.exists()) testPredictedFile.delete();
		
		BufferedWriter resultWriter = null;
		
		try {
			resultWriter = new BufferedWriter(new OutputStreamWriter(
							new FileOutputStream(resultFile, true)));
			getParams().store(resultWriter, String.format("%s Parameters",
												getMethodName()));
	
			double oldLoss  = calLoss();
			double lossDiff = Double.MAX_VALUE;
			int iteration   = 0;
			Utils.writeAndPrint(resultWriter, String.format(
					"Train RMSE = %s, Test RMSE = %s", predictTrain(),
						(testFile == null ? "--" :
							predict(testFile, testPredictedFile))), true);
			do {
	
				update1Iter();
				
				double loss = calLoss();
				lossDiff    = Math.abs((oldLoss - loss) / oldLoss);
				oldLoss     = loss;
				iteration ++;
				
				Utils.writeAndPrint(resultWriter, String.format(
					"[%s][Iteration = %s] Loss = %s, LossDiff = %s, TrainRMSE = %s, TestRMSE = %s",
					getMethodName(), iteration, loss, lossDiff, predictTrain(),
					(testFile == null ? "--" : predict(testFile, testPredictedFile))),
					true);
				
				/*double[] tauAndnDCG = Evaluator.calculateTauAndnDCG(testPredictedFile);
				Utils.writeAndPrint(resultWriter, String.format(
						"Tau=%s, nDCG=%s", tauAndnDCG[0],
						tauAndnDCG[1]), true);*/
				
				storeLatentVariables();
			} while ((iteration < 3 || lossDiff > ERROR_THRESHOLD) && 
					iteration < MAX_ITER_NUM);
			
			resultWriter.close();
		} catch (IOException e) {
			throw new CheckinException(e);
		} finally {
			Utils.cleanup(resultWriter);
		}
	}
	
	protected double calRatingDiff(double predRating, double groundtruthRating) {
		return predRating - groundtruthRating;
	}
	
	@Override
	protected Properties getParams() {
		Properties props = new Properties();
		props.put(CheckinConstants.STRING_ITEM_NUM,			String.valueOf(ITEM_NUM));
		props.put(CheckinConstants.STRING_USER_NUM,			String.valueOf(USER_NUM));
		props.put(CheckinConstants.STRING_FEATURE_NUM,		String.valueOf(FEATURE_NUM));
		props.put(CheckinConstants.STRING_RATING_RECORD_NUM,	String.valueOf(RATING_RECORD_NUM));
		props.put(CheckinConstants.STRING_MEAN_TRAIN_RATING,	String.valueOf(meanRating));
		props.put(CheckinConstants.STRING_MAX_ITERATION_NUM,	String.valueOf(MAX_ITER_NUM));
		props.put(CheckinConstants.STRING_CONVERGE_THRESHOLD, 	String.valueOf(ERROR_THRESHOLD));
		props.put(CheckinConstants.STRING_TRAIN_FILE,	trainFile.getAbsolutePath());
		props.put(CheckinConstants.STRING_TEST_FILE,	testFile.getAbsolutePath());
		props.put(CheckinConstants.STRING_OUTPUT_PATH,	outputPath.getAbsolutePath());
		props.put(CheckinConstants.STRING_PROGRAM_NAME, getMethodName());
		props.put("MINUS_MEAN_RATING",	String.valueOf(MINUS_MEAN_RATING));
		
		return props;
	}
	
	protected double predictTrain() throws CheckinException {
		return predict(trainFile, null);
	}
	
	/*
	 * The line format is as follows:
	 * 	UserId '\t' ItemId '\t' RawRating
	 */
	protected double predict(File inputFile, File outputFile)
										throws CheckinException {
		BufferedReader reader = null;
		BufferedWriter writer = null;
		try {
			reader = new BufferedReader(new InputStreamReader(
						new FileInputStream(inputFile)));
			if (outputFile != null) {
				writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(outputFile)));
			}
			String line     = null;
			double rmse     = 0;
			int recordNum   = 0;
			while ((line = reader.readLine()) != null) {
				if ("".equals(line = line.trim())) continue;
				String array[] = line.split("\t");
				if (array.length != 3) {
					reader.close();
					writer.close();
					throw new CheckinException("Failed to parse line : " + line);
				}
				int userId = Integer.parseInt(array[0]);
				int itemId = Integer.parseInt(array[1]);
				final float groundtruthRating = (float)(Float.parseFloat(array[2]));
				if (itemId < 0 || itemId >= ITEM_NUM) {
					reader.close();
					writer.close();
					throw new CheckinException("Cannot find the item %s.",
						itemId);
				}
				double predRating = predict(userId, itemId);
				double diff       = calRatingDiff(predRating, groundtruthRating);
				rmse       += diff * diff;
				recordNum ++;
				if (writer != null) {
					writer.write(String.format("%s\t%s\t%s\t%s", userId,
							itemId, groundtruthRating, predRating));
					writer.newLine();
				}
			}
			reader.close();
			if (writer != null) writer.close();
			return Math.sqrt(rmse / recordNum);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader);
			Utils.cleanup(writer);
		}

		return -1;
	}
	
	protected void init(ParamManager paramManager) throws CheckinException {
		if (paramManager == null)
			throw new CheckinException("No parameters specified.");
	
		ERROR_THRESHOLD = paramManager.getConvergeThreshold();
		FEATURE_NUM     = paramManager.getFeatureNum();
		ITEM_NUM        = paramManager.getItemNum();
		USER_NUM        = paramManager.getUserNum();
		MAX_ITER_NUM    = paramManager.getMaxInterationNum();
	
		outputPath = paramManager.getOutputPath();
		if (outputPath == null) throw new CheckinException("No output path specified.");
		
		outputPath = new File(new File(outputPath, getMethodName()),
							String.valueOf(FEATURE_NUM));
		trainFile  = paramManager.getTrainFile();
		testFile   = paramManager.getTestFile();
		resultFile = new File(outputPath, "Result");
	
		parseTrainRatingFile(paramManager.getTrainFile());
		
		initModel(paramManager);
	
		storeParams(getMethodName() + " Parameters");
	}
	
	/*
	 * The line format is as follows:
	 * 	UserId '\t' ItemId '\t' RawRating
	 * Before calling this method, must call parseTrainClusterFile()
	 */
	protected void parseTrainRatingFile(File trainRatingFile) throws CheckinException {
		BufferedReader reader = null;
		BufferedWriter writer = null;
		try {
			System.out.println(String.format("[Info] FORCE_CHECK = %s", FORCE_CHECK));

			// calculate mean rating
			if (MINUS_MEAN_RATING) {
				meanRating = calculateMeanRating(trainRatingFile);
			}
			System.out.println("[Info] MeanRating = " + meanRating);

			File tmpTrainRatingFile = new File(outputPath,
					"_tmp_" +  trainRatingFile.getName());
			if (MINUS_MEAN_RATING) {
				tmpTrainRatingFile.delete();
				if (!tmpTrainRatingFile.getParentFile().exists())
					tmpTrainRatingFile.getParentFile().mkdirs();
			}

			// load rating in item order
			Map<Integer, Map<Integer, Float>> itemUserRatingMap =
								new HashMap<Integer, Map<Integer, Float>>();
			String line = null;
			reader      = new BufferedReader(new InputStreamReader(
							new FileInputStream(trainRatingFile)));
			writer      = MINUS_MEAN_RATING ?
							new BufferedWriter(new OutputStreamWriter(
								new FileOutputStream(tmpTrainRatingFile))):null;
			int minItemId     = Integer.MAX_VALUE;
			int maxItemId     = -1;
			RATING_RECORD_NUM = 0;
			while ((line = reader.readLine()) != null) {
				if ("".equals(line = line.trim())) continue;
				String[] array = line.split("\t");
				if (array.length != 3) {
					reader.close(); reader = null;
					throw new CheckinException("Failed to parse TrainRatingFile: %s", line);
				}
				int userId   = Integer.parseInt(array[0]);
				int itemId   = Integer.parseInt(array[1]);
				float rating = Float.parseFloat(array[2]);
				if (MINUS_MEAN_RATING) rating -= meanRating;
				Map<Integer, Float> userRatingMap=itemUserRatingMap.get(itemId);
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
				if (MINUS_MEAN_RATING) {
					writer.write(userId + "\t" + itemId + "\t" + rating);
					writer.newLine();
				}
			}
			reader.close();
			if (MINUS_MEAN_RATING) writer.close();
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
			if (MINUS_MEAN_RATING) {
				reader = new BufferedReader(new InputStreamReader(
							new FileInputStream(tmpTrainRatingFile)));
			} else {
				reader = new BufferedReader(new InputStreamReader(
							new FileInputStream(trainRatingFile)));
			}
			Map<Integer, Map<Integer, Float>> userItemRatingMap =
							new HashMap<Integer, Map<Integer, Float>>();
			int minUserId = Integer.MAX_VALUE;
			int maxUserId = -1;
			while ((line = reader.readLine()) != null) {
				if ("".equals(line = line.trim())) continue;
				String[] array = line.split("\t");
				if (array.length != 3) {
					reader.close(); reader = null;
					throw new CheckinException("Failed to parse TrainRatingFile: %s", line);
				}
				int userId    = Integer.parseInt(array[0]);
				int itemId    = Integer.parseInt(array[1]);
				float rating  = Float.parseFloat(array[2]);
				Map<Integer, Float> itemRatingMap=userItemRatingMap.get(userId);
				if (itemRatingMap == null) {
					itemRatingMap = new HashMap<Integer, Float>();
					itemRatingMap.put(itemId, rating);
					userItemRatingMap.put(userId, itemRatingMap);
				} else {
					if (itemRatingMap.containsKey(itemId)) {
						throw new CheckinException(
								"Item %s has existed. Line:%s", itemId, line);
					}
					itemRatingMap.put(itemId, rating);
				}
				if (userId < minUserId) minUserId = userId;
				if (userId > maxUserId) maxUserId = userId;
			} // end
			reader.close();
			if (MINUS_MEAN_RATING) tmpTrainRatingFile.delete();
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
				ratingsInUserOrder   = new float[RATING_RECORD_NUM];
				itemIndexInUserOrder = new int[RATING_RECORD_NUM]; 
				itemNumInUserOrder   = new int[USER_NUM];
				int[] userIds = Utils.getSortedKeys(userItemRatingMap.keySet());
				int index     = 0;
				for (int userId : userIds) {
					Map<Integer, Float> itemRatingMap =
									userItemRatingMap.get(userId);
					int[] ItemIds = Utils.getSortedKeys(itemRatingMap.keySet());
					for (int itemId : ItemIds) {
						ratingsInUserOrder[index]   = itemRatingMap.get(itemId);
						itemIndexInUserOrder[index] = itemId;
						index ++;
					}
					itemNumInUserOrder[userId] = itemRatingMap.size();
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader);
			Utils.cleanup(writer);
		}
	}
	
	public static double calculateMeanRating(File inputFile)
										throws CheckinException{
		if (! Utils.exists(inputFile)) {
			throw new CheckinException("Input file does not exist.[%s]",
				inputFile == null ? null : inputFile.getAbsolutePath());
		}
		BufferedReader reader = null;
		int recordNum         = 0;
		double meanRating     = 0;
		try {
			reader      = new BufferedReader(new InputStreamReader(
					new FileInputStream(inputFile)));
			String line = null;
			while ((line = reader.readLine()) != null) {
				if ("".equals(line = line.trim())) continue;
				String[] array = line.split("\t");
				if (array.length != 3) {
					reader.close();
					reader = null;
					throw new CheckinException("Failed to parse TrainRatingFile: %s", line);
				}
				meanRating += Float.parseFloat(array[2]);
				recordNum ++;
			}
			meanRating = recordNum == 0 ? 0 : (meanRating / (recordNum + 0.0));
			return meanRating;
		} catch (IOException e) {
			e.printStackTrace();
			throw new CheckinException(e.toString());
		} finally{
			Utils.cleanup(reader);
		}
	}
	
	public static double[][] inverse(double[][] matrix) {
		return inverse(new BlockRealMatrix(matrix)).getData();
	}

	public static RealMatrix inverse(RealMatrix matrix) {
		/*int splitIndex = matrix.getColumnDimension() > matrix.getRowDimension() ?
					matrix.getRowDimension() : matrix.getColumnDimension();
		splitIndex = splitIndex == 2 ? 0 : splitIndex / 2;
		return MatrixUtils.blockInverse(matrix, splitIndex);*/
		
		return  new LUDecomposition(matrix).getSolver().getInverse();
	}
}
