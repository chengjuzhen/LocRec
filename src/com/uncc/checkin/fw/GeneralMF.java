package com.uncc.checkin.fw;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Properties;

import com.uncc.checkin.CheckinConstants;
import com.uncc.checkin.CheckinException;
import com.uncc.checkin.util.Utils;

public abstract class GeneralMF {
	
	protected boolean FORCE_CHECK   = true;
	protected float ERROR_THRESHOLD = 1.0e-5f;
	protected File outputPath       = null;
	protected File trainFile        = null;
	protected File testFile         = null;
	protected File resultFile       = null;
	
	protected abstract String getMethodName();
	protected abstract Properties getParams();
	
	public void storeParams(String comment) {
		Properties params = getParams();
		params.list(System.out);
		try {
			if (!outputPath.exists()) {
				outputPath.mkdirs();
			}
			params.store(new FileWriter(new File(outputPath,
					CheckinConstants.STRING_PARAM)), comment);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	// array is sorted in increasing order
	public static boolean exists(int[] array, int startIndex, int endIndex, int target) {
		if (startIndex > endIndex) {
			/*throw new RuntimeException(String.format(
				"ArrayLen=%s, StartIndex=%s, EndIndex=%s",
				array.length, startIndex, endIndex));*/
			return false;
		} else
		if (startIndex == endIndex) {
			return array[startIndex] == target;
		} else
		if (target < array[startIndex] || target > array[endIndex]) {
			return false;
		} else
		if (target == array[startIndex] || target == array[endIndex]) {
			return true;
		}
		int middle = (startIndex + endIndex) / 2;
		if (target < array[middle]) {
			return exists(array, startIndex + 1, middle - 1, target);
		} else
		if (target > array[middle]) {
			return exists(array, middle + 1, endIndex - 1, target);
		} else {
			return true;
		}
	}
	
	public static double[] initFeature(int len, File featureFile)
										throws CheckinException {
		double feature[] = new double[len];
		if (featureFile == null) {
			feature = new double[len];
			for (int i = 0; i < len; i ++) {
				feature[i] = Math.random() * 0.1;
				//feature[i] = new java.util.Random().nextGaussian() * 0.1;
			}
		} else {
			System.out.println("[Info] Loading feature from " +
					featureFile.getAbsolutePath());
			feature = Utils.convert2DTo1D(Utils.load2DoubleArray(
						featureFile, CheckinConstants.DELIMITER));
		if (feature.length != len) {
			throw new CheckinException("Failed to load feature: dimension = %s which should be %s.",
					feature.length, len);
			}
		}
		return feature;
	}
}
