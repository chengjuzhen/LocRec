package com.uncc.checkin.util;

import java.util.Map;
import java.util.HashMap;
import java.util.Properties;
import com.uncc.checkin.CheckinException;

/*
 * [Optional]
 * 		-WM	<int> (0: MULT, 1: SQRT, 2: LOG)
 *		-Alpha <float>
 *		-Epsilon <double>
 *		-Delta <float>
 */
public class WeightSchema {
	private static final String STRING_WMETHOD  = "WM";
	private static final String STRING_ALPHA    = "Alpha";
	private static final String STRING_EPSILON  = "Epsilon";
	private static final String STRING_DELTA    = "Delta";
	
	public double[] weightsInItemOrder         = null;
	public double[] weightsInUserOrder         = null;
	public double[] weightsMinusOneInItemOrder = null;
	public double[] weightsMinusOneInUserOrder = null;
	
	private double alpha   = 1f;
	private double epsilon = 1.0e10;
	private double delta   = 10;
	
	public enum WMethod {
		MULT,
		SQRT,
		LOG;
	}
	
	private WMethod WM = WMethod.MULT;
	
	public WeightSchema(Properties args) throws CheckinException {
		parseParams(args);
	}
	
	public WeightSchema(Properties args, int USER_NUM, int ITEM_NUM,
			float[] ratingsInItemOrder, int[] userNumInItemOrder,
				int[] userIndexInItemOrder, float[] ratingsInUserOrder,
				int[] itemNumInUserOrder, int[] itemIndexInUserOrder)
										throws CheckinException {
		parseParams(args);
		initWeight(USER_NUM, ITEM_NUM, ratingsInItemOrder,
				userNumInItemOrder, userIndexInItemOrder, 
				ratingsInUserOrder, itemNumInUserOrder,
				itemIndexInUserOrder);
	}
	
	public Properties getParams() {
		Properties props = new Properties();
		props.put(STRING_DELTA,		String.valueOf(delta));
		props.put(STRING_ALPHA,		String.valueOf(alpha));
		props.put(STRING_EPSILON,	String.valueOf(epsilon));
		props.put(STRING_WMETHOD,	String.valueOf(WM));
		return props;
	}
	
	private void parseParams(Properties props) throws CheckinException {
		if (props == null) return;

		String sprop = props.getProperty(STRING_ALPHA);
		if (sprop != null) {
			alpha = Double.parseDouble(sprop);
		}
		
		sprop = props.getProperty(STRING_DELTA);
		if (sprop != null) {
			delta = Double.parseDouble(sprop);
		}
		
		sprop = props.getProperty(STRING_EPSILON);
		if (sprop != null) {
			epsilon = Double.parseDouble(sprop);
		}
		
		sprop  = props.getProperty(STRING_WMETHOD);
		if (sprop != null) {
			int wm = Integer.parseInt(sprop);
			if (wm == WMethod.MULT.ordinal()) {
				WM = WMethod.MULT;
			} else
			if (wm == WMethod.SQRT.ordinal()) {
				WM = WMethod.SQRT;
			} else
			if (wm == WMethod.LOG.ordinal()) {
				WM = WMethod.LOG;
			} else {
				throw new CheckinException("Unknown weight method: " + sprop);
			}
		}
	}

	private void initWeight(int USER_NUM, int ITEM_NUM,
					float[] ratingsInItemOrder, int[] userNumInItemOrder,
					int[] userIndexInItemOrder, float[] ratingsInUserOrder,
					int[] itemNumInUserOrder, int[] itemIndexInUserOrder)
											throws CheckinException {
		int recordIndex            = 0;
		weightsInUserOrder         = new double[ratingsInUserOrder.length];
		weightsMinusOneInUserOrder = new double[ratingsInUserOrder.length];
		Map<Integer, Map<Integer, Integer>> userItemIndexMap =
				new HashMap<Integer, Map<Integer, Integer>>();
		for (int userId = 0; userId < USER_NUM; userId ++) {
			int itemNum = itemNumInUserOrder[userId];
			for (int index = 0; index < itemNum; index ++) {
				int itemId = itemIndexInUserOrder[recordIndex];
				double v = calWeightMinusOne(
							ratingsInUserOrder[recordIndex]);
				double w = v + 1;
				weightsMinusOneInUserOrder[recordIndex] = v;
				weightsInUserOrder[recordIndex]         = w;
				
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
		weightsInItemOrder         = new double[ratingsInItemOrder.length];
		weightsMinusOneInItemOrder = new double[ratingsInItemOrder.length];
		for (int itemId = 0; itemId < ITEM_NUM; itemId ++) {
			int userNum = userNumInItemOrder[itemId];
			for (int index = 0; index< userNum; index ++) {
				int userId = userIndexInItemOrder[recordIndex];
				Map<Integer, Integer> itemIndexMap =
						userItemIndexMap.get(userId);
				if (itemIndexMap == null ||
					! itemIndexMap.containsKey(itemId)) {
					throw new CheckinException("Cannot find userId[%s], ItemId[%s]",
						userId, itemId);
				}
				int mappedIndex = itemIndexMap.get(itemId);
				weightsInItemOrder[recordIndex]         =
					weightsInUserOrder[mappedIndex];
				weightsMinusOneInItemOrder[recordIndex] =
					weightsMinusOneInUserOrder[mappedIndex];
				recordIndex ++;
			}
		}
		if (recordIndex != ratingsInUserOrder.length) {
			throw new CheckinException("recordIndex[%s] != ratingsInItemOrder.length[%s]",
				recordIndex, ratingsInItemOrder.length);
		}
		/*for (int i = ratingsInItemOrder.length - 1; i >= 0; i --) {
			double v = getConvertedWeightMinusOne(ratingsInItemOrder[i], epsilon, alpha);
			weightsMinusOneInItemOrder[i] = v;
			weightsInItemOrder[i]         = v + 1;
		}*/
	}
	
	public double calWeightMinusOne(double rating) {
		switch (WM) {
			case LOG:
				return alpha * Math.log(1 + rating * epsilon);
			case MULT:
				return alpha * rating;
			case SQRT:
				return Math.sqrt(1 + delta * rating) - 1;
		}
		
		throw new RuntimeException("Inner Error!"); 
	}
}
