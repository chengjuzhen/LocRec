package com.uncc.checkin;

import java.io.IOException;

import com.lhy.tool.ToolException;
import com.uncc.checkin.amf.ASMF;
import com.uncc.checkin.amf.ARMF;
import com.uncc.checkin.rank.RRFM;
import com.uncc.checkin.eval.MFEval;
import com.uncc.checkin.eval.ASMFEval;

public class Main {

	public static void main(String[] args) throws IOException,
						CheckinException, ToolException {
		ParamManager paramManager = CheckinConstants.CONFIG_MANAGER;
		String modelName          = paramManager.getModelName();
		if (CheckinConstants.STRING_MODEL_RRFM.equals(modelName)) {
			new RRFM(CheckinConstants.CONFIG_MANAGER).estimate();
			new MFEval(paramManager);
		} else
		if (ASMF.METHOD_NAME.equals(modelName)) {
			new ASMF(CheckinConstants.CONFIG_MANAGER).estimate();
			new ASMFEval(paramManager);
		} else
		if (ARMF.METHOD_NAME.equals(modelName)) {
			new ARMF(CheckinConstants.CONFIG_MANAGER).estimate();
			new ASMFEval(paramManager);
		} else {
			throw new CheckinException("Unknown model: " + modelName);
		}
	}
}
