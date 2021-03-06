/*
 * Copyright (c) 2011-2019, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://boofcv.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boofcv.alg.filter.binary;

import boofcv.struct.ConfigLength;
import boofcv.struct.image.GrayF32;

class TestThresholdNick_MT extends GenericInputToBinaryCompare<GrayF32> {

	TestThresholdNick_MT() {
		ThresholdNick_MT target = new ThresholdNick_MT(ConfigLength.fixed(12), -0.2f,true);
		ThresholdNick reference = new ThresholdNick(ConfigLength.fixed(12), -0.2f,true);

		initialize(target,reference);
	}
}

