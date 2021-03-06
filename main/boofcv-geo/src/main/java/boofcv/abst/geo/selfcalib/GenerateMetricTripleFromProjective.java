/*
 * Copyright (c) 2011-2020, Peter Abeles. All Rights Reserved.
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

package boofcv.abst.geo.selfcalib;

import boofcv.abst.geo.Estimate1ofTrifocalTensor;
import boofcv.alg.geo.MetricCameras;
import boofcv.alg.geo.MultiViewOps;
import boofcv.alg.geo.robust.ModelGeneratorViews;
import boofcv.alg.geo.selfcalib.MetricCameraTriple;
import boofcv.alg.geo.trifocal.TrifocalExtractGeometries;
import boofcv.struct.geo.AssociatedTriple;
import boofcv.struct.geo.TrifocalTensor;
import boofcv.struct.image.ImageDimension;
import georegression.struct.point.Point2D_F64;
import lombok.Getter;
import org.ddogleg.struct.FastQueue;
import org.ejml.data.DMatrixRMaj;

import java.util.ArrayList;
import java.util.List;

/**
 * Wrapper around {@link ProjectiveToMetricCameras} and {@link Estimate1ofTrifocalTensor} for use in robust model
 * fitting.
 *
 * @author Peter Abeles
 */
public class GenerateMetricTripleFromProjective implements
		ModelGeneratorViews<MetricCameraTriple, AssociatedTriple, ImageDimension>
{
	// Computes a trifocal tensor from input observations from which projective cameras are extracted
	public Estimate1ofTrifocalTensor trifocal;
	// from projective cameras computes metric cameras
	public ProjectiveToMetricCameras projectiveToMetric;
	// used to get camera matrices from the trifocal tensor
	@Getter final TrifocalExtractGeometries extractor = new TrifocalExtractGeometries();
	@Getter final TrifocalTensor tensor = new TrifocalTensor();

	// storage for camera matrices
	@Getter final DMatrixRMaj P2;
	@Getter final DMatrixRMaj P3;

	// Data structures which have been converted
	@Getter final List<List<Point2D_F64>> observationsN = new ArrayList<>();
	@Getter final FastQueue<DMatrixRMaj> projective = new FastQueue<>(()->new DMatrixRMaj(3,4));
	@Getter final FastQueue<ImageDimension> dimensions = new FastQueue<>(ImageDimension::new);
	@Getter final MetricCameras metricN = new MetricCameras();

	public GenerateMetricTripleFromProjective(Estimate1ofTrifocalTensor trifocal,
											  ProjectiveToMetricCameras projectiveToMetric) {
		this.trifocal = trifocal;
		this.projectiveToMetric = projectiveToMetric;

		dimensions.resize(3);
		projective.resize(2);
		P2 = projective.get(0);
		P3 = projective.get(1);
	}

	@Override
	public void setView(int view, ImageDimension viewInfo) {
		dimensions.get(view).setTo(viewInfo);
	}

	@Override
	public int getNumberOfViews() {
		return 3;
	}

	@Override
	public boolean generate(List<AssociatedTriple> observationTriple, MetricCameraTriple output) {
		// Get trifocal tensor
		if( !trifocal.process(observationTriple,tensor) )
			return false;

		// Get camera matrices from trifocal
		extractor.setTensor(tensor);
		extractor.extractCamera(P2,P3);

		MultiViewOps.splits3Lists(observationTriple, observationsN);

		if( !projectiveToMetric.process(dimensions.toList(),projective.toList(), observationsN,metricN) )
			return false;

		// Converts the output
		output.view_1_to_2.set(metricN.motion_1_to_k.get(0));
		output.view_1_to_3.set(metricN.motion_1_to_k.get(1));

		output.view1.set(metricN.intrinsics.get(0));
		output.view2.set(metricN.intrinsics.get(1));
		output.view3.set(metricN.intrinsics.get(2));

		return true;
	}

	@Override
	public int getMinimumPoints() {
		return trifocal.getMinimumPoints();
	}
}
