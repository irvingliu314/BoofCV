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

package boofcv.alg.sfm.structure2;

import boofcv.alg.geo.MultiViewOps;
import boofcv.alg.geo.PerspectiveOps;
import boofcv.alg.geo.selfcalib.TwoViewToCalibratingHomography;
import boofcv.alg.sfm.structure2.PairwiseImageGraph2.Motion;
import boofcv.alg.sfm.structure2.PairwiseImageGraph2.View;
import boofcv.struct.calib.CameraPinhole;
import boofcv.struct.geo.AssociatedPair;
import boofcv.struct.geo.AssociatedTriple;
import georegression.struct.se.Se3_F64;
import lombok.Getter;
import lombok.Setter;
import org.ddogleg.struct.FastQueue;
import org.ddogleg.struct.VerbosePrint;
import org.ejml.data.DMatrixRMaj;

import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * TODO REWRITE
 *
 * Expands an existing metric scene to include a new view. At least two neighbors with a known metric world pose
 * and BOOOASDASD
 *
 * <ol>
 *     <li>Input: A seed view and the known graph</li>
 *     <li>Selects two other views with known camera matrices</li>
 *     <li>Finds features in common with all three views</li>
*      <li>Trifocal tensor and RANSAC to find the unknown seed camera matrix</li>
 *     <li>Bundle Adjustment to refine estimate</li>
 *     <li>Makes the new camera matrix compatible with the existing ones</li>
 * </ol>
 *
 * Previously computed 3D scene points are not used in an effort to avoid propagating errors. The trifocal tensor
 * computed independently of previous calculations won't be influenced by previous errors. Past errors only influence
 * the current projective estimate when making the two scenes compatible.
 *
 * @author Peter Abeles
 */
public class MetricExpandByOneView implements VerbosePrint {

	// Finds the calibrating homography when metric parameters are known for two views
	TwoViewToCalibratingHomography projectiveHomography = new TwoViewToCalibratingHomography();

	// Reference to the working scene graph
	SceneWorkingGraph workGraph;

	/** Common algorithms for reconstructing the projective scene */
	@Getter	@Setter PairwiseGraphUtils utils = new PairwiseGraphUtils(new ConfigProjectiveReconstruction());

	// If not null then print debugging information
	PrintStream verbose;

	//------------------------- Local work space

	// Storage fort he two selected connections with known cameras
	List<Motion> connections = new ArrayList<>();

	// Fundamental matrix between view-2 and view-3 in triplet
	DMatrixRMaj F21 = new DMatrixRMaj(3,3);
	// Storage for intrinsic camera matrices in view-2 and view-3
	DMatrixRMaj K1 = new DMatrixRMaj(3,3);
	DMatrixRMaj K2 = new DMatrixRMaj(3,3);
	FastQueue<AssociatedPair> pairs = new FastQueue<>(AssociatedPair::new);


	// candidates for being used as known connections
	List<Motion> validCandidates = new ArrayList<>();

	/**
	 * Attempts to estimate the camera model in the global projective space for the specified view
	 *
	 * @param db (Input) image data base
	 * @param workGraph (Input) scene graph
	 * @param target (Input) The view that needs its projective camera estimated and the graph is being expanded into
	 * @param foundPinhole (output) Intrinsic parameters for the target
	 * @param foundWorldToView (output) Transform from world to view coordinate systems for target
	 * @return true if successful and the camera matrix computed
	 */
	public boolean process( LookupSimilarImages db ,
							SceneWorkingGraph workGraph ,
							View target ,
							CameraPinhole foundPinhole,
							Se3_F64 foundWorldToView)
	{
		this.workGraph = workGraph;
		this.utils.db = db;

		// Select two known connected Views
		if( !selectTwoConnections(target,connections) ) {
			if( verbose != null ) {
				verbose.println( "Failed to expand because two connections couldn't be found. valid.size=" +
						validCandidates.size());
				for (int i = 0; i < validCandidates.size(); i++) {
					verbose.println("   valid view.id='"+validCandidates.get(i).other(target).id+"'");
				}
			}
			return false;
		}

		// Find features which are common between all three views
		utils.seed = connections.get(0).other(target);
		utils.viewB = connections.get(1).other(target);
		utils.viewC = target; // easier if target is viewC when doing metric elevation
		utils.createThreeViewLookUpTables();
		utils.findCommonFeatures();

		if( verbose != null ) {
			verbose.println( "Expanding to view='"+target.id+"' using views ( '"+utils.viewB.id+"' , '"+utils.viewC.id+
					"') common="+utils.commonIdx.size+" valid.size="+validCandidates.size());
		}

		// Estimate trifocal tensor using three view observations
		utils.createTripleFromCommon();
		if( !utils.estimateProjectiveCamerasRobustly() )
			return false;

		// Using known camera information elevate to a metric scene
		if (!computeCalibratingHomography())
			return false;

		// TODO transform from projective to metric cameras

		// TODO compute 3D location of points by triangulation
		// TODO reject if excessively large errors

		// TODO refine everything with SBA


		// TODO save results into the work graph?

//		// Improve the fit using bundle adjustment. This will reduce the rate at which errors are built up since
//		// It's important to optimize in the local frame since numbers involved will not be too large or small
//		// NOTE: Still might not be a bad idea to adjust the scale of everything first
//		// The lines below convert the known camera frames from global into local frame
//		CommonOps_DDRM.invert(localToGlobal,globalToLocal);
//		CommonOps_DDRM.mult(workGraph.lookupView(utils.viewB.id).projective,globalToLocal,utils.P2);
//		CommonOps_DDRM.mult(workGraph.lookupView(utils.viewC.id).projective,globalToLocal,utils.P3);
//
//		// fix cameras P2 and P3 and let everything else float
//		utils.initializeSbaSceneThreeView(false);
//		utils.initializeSbaObservationsThreeView();
//		utils.refineWithBundleAdjustment();
//
//		// Convert the refined results into global projective frame
//		CommonOps_DDRM.mult(utils.structure.getViews().get(0).worldToView,localToGlobal,cameraMatrix);

		return true;
	}

	/**
	 * Computes the transform needed to go from one projective space into another
	 */
	boolean computeCalibratingHomography() {

		// convert everything in to the correct data format
		MultiViewOps.projectiveToFundamental(utils.P2, F21);
		projectiveHomography.initialize(F21,utils.P2);

		PerspectiveOps.pinholeToMatrix(workGraph.lookupView(utils.viewB.id).pinhole, K1);
		PerspectiveOps.pinholeToMatrix(workGraph.lookupView(utils.viewC.id).pinhole, K2);

		List<AssociatedTriple> triples = utils.ransac.getMatchSet();
		pairs.resize(triples.size());
		for (int idx = 0; idx < triples.size(); idx++) {
			AssociatedTriple a = triples.get(idx);
			pairs.get(idx).set(a.p1,a.p2);
		}

		return projectiveHomography.process(K1, K2,pairs.toList());
	}

	/**
	 * Selects two views which are connected to the target by maximizing a score function. The two selected
	 * views must have 3D information, be connected to each other, and have a known camera matrix. These three views
	 * will then be used to estimate a trifocal tensor
	 *
	 * @param target (input) A view
	 * @param connections (output) the two selected connected views to the target
	 * @return true if successful or false if it failed
	 */
	public boolean selectTwoConnections( View target , List<Motion> connections )
	{
		connections.clear();

		// Create a list of connections in the target that can be used
		createListOfValid(target, validCandidates);

		double bestScore = 0.0;
		for (int connectionCnt = 0; connectionCnt < validCandidates.size(); connectionCnt++) {
			Motion connectB = validCandidates.get(connectionCnt);
			Motion connectC = findBestCommon(target,connectB, validCandidates);
			if( connectC == null )
				continue; // no common connection could be found

			double score = utils.scoreMotion.score(connectB) + utils.scoreMotion.score(connectC);
			if( score > bestScore ) {
				bestScore = score;
				connections.clear();
				connections.add(connectB);
				connections.add(connectC);
			}
		}

		return !connections.isEmpty();
	}

	/**
	 * Finds all the connections from the target view which are 3D and have known other views
	 * @param target (input)
	 * @param validConnections (output)
	 */
	void createListOfValid(View target, List<Motion> validConnections) {
		validConnections.clear();
		for (int connectionIdx = 0; connectionIdx < target.connections.size; connectionIdx++) {
			Motion connectB = target.connections.get(connectionIdx);
			if( !connectB.is3D || !workGraph.isKnown(connectB.other(target)))
				continue;
			validConnections.add(connectB);
		}
	}

	/**
	 * Selects the view C which has the best connection from A to C and B to C. Best is defined using the
	 * scoring function and being 3D.
	 *
	 * @param viewA (input) The root node all motions must connect to
	 * @param connAB (input) A connection from view A to view B
	 * @param validConnections (input) List of connections that are known to be valid potential solutions
	 * @return The selected common view. null if none could be found
	 */
	public Motion findBestCommon( View viewA, Motion connAB , List<Motion> validConnections)
	{
		double bestScore = 0.0;
		Motion bestConnection = null;

		View viewB = connAB.other(viewA);

		for (int connIdx = 0; connIdx < validConnections.size(); connIdx++) {
			Motion connAC = validConnections.get(connIdx);
			if( connAC == connAB )
				continue;
			View viewC = connAC.other(viewA);

			// The views must form a complete loop with 3D information
			Motion connBC = viewB.findMotion(viewC);
			if( null == connBC || !connBC.is3D )
				continue;

			// Maximize worst case 3D information
			double score = Math.min(utils.scoreMotion.score(connAC) , utils.scoreMotion.score(connBC));

			if( score > bestScore ) {
				bestScore = score;
				bestConnection = connAC;
			}
		}

		return bestConnection;
	}

	@Override
	public void setVerbose(@Nullable PrintStream out, @Nullable Set<String> configuration) {
		this.verbose = out;
	}
}
