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

package boofcv.abst.feature.associate;

import georegression.struct.point.Point2D_F64;
import org.ddogleg.struct.FastQueue;

/**
 * Feature set aware association algorithm that takes in account image location. It works by breaking sorting
 * descriptors into their sets and then
 * performing association independently. The matches are then combined together again into a single list.
 *
 * @author Peter Abeles
 */
public class AssociateDescriptionSets2D<Desc> extends BaseAssociateSets<Desc> {
	AssociateDescription2D<Desc> associator;

	public AssociateDescriptionSets2D(AssociateDescription2D<Desc> associator, Class<Desc> type) {
		super(associator,type);
		this.associator = associator;
	}

	/**
	 * Initializes association.
	 *
	 * @see AssociateDescription2D#initialize(int, int)
	 */
	public void initialize(int imageWidth , int imageHeight ) {
		this.associator.initialize(imageWidth, imageHeight);
	}

	@Override
	protected SetStruct newSetStruct() {
		return new SetStruct2D();
	}

	@Override
	public void clearSource() {
		super.clearSource();
		for (int i = 0; i < sets.size; i++) {
			SetStruct2D set = (SetStruct2D)sets.get(i);
			set.pixelsSrc.reset();
		}
	}


	@Override
	public void clearDestination() {
		super.clearDestination();
		for (int i = 0; i < sets.size; i++) {
			SetStruct2D set = (SetStruct2D)sets.get(i);
			set.pixelsDst.reset();
		}
	}


	/**
	 * Adds a new descriptor and its set to the list. The order that descriptors are added is important and saved.
	 */
	public void addSource( Desc description, double pixelX, double pixelY, int set )
	{
		final SetStruct2D ss = (SetStruct2D)sets.data[set];
		ss.src.add(description);
		ss.pixelsSrc.grow().set(pixelX,pixelY);
		ss.indexSrc.add(countSrc++);
	}

	/**
	 * Adds a new descriptor and its set to the list. The order that descriptors are added is important and saved.
	 */
	public void addDestination( Desc description, double pixelX, double pixelY, int set )
	{
		final SetStruct2D ss = (SetStruct2D)sets.data[set];
		ss.dst.add(description);
		ss.pixelsDst.grow().set(pixelX,pixelY);
		ss.indexDst.add(countDst++);
	}

	@Override
	public void associate() {
		if( sets.size <= 0 )
			throw new IllegalArgumentException("You must initialize first with the number of sets");

		// reset data structures
		matches.reset();
		unassociatedSrc.reset();
		unassociatedDst.reset();

		// Compute results inside each set and copy them over into the output structure
		for (int setIdx = 0; setIdx < sets.size; setIdx++) {
			SetStruct2D set = (SetStruct2D)sets.get(setIdx);

			// Associate features inside this set
			associator.setSource(set.pixelsSrc, set.src);
			associator.setDestination(set.pixelsDst, set.dst);
			associator.associate();

			saveSetAssociateResults(set);
		}
	}

	/**
	 * Adds 2D image coordinates to the set
	 */
	class SetStruct2D extends SetStruct {
		// feature locations in each set
		FastQueue<Point2D_F64> pixelsSrc = new FastQueue<>(Point2D_F64::new);
		FastQueue<Point2D_F64> pixelsDst = new FastQueue<>(Point2D_F64::new);

		@Override
		public void reset() {
			super.reset();
			pixelsSrc.reset();
			pixelsDst.reset();
		}
	}
}
