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

package boofcv.struct.geo;

import georegression.struct.point.Point2D_F64;

import static boofcv.misc.BoofMiscOps.assertBoof;

/**
 * Associated set of {@link Point2D_F64} for an arbitrary number of views that is fixed.
 *
 * @author Peter Abeles
 */
public class AssociatedTupleN {
	/** Set of associated observations */
	public final Point2D_F64[] p;

	public AssociatedTupleN(int num ) {
		p = new Point2D_F64[num];
		for (int i = 0; i < num; i++) {
			p[i] = new Point2D_F64();
		}
	}

	public double getX( int index ) {
		return p[index].x;
	}

	public double getY( int index ) {
		return p[index].y;
	}

	public Point2D_F64 get( int index ) {
		return p[index];
	}

	public void set( int index , double x , double y ) {
		p[index].set(x,y);
	}

	public void set( int index , Point2D_F64 src ) {
		p[index].set(src);
	}

	public int size() {
		return p.length;
	}

	public void setTo( AssociatedTupleN src ) {
		assertBoof(src.size()== size());

		for (int i = 0; i < p.length; i++) {
			p[i].setTo(src.p[i]);
		}
	}
}
