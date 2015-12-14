/*
 * Copyright (c) 2011-2015, Peter Abeles. All Rights Reserved.
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

package boofcv.demonstrations.shapes;

import boofcv.abst.filter.binary.InputToBinary;
import boofcv.alg.filter.binary.BinaryImageOps;
import boofcv.alg.filter.binary.Contour;
import boofcv.alg.shapes.FitData;
import boofcv.alg.shapes.ShapeFittingOps;
import boofcv.factory.filter.binary.ConfigThreshold;
import boofcv.factory.filter.binary.FactoryThresholdBinary;
import boofcv.gui.DemonstrationBase;
import boofcv.gui.binary.VisualizeBinaryData;
import boofcv.gui.feature.VisualizeFeatures;
import boofcv.gui.feature.VisualizeShapes;
import boofcv.gui.image.ImageZoomPanel;
import boofcv.gui.image.ShowImages;
import boofcv.struct.ConnectRule;
import boofcv.struct.PointIndex_I32;
import boofcv.struct.image.ImageType;
import boofcv.struct.image.ImageUInt8;
import georegression.struct.point.Point2D_I32;
import georegression.struct.shapes.EllipseRotated_F64;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Fits shapes to contours from binary images
 *
 * @author Peter Abeles
 */
public class ShapeFitContourApp
		extends DemonstrationBase<ImageUInt8>
		implements ThresholdControlPanel.Listener
{
	// displays intensity image
	VisualizePanel gui = new VisualizePanel();

	// converted input image
	ImageUInt8 inputPrev = new ImageUInt8(1,1);
	ImageUInt8 binary = new ImageUInt8(1,1);
	ImageUInt8 filtered = new ImageUInt8(1,1);
	// if it has processed an image or not
	boolean processImage = false;

	InputToBinary<ImageUInt8> inputToBinary;

	// Found contours
	List<Contour> contours;

	BufferedImage original;
	BufferedImage work = new BufferedImage(1,1,BufferedImage.TYPE_INT_RGB);

	ShapeFitContourPanel controlPanel;

	public ShapeFitContourApp(List<String> examples ) {
		super(examples, ImageType.single(ImageUInt8.class));

		controlPanel = new ShapeFitContourPanel(this);

		add(BorderLayout.WEST, controlPanel);
		add(BorderLayout.CENTER, gui);

		ConfigThreshold config = controlPanel.getThreshold().config;
		inputToBinary = FactoryThresholdBinary.threshold(config,ImageUInt8.class);
	}

	@Override
	public synchronized void processImage(final BufferedImage buffered, ImageUInt8 input ) {
		if( buffered != null ) {
			original = conditionalDeclare(buffered,original);
			work = conditionalDeclare(buffered,work);

			this.original.createGraphics().drawImage(buffered,0,0,null);

			binary.reshape(input.getWidth(), input.getHeight());
			filtered.reshape(input.getWidth(),input.getHeight());
			inputPrev.setTo(input);

			SwingUtilities.invokeLater(new Runnable() {
				@Override
				public void run() {
					Dimension d = gui.getPreferredSize();
					if( d.getWidth() < buffered.getWidth() || d.getHeight() < buffered.getHeight() ) {
						gui.setPreferredSize(new Dimension(buffered.getWidth(), buffered.getHeight()));
					}
				}});
		} else {
			input = inputPrev;
		}

		process(input);
	}

	public synchronized void viewUpdated() {
		if( contours == null )
			return;

		int view = controlPanel.getSelectedView();

		Graphics2D g2 = work.createGraphics();

		if( view == 0 ) {
			g2.drawImage(original, 0, 0, null);
		} else if( view == 1 ){
			VisualizeBinaryData.renderBinary(binary,false,work);
		} else {
			g2.setColor(Color.BLACK);
			g2.fillRect(0,0,work.getWidth(),work.getHeight());
		}

		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run() {
				gui.setBufferedImage(work);
				gui.setScale(controlPanel.getZoom());
				gui.repaint();
				gui.requestFocusInWindow();
			}
		});
	}


	public void process( final ImageUInt8 input ) {

		// threshold the input image
		inputToBinary.process(input,binary);

		// reduce noise with some filtering
		BinaryImageOps.erode8(binary, 1, filtered);
		BinaryImageOps.dilate8(filtered, 1, binary);

		// Find the contour around the shapes
		contours = BinaryImageOps.contour(binary, ConnectRule.EIGHT,null);
		processImage = true;

		viewUpdated();
	}

	@Override
	public void imageThresholdUpdated() {
		ConfigThreshold config = controlPanel.getThreshold().config;
		inputToBinary = FactoryThresholdBinary.threshold(config,ImageUInt8.class);
		processImageThread(null,null);
	}

	protected void renderVisuals( Graphics2D g2 , double scale ) {

		int activeAlg = controlPanel.getSelectedAlgorithm();


		g2.setStroke(new BasicStroke(3));

		g2.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		if( controlPanel.contoursVisible ) {
			g2.setStroke(new BasicStroke(1));
			g2.setColor(new Color(0, 100, 0));
			VisualizeBinaryData.renderExternal(contours, scale, g2);
		}

		if( activeAlg == 0 ) {

			double splitFraction = controlPanel.getSplitFraction();
			double minimumSplitFraction = controlPanel.getMinimumSplitFraction();

			for( Contour c : contours ) {
				List<PointIndex_I32> vertexes = ShapeFittingOps.fitPolygon(
						c.external, true, splitFraction, minimumSplitFraction, 100);

				g2.setColor(Color.RED);
				g2.setStroke(new BasicStroke(2));
				VisualizeShapes.drawPolygon(vertexes, true,scale, g2);

				if( controlPanel.isCornersVisible() ) {
					g2.setColor(Color.BLUE);
					g2.setStroke(new BasicStroke(2f));
					for (PointIndex_I32 p : vertexes) {
						VisualizeFeatures.drawCircle(g2, scale * (p.x+0.5), scale * (p.y+0.5), 5);
					}
				}
			}
		} else if( activeAlg == 1 ) {
			// Filter small contours since they can generate really wacky ellipses
			for( Contour c : contours ) {
				if( c.external.size() > 10) {
					FitData<EllipseRotated_F64> ellipse = ShapeFittingOps.fitEllipse_I32(c.external,0,false,null);

					g2.setColor(Color.RED);
					g2.setStroke(new BasicStroke(2.5f));
					VisualizeShapes.drawEllipse(ellipse.shape,scale, g2);
				}

				for( List<Point2D_I32> internal : c.internal ) {
					if( internal.size() <= 10 )
						continue;
					FitData<EllipseRotated_F64> ellipse = ShapeFittingOps.fitEllipse_I32(internal,0,false,null);

					g2.setColor(Color.GREEN);
					g2.setStroke(new BasicStroke(2.5f));
					VisualizeShapes.drawEllipse(ellipse.shape,scale, g2);
				}
			}
		}
	}

	class VisualizePanel extends ImageZoomPanel {
		@Override
		protected void paintInPanel(AffineTransform tran, Graphics2D g2) {
			synchronized ( ShapeFitContourApp.this ) {

				renderVisuals(g2,scale);
			}
		}
	}

	public static void main( String args[] ) {

		List<String> examples = new ArrayList<String>();
		examples.add("particles01.jpg");
		examples.add("shapes/shapes02.png");
		examples.add("shapes/line_text_test_image.png");

		ShapeFitContourApp app = new ShapeFitContourApp(examples);

		app.openFile(new File(examples.get(0)));

		app.waitUntilDoneProcessing();

		ShowImages.showWindow(app,"Contour Shape Fitting",true);
	}
}
