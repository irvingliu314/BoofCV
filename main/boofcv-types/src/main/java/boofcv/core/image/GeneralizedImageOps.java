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

package boofcv.core.image;

import boofcv.concurrency.*;
import boofcv.struct.image.*;

/**
 * <p>
 * Operations that return information about the specific image.  Useful when writing highly abstracted code
 * which is independent of the input image.
 * </p>
 *
 * @author Peter Abeles
 */
// TODO rename to GImageOps ?
public class GeneralizedImageOps {

	public static boolean isFloatingPoint(Class<?> imgType) {
		if( GrayF.class.isAssignableFrom(imgType) ) {
			return true;
		} else {
			return false;
		}
	}

	public static double get(ImageGray img, int x, int y) {
		if (img instanceof GrayI8) {
			return ((GrayI8) img).get(x, y);
		} else if (img instanceof GrayI16) {
			return ((GrayI16) img).get(x, y);
		} else if (img instanceof GrayS32) {
			return ((GrayS32) img).get(x, y);
		} else if (img instanceof GrayF32) {
			return ((GrayF32) img).get(x, y);
		} else if (img instanceof GrayF64) {
			return ((GrayF64) img).get(x, y);
		} else if (img instanceof GrayS64) {
			return ((GrayS64) img).get(x, y);
		} else {
			throw new IllegalArgumentException("Unknown or incompatible image type: " + img.getClass().getSimpleName());
		}
	}

	public static double get(ImageBase img, int x, int y , int band ) {
		if (img instanceof ImageGray) {
			return get((ImageGray) img, x, y);
		} else if (img instanceof ImageInterleaved) {
			return get((ImageInterleaved) img, x, y, band);
		} else if (img instanceof Planar) {
			return get(((Planar) img).getBand(band), x, y);
		} else {
			throw new IllegalArgumentException("Unknown or incompatible image type: " + img.getClass().getSimpleName());
		}
	}

	public static double get(ImageInterleaved img, int x, int y , int band ) {
		if (img instanceof InterleavedU8) {
			return ((InterleavedU8) img).getBand(x, y, band);
		} else if (img instanceof InterleavedS8) {
			return ((InterleavedS8) img).getBand(x, y, band);
		} else if (img instanceof InterleavedS16) {
			return ((InterleavedS16) img).getBand(x, y, band);
		} else if (img instanceof InterleavedU16) {
			return ((InterleavedU16) img).getBand(x, y, band);
		} else if (img instanceof InterleavedS32) {
			return ((InterleavedS32) img).getBand(x, y, band);
		} else if (img instanceof InterleavedS64) {
			return ((InterleavedS64) img).getBand(x, y, band);
		} else if (img instanceof InterleavedF32) {
			return ((InterleavedF32) img).getBand(x, y, band);
		} else if (img instanceof InterleavedF64) {
			return ((InterleavedF64) img).getBand(x, y, band);
		} else {
			throw new IllegalArgumentException("Unknown or incompatible image type: " + img.getClass().getSimpleName());
		}
	}

	public static <T extends ImageGray> T createSingleBand(ImageDataType type, int width, int height) {
		Class<T> typeClass = ImageType.getImageClass(ImageType.Family.GRAY, type);
		return (T)createSingleBand(typeClass, width, height);
	}

	public static <T extends ImageBase> T createImage(Class<T> type, int width, int height, int numBands ) {
		if( (Class)type == Planar.class )
			throw new IllegalArgumentException("Can't use this function with planar because the data type needs to be specified too");

		if( ImageGray.class.isAssignableFrom(type))
			return (T)createSingleBand((Class)type,width,height);
		else if( ImageInterleaved.class.isAssignableFrom(type))
			return (T)createInterleaved((Class)type,width,height,numBands);
		else
			throw new RuntimeException("Unknown. "+type.getSimpleName());
	}
	public static <T extends ImageGray<T>> T createSingleBand(Class<T> type, int width, int height) {
		type = convertGenericToSpecificType(type);

		if (type == GrayU8.class) {
			return (T)new GrayU8(width, height);
		} else if (type == GrayS8.class) {
			return (T)new GrayS8(width, height);
		} else if (type == GrayS16.class) {
			return (T)new GrayS16(width, height);
		} else if (type == GrayU16.class) {
			return (T)new GrayU16(width, height);
		} else if (type == GrayS32.class) {
			return (T)new GrayS32(width, height);
		} else if (type == GrayS64.class) {
			return (T)new GrayS64(width, height);
		} else if (type == GrayF32.class) {
			return (T)new GrayF32(width, height);
		} else if (type == GrayF64.class) {
			return (T)new GrayF64(width, height);
		} else if( (Class)type == GrayI.class ) {
			// ImageInteger is a generic type, so just create something
			return (T)new GrayS32(width,height);
		}
		throw new RuntimeException("Unknown type: "+type.getSimpleName());
	}

	public static <T extends ImageInterleaved> T createInterleaved(ImageDataType type, int width, int height , int numBands) {
		Class<T> typeClass = ImageType.getImageClass(ImageType.Family.INTERLEAVED, type);
		return (T)createInterleaved(typeClass,width,height,numBands);
	}

	public static <T extends ImageInterleaved<T>> T createInterleaved(Class<T> type, int width, int height , int numBands) {
		type = convertGenericToSpecificType(type);

		if (type == InterleavedU8.class) {
			return (T)new InterleavedU8(width, height,numBands);
		} else if (type == InterleavedS8.class) {
			return (T)new InterleavedS8(width, height,numBands);
		} else if (type == InterleavedU16.class) {
			return (T)new InterleavedU16(width, height,numBands);
		} else if (type == InterleavedS16.class) {
			return (T)new InterleavedS16(width, height,numBands);
		} else if (type == InterleavedS32.class) {
			return (T)new InterleavedS32(width, height,numBands);
		} else if (type == InterleavedS64.class) {
			return (T)new InterleavedS64(width, height,numBands);
		} else if (type == InterleavedF32.class) {
			return (T)new InterleavedF32(width, height,numBands);
		} else if (type == InterleavedF64.class) {
			return (T)new InterleavedF64(width, height,numBands);
		} else if( (Class)type == ImageInterleaved.class ) {
			// ImageInteger is a generic type, so just create something
			return (T)new InterleavedS32(width,height,numBands);
		}
		throw new RuntimeException("Unknown type: "+type.getSimpleName());
	}

	public static void set(ImageGray img, int x, int y, double value) {
		if (GrayI.class.isAssignableFrom(img.getClass())) {
			((GrayI)img).set(x,y,(int)value);
		} else if (img instanceof GrayF32) {
			((GrayF32) img).set(x, y,(float)value);
		} else if (img instanceof GrayF64) {
			((GrayF64) img).set(x, y, value);
		} else if (img instanceof GrayS64) {
			((GrayS64) img).set(x, y, (long)value);
		} else {
			throw new IllegalArgumentException("Unknown or incompatible image type: " + img.getClass().getSimpleName());
		}
	}

	public static void setM(ImageBase img, int x, int y, double... value) {
		if( img instanceof Planar) {
			Planar ms = (Planar) img;

			for (int i = 0; i < value.length; i++) {
				set(ms.getBand(i), x, y, value[i]);
			}
		} else if( img instanceof ImageInterleaved ) {
			for (int band = 0; band < value.length; band++) {
				if (img instanceof InterleavedU8) {
					((InterleavedU8) img).setBand(x, y, band, (byte) value[band]);
				} else if (img instanceof InterleavedS8) {
					((InterleavedS8) img).setBand(x, y, band, (byte) value[band]);
				} else if (img instanceof InterleavedS16) {
					((InterleavedS16) img).setBand(x, y, band, (short) value[band]);
				} else if (img instanceof InterleavedU16) {
					((InterleavedU16) img).setBand(x, y, band, (short) value[band]);
				} else if (img instanceof InterleavedS32) {
					((InterleavedS32) img).setBand(x, y, band, (int) value[band]);
				} else if (img instanceof InterleavedS64) {
					((InterleavedS64) img).setBand(x, y, band, (long) value[band]);
				} else if (img instanceof InterleavedF32) {
					((InterleavedF32) img).setBand(x, y, band, (float) value[band]);
				} else if (img instanceof InterleavedF64) {
					((InterleavedF64) img).setBand(x, y, band, value[band]);
				} else {
					throw new IllegalArgumentException("Unknown or incompatible image type: " + img.getClass().getSimpleName());
				}
			}
		} else if( img instanceof ImageGray) {
			if( value.length != 1 )
				throw new IllegalArgumentException("For a single band image the input pixel must have 1 band");
			set((ImageGray)img,x,y,value[0]);
		} else {
			throw new IllegalArgumentException("Add support for this image type!");
		}
	}

	public static void setB(ImageBase img, int x, int y, int band , double value ) {
		if( img instanceof Planar) {
			Planar ms = (Planar) img;

			GeneralizedImageOps.set(ms.getBand(band),x,y,value);
		} else if( img instanceof ImageInterleaved ) {
			if (img instanceof InterleavedU8) {
				((InterleavedU8) img).setBand(x, y, band, (byte) value);
			} else if (img instanceof InterleavedS8) {
				((InterleavedS8) img).setBand(x, y, band, (byte) value);
			} else if (img instanceof InterleavedS16) {
				((InterleavedS16) img).setBand(x, y, band, (short) value);
			} else if (img instanceof InterleavedU16) {
				((InterleavedU16) img).setBand(x, y, band, (short) value);
			} else if (img instanceof InterleavedS32) {
				((InterleavedS32) img).setBand(x, y, band, (int) value);
			} else if (img instanceof InterleavedS64) {
				((InterleavedS64) img).setBand(x, y, band, (long) value);
			} else if (img instanceof InterleavedF32) {
				((InterleavedF32) img).setBand(x, y, band, (float) value);
			} else if (img instanceof InterleavedF64) {
				((InterleavedF64) img).setBand(x, y, band, value);
			} else {
				throw new IllegalArgumentException("Unknown or incompatible image type: " + img.getClass().getSimpleName());
			}
		} else if( img instanceof ImageGray) {
			if( band != 0 )
				throw new IllegalArgumentException("For a single band image the input pixel must have 1 band");
			set((ImageGray)img,x,y,value);
		} else {
			throw new IllegalArgumentException("Add support for this image type!");
		}
	}

	public static <T extends ImageGray<T>> int getNumBits(Class<T> type) {
		if (type == GrayU8.class) {
			return 8;
		} else if (type == GrayS8.class) {
			return 8;
		} else if (type == GrayS16.class) {
			return 16;
		} else if (type == GrayU16.class) {
			return 16;
		} else if (type == GrayS32.class) {
			return 32;
		} else if (type == GrayS64.class) {
			return 64;
		} else if (type == GrayF32.class) {
			return 32;
		} else if (type == GrayF64.class) {
			return 64;
		}
		throw new RuntimeException("Unknown type: "+type.getSimpleName());
	}

	public static WorkArrays createWorkArray(ImageType<?> type) {
		if( type.getDataType().isInteger() ) {
			if( type.getDataType().getNumBits() < 64 ) {
				return new IWorkArrays();
			} else {
				return new LWorkArrays();
			}
		} else if( type.getDataType().getNumBits() < 64 ) {
			return new FWorkArrays();
		} else {
			return new DWorkArrays();
		}
	}

	/**
	 * If an image is to be created then the generic type can't be used a specific one needs to be.  An arbitrary
	 * specific image type is returned here.
	 */
	public static <T> T convertGenericToSpecificType(Class<?> type) {
		if (type == GrayI8.class)
			return (T) GrayU8.class;
		if (type == GrayI16.class)
			return (T) GrayS16.class;
		if (type == InterleavedI8.class)
			return (T) InterleavedU8.class;
		if (type == InterleavedI16.class)
			return (T) InterleavedS16.class;
		return (T) type;
	}

	public static Object getArray(ImageGray input) {
		if( input instanceof GrayI8)
			return ((GrayI8)input).data;
		else if( input instanceof GrayI16 )
			return ((GrayI16)input).data;
		else if( input instanceof GrayS32 )
			return ((GrayS32)input).data;
		else if( input instanceof GrayS64 )
			return ((GrayS64)input).data;
		else if( input instanceof GrayF32 )
			return ((GrayF32)input).data;
		else if( input instanceof GrayF64 )
			return ((GrayF64)input).data;
		else
			throw new RuntimeException("Unknown");
	}
}
