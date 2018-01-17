import gdal,osr


def save_geotiff(raster_array, area_def, export_path):
	
	# set manually the number of bands to one, because we know you have only one layer
	bands_number = 1
	
	# Create gdal geotiff driver
        gtiff_driver = gdal.GetDriverByName('GTiff')

	# Pick up the numbers format, use gdal.GDT_Float32 for floats
        gtiff_format = gdal.GDT_Float64
        gtiff_options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"]
        gtiff_options = ["COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6", "INTERLEAVE=BAND"]
        gtiff_options = []
        
	# Create output file (empty at this point) and define projection and extent parameters
	gtiff_dataset = gtiff_driver.Create(export_path,
                                             int(area_def.x_size),
                                             int(area_def.y_size),
                                             bands_number,
                                             gtiff_format,
                                             gtiff_options)

	# Define area extent for the Geotiff dataset        
	geometry_list = (area_def.area_extent[0],
                         area_def.pixel_size_x,
                         0,
                         area_def.area_extent[3],
                         0,
                         area_def.pixel_size_y * -1)

	# Set projection parameters
        gtiff_dataset.SetGeoTransform(geometry_list)
        srs = osr.SpatialReference()
        srs.ImportFromProj4(area_def.proj4_string.encode('ascii'))
        gtiff_dataset.SetProjection(srs.ExportToWkt())
	
	# Get the empty band from the dataset, so gdal knows where to write the data	
	gtiff_band = gtiff_dataset.GetRasterBand(1)
	
	# Write the layer (your raster array) data into the geotiff dataset
	gtiff_band.WriteArray(raster_array)

	gtiff_dataset = None
