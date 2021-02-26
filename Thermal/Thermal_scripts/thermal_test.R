library(Thermimage)
library(fields)


# Function that read the RJPEG image and convert it into a Matrix where the values are the absolute temperatures 
# the image can be croped (the default values take the whole image)
plot_thermal <- function(image_path,crop=c(1,512,1,640)){ #[1:512,1:640]
  img<-readflirJPG(image_path, exiftoolpath="installed")
  cams<-flirsettings(f, exiftoolpath="installed", camvals="")
  ObjectEmissivity<- cams$Info$Emissivity              # Image Saved Emissivity - should be ~0.95 or 0.96
  dateOriginal<-cams$Dates$DateTimeOriginal             # Original date/time extracted from file
  dateModif<-   cams$Dates$FileModificationDateTime     # Modification date/time extracted from file
  PlanckR1<-    cams$Info$PlanckR1                      # Planck R1 constant for camera  
  PlanckB<-     cams$Info$PlanckB                       # Planck B constant for camera  
  PlanckF<-     cams$Info$PlanckF                       # Planck F constant for camera
  PlanckO<-     cams$Info$PlanckO                       # Planck O constant for camera
  PlanckR2<-    cams$Info$PlanckR2                      # Planck R2 constant for camera
  ATA1<-        cams$Info$AtmosphericTransAlpha1        # Atmospheric Transmittance Alpha 1
  ATA2<-        cams$Info$AtmosphericTransAlpha2        # Atmospheric Transmittance Alpha 2
  ATB1<-        cams$Info$AtmosphericTransBeta1         # Atmospheric Transmittance Beta 1
  ATB2<-        cams$Info$AtmosphericTransBeta2         # Atmospheric Transmittance Beta 2
  ATX<-         cams$Info$AtmosphericTransX             # Atmospheric Transmittance X
  OD<-          cams$Info$ObjectDistance                # object distance in metres
  FD<-          cams$Info$FocusDistance                 # focus distance in metres
  ReflT<-       cams$Info$ReflectedApparentTemperature  # Reflected apparent temperature
  AtmosT<-      cams$Info$AtmosphericTemperature        # Atmospheric temperature
  IRWinT<-      cams$Info$IRWindowTemperature           # IR Window Temperature
  IRWinTran<-   cams$Info$IRWindowTransmission          # IR Window transparency
  RH<-          cams$Info$RelativeHumidity              # Relative Humidity
  h<-           cams$Info$RawThermalImageHeight         # sensor height (i.e. image height)
  w<-           cams$Info$RawThermalImageWidth          # sensor width (i.e. image width)
  
  temperature<-raw2temp(img, ObjectEmissivity, OD, ReflT, AtmosT, IRWinT, IRWinTran, RH,
                        PlanckR1, PlanckB, PlanckF, PlanckO, PlanckR2, 
                        ATA1, ATA2, ATB1, ATB2, ATX)
  temperature <- temperature[crop[1]:crop[2],crop[3]:crop[4]]
  h2 = dim(temperature)[1]
  w2 = dim(temperature)[2]
  toto <- plotTherm(temperature, h=h2, w=w2, minrangeset=min(temperature)-0.5, maxrangeset=max(temperature)+0.5, trans="rotate270.matrix")
  return(temperature)
}

########### Examples ####################


#f<-paste0(system.file("extdata/IR_2412.jpg", package="Thermimage"))
#img_names <- list.files("/Users/stouzani/Google\ Drive/LBL-drone-thermal-project/Drone_Data_Capture/Richmond_Field_Station/02_26_2020/Thermal/", pattern = "*.jpg", full.names = TRUE) 
#img_names <- list.files("/Users/stouzani/Desktop/Unstructured_ML/Drone/Drone_Data_Capture/Alameda_July_2020/thermal/", pattern = "*.jpg", full.names = TRUE) 
img_names <- list.files("/Users/stouzani/Desktop/Unstructured_ML/Drone/Drone_Data_Capture/Alameda_July_2020/thermal/", pattern = "*.jpg", full.names = TRUE) 


#f <- img_names[130]
f <- "/Users/stouzani/Desktop/Unstructured_ML/Drone/Drone_Data_Capture/Veracruz/Thermal/DJI_0490.jpg"
f <- "/Users/stouzani/Desktop/Unstructured_ML/Drone/Drone_Data_Capture/singapour/Thermal/IX-x1-00057_0278_0109_THM.jpg"

temp<-plot_thermal(f)#,crop=c(170,512,1,280))

write.table(temp,"/Users/stouzani/Desktop/Unstructured_ML/Drone/Drone_Data_Capture/Veracruz/Thermal_preproc/DJI_0490.csv",row.names = FALSE, col.names =FALSE,sep = ",")
# write.table(temp,"/Users/stouzani/Desktop/Unstructured_ML/Drone/Drone_Data_Capture/Veracruz/Thermal_preproc/IX-x1-00057_0278_0109_THM_roof.csv",row.names = FALSE, col.names =FALSE,sep = ",")


#/Users/stouzani/Google\ Drive/LBL-drone-thermal-project/Drone_Images_December2019/Thermal
#/Users/stouzani/Desktop/Unstructured_ML/DL_Models/a3dbr/data/Drone_Flight/Thermal
img_names <- list.files("/Users/stouzani/Google\ Drive/LBL-drone-thermal-project/Drone_Images_December2019/Thermal", pattern = "*.jpg", full.names = TRUE) 
f <- img_names[335]
f
temp<-plot_thermal(f)
write.table(temp,"/Users/stouzani/Desktop/Unstructured_ML/Drone/Drone_Data_Capture/Alameda_December_2019/Thermal_preproc/DJI_0351.csv",row.names = FALSE, col.names =FALSE,sep = ",")







#####################  Old scripts ##########################################################

img<-readflirJPG(f, exiftoolpath="installed")

dim(img)



cams<-flirsettings(f, exiftoolpath="installed", camvals="")
head(cbind(cams$Info), 20) # Large amount of Info, show just the first 20 tages for readme

plancks<-flirsettings(f, exiftoolpath="installed", camvals="-*Planck*")
unlist(plancks$Info)
cbind(unlist(cams$Dates))
cams$Dates$DateTimeOriginal


ObjectEmissivity<- cams$Info$Emissivity              # Image Saved Emissivity - should be ~0.95 or 0.96
dateOriginal<-cams$Dates$DateTimeOriginal             # Original date/time extracted from file
dateModif<-   cams$Dates$FileModificationDateTime     # Modification date/time extracted from file
PlanckR1<-    cams$Info$PlanckR1                      # Planck R1 constant for camera  
PlanckB<-     cams$Info$PlanckB                       # Planck B constant for camera  
PlanckF<-     cams$Info$PlanckF                       # Planck F constant for camera
PlanckO<-     cams$Info$PlanckO                       # Planck O constant for camera
PlanckR2<-    cams$Info$PlanckR2                      # Planck R2 constant for camera
ATA1<-        cams$Info$AtmosphericTransAlpha1        # Atmospheric Transmittance Alpha 1
ATA2<-        cams$Info$AtmosphericTransAlpha2        # Atmospheric Transmittance Alpha 2
ATB1<-        cams$Info$AtmosphericTransBeta1         # Atmospheric Transmittance Beta 1
ATB2<-        cams$Info$AtmosphericTransBeta2         # Atmospheric Transmittance Beta 2
ATX<-         cams$Info$AtmosphericTransX             # Atmospheric Transmittance X
OD<-          cams$Info$ObjectDistance                # object distance in metres
FD<-          cams$Info$FocusDistance                 # focus distance in metres
ReflT<-       cams$Info$ReflectedApparentTemperature  # Reflected apparent temperature
AtmosT<-      cams$Info$AtmosphericTemperature        # Atmospheric temperature
IRWinT<-      cams$Info$IRWindowTemperature           # IR Window Temperature
IRWinTran<-   cams$Info$IRWindowTransmission          # IR Window transparency
RH<-          cams$Info$RelativeHumidity              # Relative Humidity
h<-           cams$Info$RawThermalImageHeight         # sensor height (i.e. image height)
w<-           cams$Info$RawThermalImageWidth          # sensor width (i.e. image width)


temperature<-raw2temp(img, ObjectEmissivity, OD, ReflT, AtmosT, IRWinT, IRWinTran, RH,
                      PlanckR1, PlanckB, PlanckF, PlanckO, PlanckR2, 
                      ATA1, ATA2, ATB1, ATB2, ATX)
str(temperature)   

temperature2 <- temperature#[70:432, 10:490]
dim(temperature2)
h2 = dim(temperature2)[1]
w2 = dim(temperature2)[2]

max(temperature2)
min(temperature2)

library(fields) # should be imported when installing Thermimage

toto <- plotTherm(temperature2, h=h2, w=w2, minrangeset=min(temperature2)-0.5, maxrangeset=max(temperature2)+0.5, trans="rotate270.matrix")


temperature2 <- wvtool::rotate.matrix(temperature2, 270)

temperature2_0 <- imager::as.cimg(temperature2)

#imager::save.image(temperature2_0,"/Users/stouzani/Desktop/Unstructured_ML/Drone/Drone_Data_Capture/Richmond_Field_Station/temperature2.png")

write.table(temperature2,"/Users/stouzani/Desktop/Unstructured_ML/Drone/Drone_Data_Capture/Veracruz/Thermal_preproc/temperature_DJI_0866.csv",row.names = FALSE, col.names =FALSE,sep = ",")




library(tiff)
test <- "/Users/stouzani/Google\ Drive/LBL-drone-thermal-project/Drone_Data_Capture/Richmond_Field_Station/03-12-2020/Thermal/DJI_0001.tif"

img <- readTIFF(test)





