from glob import glob
import os

imDefect1 = "data/train/F/Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png"
imDefect2 = "data/train/A/RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png"

pathData = "data/train/"
pathValidation = "data/validation/"
pathTest = "data/test/"


def separateData(partition):
    if not os.path.exists(partition):
        os.mkdir(partition)
        print("Directory ", partition,  " Created ")
    else:
        print("Directory ", partition,  " already exists")

    start = 65
    end = start+10
    for i in range(start, end):
        folderName = chr(i)
        folderPath = pathData+folderName+"/*.png"
        # print("======================================================" +
        #       folderName+"======================================================\n")

        newPartition = partition+"/"+folderName
        if not os.path.exists(newPartition):
            os.mkdir(newPartition)

        i = 1
        for fn in glob(folderPath):
            #print(fn)
            if fn != imDefect2 and fn != imDefect1:                
                if i < 500:
                    fn = fn.replace("\\", "/")
                    fileName = fn.split("/")[3]
                    os.rename(fn, newPartition+"/"+fileName)
                    i += 1

            else:  # supprimer les deux images défectueuses
                print("delet defect image : " + fn)
                os.remove(fn)


separateData(pathValidation)
separateData(pathTest)
