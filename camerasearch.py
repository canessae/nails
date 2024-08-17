import subprocess

class CameraSearch:
    def __init__(self):
        self.cameraList = []

    def detect(self):
        command='''
        for dev in `find /dev -maxdepth 1 -iname 'video*' -printf "%f\n"`; do
         v4l2-ctl --list-formats --device /dev/$dev | grep -qE '\[[0-9]\]' && echo $dev `cat /sys/class/video4linux/$dev/name`;
        done'''
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        self.cameraList = []
        for line in process.stdout:
            if line.decode() != '\n':
                print(line.decode().strip())
                good = line.decode().strip()
                pieces = good.split(' ')
                new = {'devid': int(pieces[0][5]), 'name': ' '.join(pieces[1:])}
                self.cameraList.append(new)

        #order camera by ids
        self.cameraList.sort(key=lambda x: x["devid"], reverse=False)
        return self.cameraList


    def getNames(self):
        retval = []
        for item in self.cameraList:
            retval.append(item['name'])
        return retval


    def getIds(self):
        retval = []
        for item in self.cameraList:
            retval.append(item['devid'])
        return retval


    def getAllCameras(self):
        return self.cameraList


if __name__ == "__main__":
    list = CameraSearch()
    ret = list.detect()
    print(ret)
    print(list.getNames())
    print(list.getIds())
