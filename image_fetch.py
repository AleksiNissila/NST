import requests
import uuid

def get_image():
    imname = 'img_'
    imname = imname + str(uuid.uuid4())
    impath = './content_images/' + imname + '.jpg'

    try:

        # Get an image from internet, save it to path and return path
        with open(impath, 'wb') as handle:

                response = requests.get('https://source.unsplash.com/random/1920x1080', stream=True)

                if not response.ok:
                    print(response)

                for block in response.iter_content(1024):
                    if not block:
                        break

                    handle.write(block)
        print("no error")
        return impath

    except:
        print("error in fetching image")
        pass



