from flask import Flask,render_template,request
from PIL import Image 
#from MIRNet.mirnet.inference import Inferer
from enhancer import Inferer

inferer = Inferer()
inferer.build_model(
num_rrg=3, num_mrb=2, channels=64,
weights_path='low_light_weights_best.h5'
)

app = Flask(__name__)     


@app.route('/')
def Home():
    return render_template('index.html')              

@app.route('/', methods = ['POST','GET'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
       
        f.save("./static/Image/new.jpg")            

        a = Image.open('./static/Image/new.jpg')
        a.save('./static/Image/original.jpg')      
       
        img_path = './static/Image/new.jpg'                  # 2 getting path to display image

    return render_template('index.html',user_image = img_path) 

@app.route('/enhanced')
def enhanceImage():
 
    img_path = 'static/Image/new.jpg'
    
    original_image, output_image = inferer.infer(img_path)

    output_image.save('static/Image/new.jpg')
    
    return render_template('index.html',user_image = img_path) 


@app.route('/compare')
def compareImg():
    return render_template('compare.html') 

@app.route('/home')
def home():
    return render_template('index.html') 


if __name__ == "__main__":
    app.run(debug = True, port = 5000)



