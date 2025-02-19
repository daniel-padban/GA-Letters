console.log("ONNX Runtime:", ort);


const canvas = document.getElementById('drawCanvas')
const ctx = canvas.getContext('2d')
let drawing = false
let lastX = 0, lastY = 0;
let needsUpdate = false;
ctx.fillStyle = 'black'
ctx.fillRect(0,0,canvas.width,canvas.height)

if (!ctx){
    console.error('Canvas could not be loaded') 
}

async function loadModel() { // init model
    const session = await ort.InferenceSession.create("model.onnx");
    console.log("ONNX model loaded!");

    return session
}
const session = loadModel() // assign model session to variable

async function runModel(input) {
    feeds = 
}

// Set initial canvas background to white
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

function setupCanvas() {
    canvas.addEventListener('mousedown', (e)=>{
        drawing = true
        ctx.beginPath()
    })
    canvas.addEventListener('mousemove', (e)=>{
        if (drawing){
            ctx.lineTo(e.offsetX, e.offsetY)
            ctx.strokeStyle = 'white'
            ctx.lineWidth = 10
            ctx.stroke()
        }
    })
    canvas.addEventListener('mouseup',(e)=>{
        drawing=false
        ctx.closePath()
        results = runModel(input=preprocessImage(canvas))
        document.getElementById('results').innerHTML = results
    })
}
//setupCanvas()

function resizeIm(imageData,size=28){
    const imCnv = document.createElement('imCnv')
    const imCtx = imCnv.getContext("2d");
    imCnv.width = size
    imCnv.height = size

    imCtx.drawImage(imageData,0,0,size,size)
    return imCnv
}

function toGrayScale(canvas){
    const imCtx = canvas.getContext('2d')
    const imData = canvas.getImageData(0,0,canvas.width,canvas.height)
    const data = imData.data

    for (let i =0;i<data.length;i+=4){    
        const gray = 0.2989 * data[i] + 0.5870 * data[i + 1] + 0.1140 * data[i + 2];
        data[i] = data[i+1] = data[i+2] = gray
    }
    imCtx.putImageData()
    return canvas
}

function normalize2Arr(canvas){
    const imCtx = canvas.getContext()
    const imData = imCtx.getImageData(0,0,canvas,width,canvas.height)
    const data = imData.data

    const imaArr = new Float32Array(28*28)
    const mean = 0.1736;
    const std = 0.3248;

    for (let i = 0; i < data.length; i += 4) {
        // Assuming the image is grayscale, so we just take one channel (R, G, B)
        const pixelValue = data[i] / 255; // Normalize between 0 and 1
        imaArr[Math.floor(i / 4)] = pixelValue;
    }    
    return imaArr.map(value => (value - mean) / std);
}

function preprocessImage(canvas){
    const imageData = canvas.getImageData(0,0,canvas.width,canvas.height)
    canvas = resizeIm(imageData)
    canvas = toGrayScale(canvas)
    let imArr = normalize2Arr(canvas)
    return imArr
}