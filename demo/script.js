console.log("ONNX Runtime:", ort);


const canvas = document.getElementById('drawCanvas')
const ctx = canvas.getContext('2d', { willReadFrequently: true });
let drawing = false
let lastX = 0, lastY = 0;
let needsUpdate = false;
ctx.fillStyle = 'black'
ctx.fillRect(0,0,canvas.width,canvas.height)

if (!ctx){
    console.error('Canvas could not be loaded') 
}

var model_path = 'demo_models/Demo-12000.onnx'

async function selectModel(sub_path){
    model_path = `demo_models/${sub_path}.onnx`
    await loadModel()
    drawing=false
    
    ctx.closePath()
    let inArr = preprocessImage(canvas)
    let inTensor = new ort.Tensor("float32",inArr,[1,1,28,28])
    let output = await runModel(inTensor)
    console.log(output)
    let results = handleOutput(output)
    console.log(results)
    inner_list = ''
    for (let i=0;i<results.length;i++){
        inner_list += `<li>${results[i][0]}: ${results[i][1]}</li>`
        }
    //results_html = '${inner_list} </ol>`
    results_html = inner_list

    document.getElementById('results').innerHTML = results_html
}


let session //init ort session
async function loadModel() { // init model
       try {
        session = await ort.InferenceSession.create(model_path);
        console.log("ONNX model loaded!",session);
        setupCanvas()
        return session;
    } catch (err) {
        console.error("Error loading ONNX model:", err);
    }
}

loadModel() // execute model load

async function runModel(input) {
    let feeds = { "input.1": input }
    let output = await session.run(feeds)
    return output
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

    canvas.addEventListener('mouseup',async (e)=>{
        drawing=false
        ctx.closePath()
        let inArr = preprocessImage(canvas)
        let inTensor = new ort.Tensor("float32",inArr,[1,1,28,28])
        let output = await runModel(inTensor)
        console.log(output)
        let results = handleOutput(output)
        console.log(results)
        inner_list = ''
        for (let i=0;i<results.length;i++){
            inner_list += `<li>${results[i][0]}: ${results[i][1]}</li>`
            }
        //results_html = '${inner_list} </ol>`
        results_html = inner_list

        document.getElementById('results').innerHTML = results_html
    })
}
function clearCanvas(){
    ctx.fillStyle = 'black'
    ctx.fillRect(0,0,canvas.width,canvas.height)
    document.getElementById('results').innerHTML = 'Nothing has been drawn'
}

//setupCanvas()

function resizeIm(OGCanvas,size=28){
    const imCnv = document.createElement('canvas')
    const imCtx = imCnv.getContext('2d', { willReadFrequently: true });

    imCnv.width = size
    imCnv.height = size

    imCtx.drawImage(OGCanvas,0,0,size,size)
    return imCnv
}

function toGrayScale(canvas){
    const imCtx = canvas.getContext('2d', { willReadFrequently: true });
    const imData = imCtx.getImageData(0,0,canvas.width,canvas.height)
    const data = imData.data

    for (let i =0;i<data.length;i+=4){    
        const gray = 0.2989 * data[i] + 0.5870 * data[i + 1] + 0.1140 * data[i + 2];
        data[i] = data[i+1] = data[i+2] = gray
    }
    imCtx.putImageData(imData,0,0)
    return canvas
}

function normalize2Arr(canvas){
    const imCtx = canvas.getContext('2d', { willReadFrequently: true });
    const imData = imCtx.getImageData(0,0,canvas.width,canvas.height)
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
    canvas = resizeIm(canvas)
    canvas = toGrayScale(canvas)
    let imArr = normalize2Arr(canvas)
    return imArr
}

function handleOutput(output,topn=3){
    console.log("Model output:", output);
    let outData = output['31'].cpuData
    let probs = softmax(outData)
    let alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    let char_probs = {}
    for (let i=0;i<probs.length;i++){
        char_probs[alphabet[i]] = probs[i]
    }
    sorted_probs_arr = Object.entries(char_probs).sort((a,b)=>b[1]-a[1])

    return sorted_probs_arr.slice(0,topn)
}

function softmax(arr){
    let softmax_arr = []
    let sum = 0
    for (let i=0;i<arr.length;i++){
        sum += Math.exp(arr[i])
    }
    for(let i=0;i<arr.length;i++){
        softmax_arr[i] = Math.exp(arr[i])/sum
    }
    return softmax_arr
}