const video = document.getElementById('video')

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('/models_emo'),
  // faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  // faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models_emo')
]).then(startVideo)

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => video.srcObject = stream,
    err => console.error(err)
  )
  toggle = true;
}

function stopVideo() {
  video.srcObject.getTracks()[0].stop();
  toggle = false;
}

video.addEventListener('play', () => {
  // const canvas = faceapi.createCanvasFromMedia(video)
  // document.body.append(canvas)
  // const displaySize = { width: video.width, height: video.height }
  // faceapi.matchDimensions(canvas, displaySize)
  setInterval(async () => {
    var emotion_checkbox=document.getElementById('customSwitch1');
    var doc = document.getElementById("page-container");
    if (emotion_checkbox.checked){
      if(toggle==false){
        startVideo();
      }

      const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceExpressions()
      // console.log(detections[0]['expressions'])
      delete detections[0]['expressions']['neutral']
      
      let max_emotion = Object.keys(detections[0]['expressions']).reduce(function(a, b){ return detections[0]['expressions'][a] > detections[0]['expressions'][b] ? a : b });
      // console.log(max_emotion)
      let colormatch = { angry: `rgba(235,80,71,${detections[0]['expressions']["angry"]})`, 
                        happy: `rgba(232,217,63,${detections[0]['expressions']["happy"]})`, 
                        sad: `rgba(42,74,156,${detections[0]['expressions']["sad"]})`, 
                        surprised: `rgba(74,202,232,${detections[0]['expressions']["surprised"]})`, 
                        disgusted: `rgba(127,39,8,${detections[0]['expressions']["disgusted"]})`, 
                        fearful: `rgba(27,29,23,${detections[0]['expressions']["fearful"]})`
                        // neutral: "white",
                        // sad: "rgba(42,74,156,0.67)", surprised: "rgba(74,202,232,0.67)", disgusted:"rgba(127,39,8,0.67)",
                        // fearful:"rgba(27,29,23,0.67)"
                      };
      
      // doc.style.backgroundColor = colormatch[max_emotion]
      // console.log(colormatch[max_emotion])
      doc.style.backgroundColor = colormatch[max_emotion];

    }else{doc.style.backgroundColor = "white";
      stopVideo();}
    // console.log(document.getElementById("page-container").style.backgroundColor)
    // console.log( document.getElementById("page-container");
    // )
          
    // const resizedDetections = faceapi.resizeResults(detections, displaySize)
    // canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    // // faceapi.draw.drawDetections(canvas, resizedDetections)
    // // faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
    // faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
  }, 500)
})
