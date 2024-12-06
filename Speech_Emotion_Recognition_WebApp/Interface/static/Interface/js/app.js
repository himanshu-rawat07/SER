//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; //stream from getUserMedia()
var rec; //Recorder.js object
var input; //MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb.
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext; //audio context to help us record

var audio = {};

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");
var classifyButton = document.getElementById("classifyButton");

//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
pauseButton.addEventListener("click", pauseRecording);

function startRecording() {
    console.log("recordButton clicked");

    /*
		Simple constraints object, for more advanced audio features see
		https://addpipe.com/blog/audio-constraints-getusermedia/
	*/

    var constraints = { audio: true, video: false };

    /*
    	Disable the record button until we get a success or fail from getUserMedia() 
	*/

    recordButton.disabled = true;
    stopButton.disabled = false;
    pauseButton.disabled = false;

    /*
    	We're using the standard promise based getUserMedia() 
    	https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
	*/

    navigator.mediaDevices
        .getUserMedia(constraints)
        .then(function (stream) {
            console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

            /*
			create an audio context after getUserMedia is called
			sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
			the sampleRate defaults to the one set in your OS for your playback device
		*/
            audioContext = new AudioContext();

            //update the format
            document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz";

            /*  assign to gumStream for later use  */
            gumStream = stream;

            /* use the stream */
            input = audioContext.createMediaStreamSource(stream);

            /* 
			Create the Recorder object and configure to record mono sound (1 channel)
			Recording 2 channels  will double the file size
		*/
            rec = new Recorder(input, { numChannels: 1 });

            //start the recording process
            rec.record();

            console.log("Recording started");
        })
        .catch(function (err) {
            //enable the record button if getUserMedia() fails
            recordButton.disabled = false;
            stopButton.disabled = true;
            pauseButton.disabled = true;
        });
}

function pauseRecording() {
    console.log("pauseButton clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause
        rec.stop();
        pauseButton.innerHTML = "Resume";
    } else {
        //resume
        rec.record();
        pauseButton.innerHTML = "Pause";
    }
}

function stopRecording() {
    console.log("stopButton clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton.disabled = true;
    recordButton.disabled = false;
    pauseButton.disabled = true;
    classifyButton.disabled = false;

    //reset button just in case the recording is stopped while paused
    pauseButton.innerHTML = "Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink);
    console.log(rec);
    console.log(typeof rec);
}

function createDownloadLink(blob) {
    audio = blob;

    console.log("audio->", audio);
    var url = URL.createObjectURL(blob);
    var au = document.createElement("audio");
    var li = document.createElement("li");
    var link = document.createElement("a");

    //name of .wav file to use during upload and download (without extendion)
    var filename = new Date().toISOString();

    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    //save to disk link
    link.href = url;
    link.download = filename + ".wav"; //download forces the browser to donwload the file using the  filename
    link.innerHTML = "Save to disk";

    //add the new audio element to li
    li.appendChild(au);

    //add the filename to the li
    li.appendChild(document.createTextNode(filename + ".wav "));

    //add the save to disk link to li
    li.appendChild(link);

    //add the li element to the ol
    recordingsList.appendChild(li);
    sendToBackend();
}

// Get the CSRF token from the cookie
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
        const cookies = document.cookie.split(";");
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Check if the cookie name matches the CSRF token cookie name
            if (cookie.startsWith(name + "=")) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function sendToBackend() {
    // console.log("classifyButton clicked");

    // Convert audio data to FormData
    var fd = new FormData();
    console.log(typeof audio);
    fd.append("audio_data", audio, "audio.wav");

    // Create XMLHttpRequest object
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/save_file/"); // Update the URL to the appropriate endpoint in your Django backend
    const csrftoken = getCookie("csrftoken");
    xhr.setRequestHeader("X-CSRFToken", csrftoken);
    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                // Request was successful, handle response here
                // window.location.href = "/predict";
                console.log("Response from backend:", xhr.responseText);
                // You can update the DOM or perform other actions based on the response from the backend
            } else {
                // Request failed, handle error here
                console.error("Error:", xhr.statusText);
            }
        }
    };

    // Send the FormData object as the request body
    xhr.send(fd);
}
