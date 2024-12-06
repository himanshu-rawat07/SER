from django.shortcuts import render
from django.http import JsonResponse
import librosa
import sounddevice as sd
from .scripts.Prediction import start 
# import subprocess

def index(request):
    return render(request, 'index.html')

def about_us(request):
    return render(request, "about_us.html")

# @csrf_exempt
def save_file(request):    
    if request.method == 'POST':
        audio_blob = request.FILES['audio_data']            
        # Save the uploaded file to a temporary location
        with open('data.wav', 'wb') as f:
            for chunk in audio_blob.chunks():
                f.write(chunk)                
        # Load the audio file using librosa
        y, sr = librosa.load('data.wav', sr=None)        
        # Play the audio file using sounddevice
        sd.play(y, sr)
        sd.wait()  # Wait for the audio to finish playing        
        # Render the new.html template
        return JsonResponse({'result': 'error', 'message': 'Sucess'})        
    else:
        return JsonResponse({'result': 'error', 'message': 'Invalid request'})
    
def classify(request): 
    result=start()
    print(type(result))
    for i in result:
        new_string = i.split('_')
    context = {"result":new_string[0],"result2":new_string[1]}
    # context = {"result":"male","result2":"sad"}
    return render(request, "classify.html",context)