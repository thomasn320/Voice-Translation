import gradio as gr
from faster_whisper import WhisperModel

#Using CPU so compute type is int8
model = WhisperModel("base.en", device = "cpu", compute_type="int8")

#Main function to turn speech to text
def transcribe(audio_path):
    if audio_path is None:
        return "No audio"
    try:
        segments, info = model.transcribe(audio_path,beam_size=5)
        text = ""
        for segment in segments:
            text = text + segment.text
        text = text.strip()
        if not text:
            return "No Audio Detected"
        return text
    except Exception as e:
        return f"Trascription error : {e}"

#Use Gradio for UI
with gr.Blocks() as app:
    gr.Markdown("## Speech to Text Translation App")
    mic = gr.Audio(sources=["microphone"], type="filepath",label="Click button to record and click again to stop",
                   waveform_options=gr.WaveformOptions(show_recording_waveform= False))
    output = gr.Textbox(label = "You said", lines=4)

    mic.stop_recording(transcribe,inputs=mic, outputs=output)

if __name__ == "__main__":
    app.launch()
    