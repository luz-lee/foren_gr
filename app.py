import gradio as gr
from image import apply_image_watermark
from video import apply_video_watermark

# Image Interface
image_inputs = [
    gr.Image(type="numpy", label="Upload Image"),
    gr.Textbox(label="Watermark Text")
]

image_outputs = gr.Image(type="numpy", label="Watermarked Image")

image_interface = gr.Interface(
    fn=apply_image_watermark,
    inputs=image_inputs,
    outputs=image_outputs,
    title="Image Watermark Application",
    description="Upload an image and add a watermark text."
)

# Video Interface
video_inputs = [
    gr.Video(label="Upload Video"),
    gr.Textbox(label="Watermark Text")
]

video_outputs = gr.Video(label="Watermarked Video")

video_interface = gr.Interface(
    fn=apply_video_watermark,
    inputs=video_inputs,
    outputs=video_outputs,
    title="Video Watermark Application",
    description="Upload a video and add a watermark text."
)

# Combine both interfaces in tabs
app = gr.TabbedInterface(
    interface_list=[image_interface, video_interface],
    tab_names=["Image", "Video"]
)

if __name__ == "__main__":
    app.launch()
