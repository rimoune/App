import os
import subprocess
import base64
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from PIL import Image
import open3d as o3d
import numpy as np
import cProfile

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Upload(
        id='upload-images',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Images')
        ]),
        multiple=True
    ),
    html.Button('Process Images', id='process-button'),
#    html.Div(id='output-message')
    html.Div(id='output-data-upload'),
#    html.Img(id='top-down-view',src='./assets/top_down_view.jpg')
    html.Img(id='top-down-view',alt='cant display')
])


# Function to process a single image
#@cProfile.Profile
def capture_screen_image_top(point_cloud):
    print("in capture screen")
    pcd = o3d.io.read_point_cloud(point_cloud)    #pcd = o3d.io.read_point_cloud("tst.ply")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    top_down_image_path=os.path.join('./assets/',"top_down_view.jpg")
    vis.capture_screen_image(top_down_image_path)
    vis.destroy_window()
    print("almost there")
    return  top_down_image_path




@app.callback(
    #Output('top-down-view', 'src'),
    Output('top-down-view', 'src'),
    Input('process-button', 'n_clicks'),
    State('upload-images', 'filename'),
    State('upload-images', 'contents')
)
def process_images(n_clicks, filenames, contents):
    if n_clicks is None:
        raise PreventUpdate

    if not filenames:
        return "Please upload some images before processing."

    try:
        # Create sa temporary directory to store uploaded images
        temp_dir = 'temp_images'
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded images to the temporary directory
        image_paths = []
        for filename, content in zip(filenames, contents):
            content_type, content_string = content.split(',')
            image_data = base64.b64decode(content_string)
            with open(os.path.join(temp_dir, filename), 'wb') as f:
                f.write(image_data)

        output_folder = 'assets\preprocessed'

            # Ensure the output folder exists, or create it if necessary
        os.makedirs(output_folder, exist_ok=True)
        # Call your Python program on the images
        cmd = ['python', 'HorizonNet/preprocess.py', '--img_glob', os.path.join(temp_dir, filename), '--output_dir', output_folder] # Replace 'your_program.py' with your Python program's filename
        result=subprocess.run(cmd)

        print(f"Executing subprocess: {' '.join(cmd)}")
        # Clean up: Remove temporary directory and images
        for filename in filenames:
            os.remove(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)

        checkpoint_folder='HorizonNet/ckpt/resnet50_rnn__st3d.pth'

        cmd = ['python', 'HorizonNet/inference.py', '--pth', checkpoint_folder, '--img_glob', 'assets/preprocessed/'+filename.split('.')[0]+'_aligned_rgb.png','--output_dir', 'assets/inferenced']

        result=subprocess.run(cmd)

        print(f"Done executing subprocess: {' '.join(cmd)}")
        print("*done")

        cmd = ['python', 'HorizonNet/layout_viewer.py', '--img', os.path.join('assets/preprocessed',filename.split('.')[0]+'_aligned_rgb.png'), '--layout', os.path.join('assets/inferenced',filename.split('.')[0]+'_aligned_rgb.json'),'--ignore_ceiling','--out','tst.ply']


        result=subprocess.run(cmd)

        print(f"Done executing subprocess: {' '.join(cmd)}")
        print("results were: ", result)
         

        # Load the point cloud data from 'tst.ply'
        #point_cloud = o3d.io.read_point_cloud('tst.ply')
        try:
            # Capture the top-down view image
            image = capture_screen_image_top('tst.ply')

            print("image is: ",image)
            print(type(image))
            return image# "Image processing successful."
        except Exception as e:
            return f"could not return the image path : {str(e)}"

    except Exception as e:
        return f"Image processing failed: {str(e)}"


if __name__ == '__main__':
    with cProfile.Profile() as pr:
        app.run_server(debug=True)
    pr.print_stats()
