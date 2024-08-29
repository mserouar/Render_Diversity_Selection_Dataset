# import os
# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output, State
# import base64
# import io
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import models
# import albumentations as A
# import albumentations.pytorch.transforms as transforms
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import cv2
# import time
# import pandas as pd
# import plotly.express as px
# from scipy.cluster.hierarchy import linkage, fcluster
# from scipy.spatial.distance import pdist, squareform
# import random
# import shutil

# import matplotlib
# matplotlib.use('Agg')

# # Initialize the Dash app
# app = dash.Dash(__name__, suppress_callback_exceptions=True)

# # Global variable to keep track of progress and results
# progress = {
#     'current': 0,
#     'total': 0
# }

# results = {
#     'tx': None,
#     'ty': None,
#     'images': None,
#     'features': None
# }

# # Feature extractor class
# class Features_extractor(nn.Module):
#     def __init__(self, conv_layer=11):
#         super(Features_extractor, self).__init__()
#         vgg16 = models.vgg16(pretrained=True)
#         self.vgg_features = list(vgg16.features.children())

#         idx = self.find_layer(conv_layer)

#         new_extractor = nn.Sequential(*self.vgg_features)[:idx + 1]
#         self.features = new_extractor

#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

#     def find_layer(self, n_conv):
#         count = 0
#         look_for = n_conv

#         for idx, i in enumerate(self.vgg_features):
#             if isinstance(i, nn.modules.activation.ReLU):
#                 count += 1
#                 if count == look_for:
#                     break
#         return idx

#     def norm_aac(self, x):
#         x = self.features(x)
#         x = self.avg_pool(x)
#         x = x.squeeze(-1).squeeze(-1)
#         return x / torch.norm(x)

#     def forward(self, x):
#         x = self.norm_aac(x)
#         return x

# def torchload_img(imgp):
#     data_transforms = A.Compose([
#         A.Resize(256, 256),
#         A.CenterCrop(224, 224),
#         A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         transforms.ToTensorV2(),
#     ], p=1)

#     img = cv2.imread(imgp)[..., ::-1]
#     augmented = data_transforms(image=img)
#     img = augmented["image"].unsqueeze(0)
#     return img

# # Define the app layout
# app.layout = html.Div([
#     dcc.Input(
#         id='input-folder-path',
#         type='text',
#         placeholder='Enter folder path...',
#         style={'width': '50%'}
#     ),
#     html.Button('Run Analysis', id='run-analysis-button'),
#     dcc.Interval(
#         id='progress-interval',
#         interval=1000,  # 1 second
#         n_intervals=0,
#     ),
#     html.Div(
#         id='progress-bar-container',
#         children=[
#             html.Div(
#                 id='progress-bar',
#                 style={'width': '0%', 'height': '30px', 'background-color': '#4caf50', 'color': 'white', 'text-align': 'center', 'line-height': '30px'}
#             )
#         ],
#         style={'width': '50%', 'margin-top': '10px', 'border': '1px solid #ddd'}
#     ),
#     html.Div(id='analysis-output'),
#     html.Div(id='progress-output', style={'margin-top': '10px'}),
#     html.Div(id='mode-buttons', children=[
#         html.Button('Interactive Mode', id='interactive-mode-button', style={'display': 'none'}),
#         html.Button('Random Mode', id='random-mode-button', style={'display': 'none'})
#     ]),
#     html.Div(id='mode-output')
# ])

# @app.callback(
#     Output('progress-bar', 'style'),
#     Output('progress-output', 'children'),
#     Output('interactive-mode-button', 'style'),
#     Output('random-mode-button', 'style'),
#     Input('progress-interval', 'n_intervals')
# )
# def update_progress_bar(n_intervals):
#     if progress['total'] > 0:
#         progress_percentage = (progress['current'] / progress['total']) * 100
#         return {'width': f'{progress_percentage}%', 'height': '30px', 'background-color': '#4caf50', 'color': 'white', 'text-align': 'center', 'line-height': '30px'}, f'Processing: {progress["current"]}/{progress["total"]} files processed', {'display': 'inline-block'}, {'display': 'inline-block'}
#     return {'width': '0%', 'height': '30px', 'background-color': '#4caf50', 'color': 'white', 'text-align': 'center', 'line-height': '30px'}, 'Initializing...', {'display': 'none'}, {'display': 'none'}

# @app.callback(
#     Output('analysis-output', 'children'),
#     Input('run-analysis-button', 'n_clicks'),
#     State('input-folder-path', 'value')
# )
# def run_analysis(n_clicks, folder_path):
#     if n_clicks is None or not folder_path or not os.path.isdir(folder_path):
#         return html.Div('Please enter a valid folder path.')

#     global progress, results
#     progress['current'] = 0
#     progress['total'] = len(os.listdir(folder_path))

#     # Initialize the feature extractor
#     DL_extract = Features_extractor()
#     DLfeatures = []

#     data = []
#     for i, filename in enumerate(os.listdir(folder_path)):
#         image_path = os.path.join(folder_path, filename)
#         if os.path.isfile(image_path):
#             img = torchload_img(image_path)
#             feature = DL_extract(img).detach().numpy().flatten()
#             data.append([feature, image_path])
            
#             # Update progress
#             progress['current'] += 1

#             # Simulate processing delay
#             time.sleep(0.1)  # Adjust based on actual processing time

#     if not data:
#         return html.Div('No images found or processed.')

#     features, images = zip(*data)
#     features = np.array(features)

#     # PCA transformation
#     pca = PCA(n_components=10)
#     pca_features = pca.fit_transform(features)

#     # t-SNE transformation
#     tsne = TSNE(n_components=2, learning_rate=350, perplexity=30, angle=0.2, verbose=2).fit_transform(pca_features)
#     tx, ty = tsne[:, 0], tsne[:, 1]
#     tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
#     ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

#     results['tx'] = tx
#     results['ty'] = ty
#     results['images'] = images
#     results['features'] = features

#     # Create the composite image
#     width = 4000
#     height = 3000
#     max_dim = 100

#     full_image = Image.new('RGBA', (width, height))
#     for img, x, y in zip(images, tx, ty):
#         tile = Image.open(img)
#         rs = max(1, tile.width / max_dim, tile.height / max_dim)
#         tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.Resampling.LANCZOS)
#         full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))

#     # Convert the composite image to a format that can be displayed in Dash
#     buf = io.BytesIO()
#     full_image.save(buf, format="PNG")
#     buf.seek(0)
#     encoded_image = base64.b64encode(buf.read()).decode()

#     # Plot the composite image using Matplotlib
#     plt.figure(figsize=(16, 12))
#     plt.imshow(full_image)
#     plt.axis('off')

#     # Save the plot to a BytesIO object
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     buf.seek(0)
#     plot_encoded_image = base64.b64encode(buf.read()).decode()

#     return html.Div([
#         html.Img(src=f'data:image/png;base64,{plot_encoded_image}', style={'max-width': '100%'}),
#         html.Div('Analysis complete!')
#     ])



# @app.callback(
#     Output('mode-output', 'children'),
#     Input('interactive-mode-button', 'n_clicks'),
#     Input('random-mode-button', 'n_clicks')
# )
# def display_mode(interactive_clicks, random_clicks):
#     ctx = dash.callback_context

#     if not ctx.triggered:
#         return dash.no_update

#     button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     tx = results['tx']
#     ty = results['ty']
#     images = results['images']
#     features = results['features']

#     if button_id == 'interactive-mode-button':
#         return html.Div([
#             dcc.Graph(id='scatter-plot'),
#             html.Button('Download Selected Images CSV', id='download-csv-button'),
#             dcc.Download(id='download-csv')  # Make sure this ID is included in the layout
#         ])

#     elif button_id == 'random-mode-button':
#         return html.Div([
#             html.Img(id='random-mode-image', style={'max-width': '100%'}),
#             html.Div('Random mode complete!')
#         ])

# @app.callback(
#     Output('scatter-plot', 'figure'),
#     Input('interactive-mode-button', 'n_clicks')
# )
# def update_scatter_plot(n_clicks):
#     tx = results['tx']
#     ty = results['ty']
#     images = results['images']

#     positions = np.array(list(zip(tx, ty)))
#     linked = linkage(positions, 'ward')
#     max_d = 0.5
#     clusters = fcluster(linked, max_d, criterion='distance')

#     df = pd.DataFrame({
#         'x': positions[:, 0] * 4000,
#         'y': positions[:, 1] * 3000,
#         'cluster': clusters,
#         'image_name': [images[i] for i in range(len(images))]
#     })

#     fig = px.scatter(df, x='x', y='y', color='cluster',
#                     color_continuous_scale=px.colors.sequential.Viridis,
#                     labels={'x': 'X Coordinate', 'y': 'Y Coordinate', 'cluster': 'Cluster'},
#                     title='Interactive Cluster Plot')

#     fig.update_traces(marker=dict(size=10),
#                       selector=dict(mode='markers'),
#                       hovertemplate='Cluster: %{customdata[0]}<br>Image: %{customdata[1]}<extra></extra>',
#                       customdata=df[['cluster', 'image_name']])

#     fig.update_layout(width=900, height=700)

#     return fig

# @app.callback(
#     Output('download-csv', 'data'),
#     Input('scatter-plot', 'clickData'),
#     State('scatter-plot', 'figure'),
#     prevent_initial_call=True
# )
# def save_selected_images_to_csv(clickData, figure):
#     if clickData is None:
#         return dash.no_update

#     # Get cluster from clicked point
#     point_index = clickData['points'][0]['pointIndex']
#     df = pd.DataFrame(figure['data'][0]['customdata'], columns=['cluster', 'image_name'])
    
#     selected_cluster = df.iloc[point_index]['cluster']
    
#     # Filter images in the selected cluster
#     cluster_images = df[df['cluster'] == selected_cluster]['image_name']
    
#     # Select 5 random images from the cluster
#     selected_images = random.sample(list(cluster_images), min(5, len(cluster_images)))
    
#     if not os.path.exists('/home/mario/Bureau/Codes/DIVERSITY/Selected'):
#         os.makedirs('/home/mario/Bureau/Codes/DIVERSITY/Selected')
    
#     # Copy images to the destination folder
#     for image_name in selected_images:
#         src_path = os.path.join(image_name) 

#         if os.path.isfile(src_path):
            
#             original_path = src_path
#             new_directory = '/home/mario/Bureau/Codes/DIVERSITY/Selected'
#             filename = os.path.basename(original_path)
#             new_path = os.path.join(new_directory, filename)
#             dest_path = new_path
     
#             shutil.copy2(src_path, '/home/mario/Bureau/Codes/DIVERSITY/Selected')
#             print(f"Copied: {src_path} to {dest_path}")
#         else:
#             print(f"File not found: {src_path}")

#     return dict(message="Images have been copied successfully.")




# @app.callback(
#     Output('random-mode-image', 'src'),
#     Input('random-mode-button', 'n_clicks')
# )
# def generate_random_mode_image(n_clicks):
#     tx = results['tx']
#     ty = results['ty']
#     images = results['images']

#     coordinates = np.array(list(zip(tx, ty)))
#     distances = squareform(pdist(coordinates, metric='euclidean'))

#     def select_diverse_images(num_images, distances):
#         num_total_images = distances.shape[0]
#         selected_indices = []
        
#         # Start with a random image
#         current_index = np.random.randint(num_total_images)
#         selected_indices.append(current_index)
        
#         for _ in range(1, num_images):
#             # Compute distance from already selected images
#             min_distances = np.min(distances[:, selected_indices], axis=1)
#             # Exclude already selected indices
#             min_distances[selected_indices] = -1
            
#             # Select the index with the maximum minimum distance
#             next_index = np.argmax(min_distances)
#             selected_indices.append(next_index)
        
#         return selected_indices

#     # Select 10 most diverse images
#     top_indices = select_diverse_images(5, distances)
#     other_indices = list(set(range(len(images))) - set(top_indices))

#     # Parameters for final image
#     width = 4000
#     height = 3000
#     max_dim = 100

#     # Create a new image to paste all the images onto
#     full_image = Image.new('RGBA', (width, height))

#     # Helper function to open and resize images
#     def process_image(img_path, alpha=1.0):
#         tile = Image.open(img_path)
#         rs = max(1, tile.width / max_dim, tile.height / max_dim)
#         tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.Resampling.LANCZOS)
#         if alpha < 1.0:
#             # Apply transparency
#             tile = tile.convert('RGBA')
#             data = tile.getdata()
#             new_data = [(r, g, b, int(a * alpha)) for (r, g, b, a) in data]
#             tile.putdata(new_data)
#         return tile

#     # Paste the top 10 most diverse images with full opacity
#     for i in top_indices:
#         x = int((width - max_dim) * tx[i])
#         y = int((height - max_dim) * ty[i])
#         tile = process_image(images[i])
#         full_image.paste(tile, (x, y), mask=tile.convert('RGBA'))

#         if not os.path.exists('/home/mario/Bureau/Codes/DIVERSITY/Selected'):
#             os.makedirs('/home/mario/Bureau/Codes/DIVERSITY/Selected')
        
#         shutil.copy2(images[i], '/home/mario/Bureau/Codes/DIVERSITY/Selected')


#     # Paste other images with 0.5 alpha
#     for i in other_indices:
#         x = int((width - max_dim) * tx[i])
#         y = int((height - max_dim) * ty[i])
#         tile = process_image(images[i], alpha=0.5)
#         full_image.paste(tile, (x, y), mask=tile.convert('RGBA'))

#     # Display the final image using matplotlib
#     buf = io.BytesIO()
#     plt.figure(figsize=(16, 12))
#     plt.imshow(full_image)
#     plt.axis('off')
#     plt.savefig(buf, format='png')
#     plt.close()
#     buf.seek(0)
#     plot_encoded_image = base64.b64encode(buf.read()).decode()

#     return f'data:image/png;base64,{plot_encoded_image}'



# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)








# python app.py
# http://127.0.0.1:8050/


import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
import albumentations as A
import albumentations.pytorch.transforms as transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
import time
import pandas as pd
import plotly.express as px
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import random
import shutil

import matplotlib
matplotlib.use('Agg')

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Global variable to keep track of progress and results
progress = {
    'current': 0,
    'total': 0
}

results = {
    'tx': None,
    'ty': None,
    'images': None,
    'features': None
}

# Feature extractor class
class Features_extractor(nn.Module):
    def __init__(self, conv_layer=11):
        super(Features_extractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.vgg_features = list(vgg16.features.children())

        idx = self.find_layer(conv_layer)

        new_extractor = nn.Sequential(*self.vgg_features)[:idx + 1]
        self.features = new_extractor

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def find_layer(self, n_conv):
        count = 0
        look_for = n_conv

        for idx, i in enumerate(self.vgg_features):
            if isinstance(i, nn.modules.activation.ReLU):
                count += 1
                if count == look_for:
                    break
        return idx

    def norm_aac(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        return x / torch.norm(x)

    def forward(self, x):
        x = self.norm_aac(x)
        return x

def torchload_img(imgp):
    data_transforms = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.ToTensorV2(),
    ], p=1)

    img = cv2.imread(imgp)[..., ::-1]
    augmented = data_transforms(image=img)
    img = augmented["image"].unsqueeze(0)
    return img

# Define the app layout
app.layout = html.Div([
    dcc.Input(
        id='input-folder-path',
        type='text',
        placeholder='Enter folder path...',
        style={'width': '50%'}
    ),
    html.Button('Run Analysis', id='run-analysis-button'),
    dcc.Interval(
        id='progress-interval',
        interval=1000,  # 1 second
        n_intervals=0,
    ),
    html.Div(
        id='progress-bar-container',
        children=[
            html.Div(
                id='progress-bar',
                style={'width': '0%', 'height': '30px', 'background-color': '#4caf50', 'color': 'white', 'text-align': 'center', 'line-height': '30px'}
            )
        ],
        style={'width': '50%', 'margin-top': '10px', 'border': '1px solid #ddd'}
    ),
    html.Div(id='analysis-output'),
    html.Div(id='progress-output', style={'margin-top': '10px'}),
    html.Div(id='mode-buttons', children=[
        html.Button('Interactive Mode', id='interactive-mode-button', style={'display': 'none'}),
        html.Button('Random Mode', id='random-mode-button', style={'display': 'none'})
    ]),
    html.Div(id='mode-output')
])

@app.callback(
    Output('progress-bar', 'style'),
    Output('progress-output', 'children'),
    Output('interactive-mode-button', 'style'),
    Output('random-mode-button', 'style'),
    Input('progress-interval', 'n_intervals')
)
def update_progress_bar(n_intervals):
    if progress['total'] > 0:
        progress_percentage = (progress['current'] / progress['total']) * 100
        return {'width': f'{progress_percentage}%', 'height': '30px', 'background-color': '#4caf50', 'color': 'white', 'text-align': 'center', 'line-height': '30px'}, f'Processing: {progress["current"]}/{progress["total"]} files processed', {'display': 'inline-block'}, {'display': 'inline-block'}
    return {'width': '0%', 'height': '30px', 'background-color': '#4caf50', 'color': 'white', 'text-align': 'center', 'line-height': '30px'}, 'Initializing...', {'display': 'none'}, {'display': 'none'}





@app.callback(
   Output('analysis-output', 'children'),
    Input('run-analysis-button', 'n_clicks'),
    State('input-folder-path', 'value')
)
def run_analysis(n_clicks, folder_path):
    
    print(folder_path)
    paths = folder_path.split(',')
    print(paths)

    # Initialize the feature extractor
    DL_extract = Features_extractor()
    DLfeatures = []

    data = []
    for j in paths:
        print(j)
        global progress, results
        progress['current'] = 0

        for i, filename in enumerate(os.listdir(j)):

            progress['total'] = len(os.listdir(j))

            image_path = os.path.join(j, filename)
            if os.path.isfile(image_path):
                img = torchload_img(image_path)
                feature = DL_extract(img).detach().numpy().flatten()
                data.append([feature, image_path])
                
                # Update progress
                progress['current'] += 1

                # Simulate processing delay
                time.sleep(0.1)  # Adjust based on actual processing time


    if not data:
        return html.Div('No images found or processed.')

    features, images = zip(*data)
    features = np.array(features)

    # PCA transformation
    pca = PCA(n_components=10)
    pca_features = pca.fit_transform(features)

    # t-SNE transformation
    tsne = TSNE(n_components=2, learning_rate=350, perplexity=30, angle=0.2, verbose=2).fit_transform(pca_features)
    tx, ty = tsne[:, 0], tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    results['tx'] = tx
    results['ty'] = ty
    results['images'] = images
    results['features'] = features

    # Create the composite image
    width = 2000
    height = 1500
    max_dim = 80

    full_image = Image.new('RGBA', (width, height))
    for img, x, y in zip(images, tx, ty):
        tile = Image.open(img)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.Resampling.LANCZOS)
        full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))

    # Convert the composite image to a format that can be displayed in Dash
    buf = io.BytesIO()
    full_image.save(buf, format="PNG")
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode()

    # Plot the composite image using Matplotlib
    plt.figure(figsize=(16, 12))
    plt.imshow(full_image)
    plt.axis('off')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_encoded_image = base64.b64encode(buf.read()).decode()

    return html.Div([
        html.Img(src=f'data:image/png;base64,{plot_encoded_image}', style={'max-width': '100%'}),
        html.Div('Analysis complete!')
    ])



@app.callback(
    Output('mode-output', 'children'),
    Input('interactive-mode-button', 'n_clicks'),
    Input('random-mode-button', 'n_clicks')
)
def display_mode(interactive_clicks, random_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    tx = results['tx']
    ty = results['ty']
    images = results['images']
    features = results['features']

    if button_id == 'interactive-mode-button':
        return html.Div([
            dcc.Graph(id='scatter-plot'),
            html.Button('Download Selected Images CSV', id='download-csv-button'),
            dcc.Download(id='download-csv')  # Make sure this ID is included in the layout
        ])

    elif button_id == 'random-mode-button':
        return html.Div([
            html.Img(id='random-mode-image', style={'max-width': '100%'}),
            html.Div('Random mode complete!')
        ])



@app.callback(
    Output('scatter-plot', 'figure'),
    Input('interactive-mode-button', 'n_clicks')
)
def update_scatter_plot(n_clicks):
    tx = results['tx']
    ty = results['ty']
    images = results['images']

    positions = np.array(list(zip(tx, ty)))
    linked = linkage(positions, 'ward')
    max_d = 0.5
    clusters = fcluster(linked, max_d, criterion='distance')

    df = pd.DataFrame({
        'x': positions[:, 0] * 2000,
        'y': positions[:, 1] * 1500,
        'cluster': clusters,
        'image_name': [images[i] for i in range(len(images))]
    })

    fig = px.scatter(df, x='x', y='y', color='cluster',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    labels={'x': 'X Coordinate', 'y': 'Y Coordinate', 'cluster': 'Cluster'},
                    title='Interactive Cluster Plot')

    fig.update_traces(marker=dict(size=25),
                      selector=dict(mode='markers'),
                      hovertemplate='Cluster: %{customdata[0]}<br>Image: %{customdata[1]}<extra></extra>',
                      customdata=df[['cluster', 'image_name']])

    fig.update_layout(width=900, height=700, yaxis = dict(autorange="reversed"))

    return fig

@app.callback(
    Output('download-csv', 'data'),
    Input('scatter-plot', 'clickData'),
    State('scatter-plot', 'figure'),
    prevent_initial_call=True
)
def save_selected_images_to_csv(clickData, figure):
    if clickData is None:
        return dash.no_update

    # Get cluster from clicked point
    point_index = clickData['points'][0]['pointIndex']
    df = pd.DataFrame(figure['data'][0]['customdata'], columns=['cluster', 'image_name'])
    
    selected_cluster = df.iloc[point_index]['cluster']
    
    # Filter images in the selected cluster
    cluster_images = df[df['cluster'] == selected_cluster]['image_name']
    
    # Select 5 random images from the cluster
    selected_images = random.sample(list(cluster_images), min(5, len(cluster_images)))
    
    if not os.path.exists('/home/mario/Bureau/Codes/DIVERSITY/Selected'):
        os.makedirs('/home/mario/Bureau/Codes/DIVERSITY/Selected')
    
    # Copy images to the destination folder
    for image_name in selected_images:
        src_path = os.path.join(image_name) 

        if os.path.isfile(src_path):
            
            original_path = src_path
            new_directory = '/home/mario/Bureau/Codes/DIVERSITY/Selected'
            filename = os.path.basename(original_path)
            new_path = os.path.join(new_directory, filename)
            dest_path = new_path
     
            shutil.copy2(src_path, '/home/mario/Bureau/Codes/DIVERSITY/Selected')

    return dict(message="Images have been copied successfully.")



@app.callback(
    Output('random-mode-image', 'src'),
    Input('random-mode-button', 'n_clicks')
)
def generate_random_mode_image(n_clicks):
    tx = results['tx']
    ty = results['ty']
    images = results['images']

    coordinates = np.array(list(zip(tx, ty)))
    distances = squareform(pdist(coordinates, metric='euclidean'))

    def select_diverse_images(num_images, distances):
        num_total_images = distances.shape[0]
        selected_indices = []
        
        # Start with a random image
        current_index = np.random.randint(num_total_images)
        selected_indices.append(current_index)
        
        for _ in range(1, num_images):
            # Compute distance from already selected images
            min_distances = np.min(distances[:, selected_indices], axis=1)
            # Exclude already selected indices
            min_distances[selected_indices] = -1
            
            # Select the index with the maximum minimum distance
            next_index = np.argmax(min_distances)
            selected_indices.append(next_index)
        
        return selected_indices

    # Select 10 most diverse images
    top_indices = select_diverse_images(5, distances)
    other_indices = list(set(range(len(images))) - set(top_indices))

    # Parameters for final image
    width = 2000
    height = 1500
    max_dim = 80

    # Create a new image to paste all the images onto
    full_image = Image.new('RGBA', (width, height))

    # Helper function to open and resize images
    def process_image(img_path, alpha=1.0):
        tile = Image.open(img_path)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.Resampling.LANCZOS)
        if alpha < 1.0:
            # Apply transparency
            tile = tile.convert('RGBA')
            data = tile.getdata()
            new_data = [(r, g, b, int(a * alpha)) for (r, g, b, a) in data]
            tile.putdata(new_data)
        return tile

    # Paste the top 10 most diverse images with full opacity
    for i in top_indices:
        x = int((width - max_dim) * tx[i])
        y = int((height - max_dim) * ty[i])
        tile = process_image(images[i])
        full_image.paste(tile, (x, y), mask=tile.convert('RGBA'))

        if not os.path.exists('/home/mario/Bureau/Codes/DIVERSITY/Selected'):
            os.makedirs('/home/mario/Bureau/Codes/DIVERSITY/Selected')
        
        shutil.copy2(images[i], '/home/mario/Bureau/Codes/DIVERSITY/Selected')


    # Paste other images with 0.5 alpha
    for i in other_indices:
        x = int((width - max_dim) * tx[i])
        y = int((height - max_dim) * ty[i])
        tile = process_image(images[i], alpha=0.5)
        full_image.paste(tile, (x, y), mask=tile.convert('RGBA'))

    # Display the final image using matplotlib
    buf = io.BytesIO()
    plt.figure(figsize=(16, 12))
    plt.imshow(full_image)
    plt.axis('off')
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_encoded_image = base64.b64encode(buf.read()).decode()

    return f'data:image/png;base64,{plot_encoded_image}'



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
