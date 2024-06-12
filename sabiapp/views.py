import os
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImageUploadForm
from .models import UploadedImage
import pixellib
from pixellib.tune_bg import alter_bg
import cv2

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()

            # Get name from form input
            name = request.POST.get('name', '')

            # Perform background alteration
            change_bg = alter_bg()
            model_path = os.path.join(settings.BASE_DIR, 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')
            # Specify the background image path
            background_image_path = os.path.join(settings.BASE_DIR, 'bgg.jpg')

            try:
                change_bg.load_pascalvoc_model(model_path)
            except ValueError as e:
                return render(request, 'sabiapp/index.html', {'error': str(e)})
            input_image_path = uploaded_image.image.path
            output_image_path = os.path.join(settings.MEDIA_ROOT, 'output', f'new_img_{uploaded_image.id}.jpg')
            change_bg.change_bg_img(f_image_path=input_image_path,  b_image_path=background_image_path, output_image_name=output_image_path)  # RGB for rose red
            background_path = os.path.join(settings.BASE_DIR, 'sabicard.jpeg')
            add_photo_on_photo_with_coordinates(background_path, output_image_path, output_image_path, name)

            uploaded_image.output_image.name = f'output/new_img_{uploaded_image.id}.jpg'
            uploaded_image.save()
            return redirect('sabiapp:result', uploaded_image.id)
    else:
        form = ImageUploadForm()
    return render(request, 'sabiapp/upload.html', {'form': form})

def result(request, image_id):
    image = UploadedImage.objects.get(id=image_id)
    return render(request, 'sabiapp/result.html', {'image': image})

def download_image(request, image_id):
    image = UploadedImage.objects.get(id=image_id)
    file_path = image.output_image.path
    with open(file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='image/jpeg')
        response['Content-Disposition'] = f'attachment; filename="new_img_{image_id}.jpg"'
        return response

import cv2

def add_photo_on_photo_with_coordinates(background_path, foreground_path, output_path, name):
    # Load background and foreground images
    background = cv2.imread(background_path)
    foreground = cv2.imread(foreground_path)  # Load without alpha channel
    coordinates = (85, 81, 332, 352)

    # Extract region of interest from the background
    roi = background[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]

    # Set the region of interest to have no color (black)
    roi[:, :] = [0, 0, 0]  # Black background

    # Calculate the required padding to match the dimensions of the region of interest
    padding_size = (coordinates[2] - coordinates[0]) // 20  # 10% padding of the region of interest width

    # Resize foreground image to fit the specified region with padding
    foreground_resized = cv2.resize(foreground, (coordinates[2] - coordinates[0], coordinates[3] - coordinates[1]))

    # Add padding to the foreground image with the specified color
    padding_color = (97, 115, 222)  # Light red padding color (BGR format)
    padded_foreground = cv2.copyMakeBorder(foreground_resized, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=padding_color)

    # Resize the padded foreground image to match the dimensions of the background region
    padded_foreground_resized = cv2.resize(padded_foreground, (roi.shape[1], roi.shape[0]))

    # Overlay the smoothed padded foreground onto the region of interest
    roi_with_foreground = cv2.addWeighted(roi, 1,padded_foreground_resized, 1, 0)

    # Replace the region of interest in the background with the overlay
    background[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]] = roi_with_foreground

    name_coordinates = (505, 182, 677, 225)
    cv2.rectangle(background, (name_coordinates[0], name_coordinates[1]), (name_coordinates[2], name_coordinates[3]),
                  (0, 0, 0), -1)
    # Save the resulting image
    cv2.imwrite(output_path, background)

    # Add name to the background image
    image_with_photo = cv2.imread(output_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (507, 215)
    font_scale = 1
    font_thickness = 2
    font_color = (255, 255, 255)  # White color
    cv2.putText(image_with_photo, name, bottom_left_corner, font, font_scale, font_color, font_thickness)

    # Save the resulting image
    cv2.imwrite(output_path, image_with_photo)