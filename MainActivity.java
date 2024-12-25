package com.example.meatqualityclassifier;

import static com.example.meatqualityclassifier.R.id.galleryButton;

import android.Manifest;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private Interpreter interpreter;
    private static final String MODEL_FILE_NAME = "model.tflite"; // Replace with your model's filename
    private static final int IMAGE_SIZE = 224; // Input size for the model
    private static final int CAMERA_REQUEST_CODE = 1;
    private static final int GALLERY_REQUEST_CODE = 2;

    private ImageView imageView;
    private TextView resultTextView;
    private static final String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        resultTextView = findViewById(R.id.resultTextView);
        Button cameraButton = findViewById(R.id.cameraButton);
        Button galleryButton = findViewById(R.id.galleryButton);

        // Initialize the TFLite model
        try {
            interpreter = new Interpreter(loadModelFile(MODEL_FILE_NAME));
        } catch (IOException e) {
            Log.e(TAG, "Error loading model", e);
            resultTextView.setText("Error loading model");
        }

        // Camera button logic
        cameraButton.setOnClickListener(view -> {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, CAMERA_REQUEST_CODE);
            } else {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
            }
        });

        // Gallery button logic
        galleryButton.setOnClickListener(view -> {
            Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(galleryIntent, GALLERY_REQUEST_CODE);
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            Bitmap image = null;
            if (requestCode == CAMERA_REQUEST_CODE) {
                // Handle camera image
                image = (Bitmap) data.getExtras().get("data");
                if (image != null) {
                    imageView.setImageBitmap(image);
                    classifyImage(Bitmap.createScaledBitmap(image, IMAGE_SIZE, IMAGE_SIZE, false));
                }
            } else if (requestCode == GALLERY_REQUEST_CODE) {
                // Handle gallery image
                try {
                    Uri uri = data.getData();
                    image = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                    if (image != null) {
                        imageView.setImageBitmap(image);
                        classifyImage(Bitmap.createScaledBitmap(image, IMAGE_SIZE, IMAGE_SIZE, false));
                    }
                } catch (IOException e) {
                    Log.e(TAG, "Error loading image from gallery", e);
                }
            }
        }
    }

    private MappedByteBuffer loadModelFile(String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void classifyImage(Bitmap image) {
        if (interpreter == null) {
            resultTextView.setText("Model not initialized");
            return;
        }

        // Prepare ByteBuffer for input
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
        image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

        int pixel = 0;
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));  // Red channel
                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));   // Green channel
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255));          // Blue channel
            }
        }

        // Define the output array for model results (adjust as necessary for your model)
        float[][] resultArray = new float[1][3]; //  model has 3 output classes
        interpreter.run(byteBuffer, resultArray);

        // Debugging: Log the raw result
        Log.d(TAG, "Raw model output: " + resultArray[0][0] + ", " + resultArray[0][1] + ", " + resultArray[0][2]);

        // Process results (assuming shape: 1x3, where 3 corresponds to 3 classes)
        float[] confidences = resultArray[0];

        // Find the class with the highest probability
        int maxPos = 0;
        float maxConfidence = confidences[0]; // Start with the first class's confidence
        for (int i = 0; i < confidences.length; i++) {
            Log.d(TAG, "Confidence for class " + i + ": " + confidences[i]);
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i];
                maxPos = i;
            }
        }

        // Class labels for your model output
        String[] classes = {"Fresh","Borderline", "Spoiled"};
        if(classes[maxPos] =="Fresh" ) {resultTextView.setText("Prediction: " + classes[maxPos]+"\nGood to Go: " +
                "This item is perfectly fresh and ready to use." );}
        else if (classes[maxPos] =="Borderline" ) {resultTextView.setText("Prediction: "+ classes[maxPos]+
           "\nOn the Edge:  This item is still usable but might not last long. Use it soon after checking carefully.");}
            else{resultTextView.setText("Prediction: "+ classes[maxPos]+"\nNo Good:  This item is spoiled and should be discarded.");}


    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (interpreter != null) {
            interpreter.close();
        }
    }
}
