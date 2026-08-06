package com.smartscanai.scanner;

import android.Manifest;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.widget.Toast;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.getcapacitor.BridgeActivity;
import java.io.OutputStream;

/**
 * The app is a thin WebView wrapper around the deployed Streamlit app (see
 * capacitor.config.json) - all document-scanning logic stays server-side in
 * the existing, validated Python pipeline. Two things a bare WebView does
 * not handle on its own, and that this class exists to fix:
 *
 * 1. Camera access. st.camera_input calls the browser's getUserMedia() API.
 *    A WebView's default WebChromeClient does not grant that request, so
 *    without onPermissionRequest below the camera option would silently
 *    fail to produce a video stream.
 * 2. File downloads. st.download_button renders the PDF as a data: URI:
 *    a plain WebView has no default handler for that scheme, so clicking
 *    "Download PDF" would do nothing. onDownloadStart decodes it and saves
 *    to the Downloads folder via MediaStore instead.
 */
public class MainActivity extends BridgeActivity {

    private static final int CAMERA_PERMISSION_REQUEST = 1001;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this, new String[] {Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
        }

        WebView webView = getBridge().getWebView();

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(final PermissionRequest request) {
                runOnUiThread(() -> {
                    if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA)
                            == PackageManager.PERMISSION_GRANTED) {
                        request.grant(request.getResources());
                    } else {
                        request.deny();
                    }
                });
            }
        });

        webView.setDownloadListener((url, userAgent, contentDisposition, mimeType, contentLength) -> {
            try {
                if (url.startsWith("data:")) {
                    saveDataUri(url, mimeType);
                } else {
                    android.app.DownloadManager.Request request =
                            new android.app.DownloadManager.Request(Uri.parse(url));
                    request.setNotificationVisibility(
                            android.app.DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
                    request.setDestinationInExternalPublicDir(
                            android.os.Environment.DIRECTORY_DOWNLOADS, "smartscan_output.pdf");
                    android.app.DownloadManager manager =
                            (android.app.DownloadManager) getSystemService(DOWNLOAD_SERVICE);
                    manager.enqueue(request);
                    Toast.makeText(this, "Downloading...", Toast.LENGTH_SHORT).show();
                }
            } catch (Exception e) {
                Toast.makeText(this, "Download failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        });
    }

    /** Decode a data: URI (base64-encoded PDF from st.download_button) and save it to Downloads. */
    private void saveDataUri(String dataUri, String fallbackMimeType) {
        int comma = dataUri.indexOf(',');
        if (comma < 0) {
            Toast.makeText(this, "Nothing to save.", Toast.LENGTH_SHORT).show();
            return;
        }
        String header = dataUri.substring(5, comma); // strip "data:"
        String base64Data = dataUri.substring(comma + 1);
        String mimeType = header.contains(";") ? header.substring(0, header.indexOf(';')) : header;
        if (mimeType.isEmpty()) {
            mimeType = fallbackMimeType != null ? fallbackMimeType : "application/pdf";
        }
        byte[] bytes = Base64.decode(base64Data, Base64.DEFAULT);

        String fileName = "smartscan_output_" + System.currentTimeMillis() + ".pdf";
        ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
        values.put(MediaStore.MediaColumns.MIME_TYPE, mimeType);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            values.put(MediaStore.MediaColumns.RELATIVE_PATH, android.os.Environment.DIRECTORY_DOWNLOADS);
        }

        Uri collection = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q
                ? MediaStore.Downloads.EXTERNAL_CONTENT_URI
                : MediaStore.Files.getContentUri("external");
        Uri item = getContentResolver().insert(collection, values);
        if (item == null) {
            Toast.makeText(this, "Could not save file.", Toast.LENGTH_LONG).show();
            return;
        }
        try (OutputStream out = getContentResolver().openOutputStream(item)) {
            if (out != null) {
                out.write(bytes);
            }
            Toast.makeText(this, "Saved to Downloads: " + fileName, Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            Toast.makeText(this, "Save failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }
}
