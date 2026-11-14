package com.balltracker.capture

import android.Manifest
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.balltracker.capture.ui.BallTrackerTheme
import com.balltracker.capture.ui.CaptureScreen
import com.balltracker.capture.ui.CaptureViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent {
            BallTrackerTheme {
                CaptureRoute()
            }
        }
    }
}

@Composable
private fun CaptureRoute(model: CaptureViewModel = viewModel()) {
    val state by model.state.collectAsState()
    val context = LocalContext.current
    val launcher = androidx.activity.compose.rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        model.onPermission(granted)
    }
    LaunchedEffect(Unit) {
        val granted = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.CAMERA,
        ) == android.content.pm.PackageManager.PERMISSION_GRANTED
        if (granted) model.onPermission(true) else launcher.launch(Manifest.permission.CAMERA)
    }
    CaptureScreen(
        state = state,
        onAttach = model::attachPreview,
        onDetach = model::detachPreview,
        onNudge = model::nudgeMarker,
        onMarksSettled = model::commitMarks,
        onRecord = model::toggleRecord,
        onAllowCamera = { launcher.launch(Manifest.permission.CAMERA) },
        onOrientationChanged = model::refreshPreview,
        onOpen = model::open,
        onDelete = model::deleteBall,
        onClipSeconds = model::setClipSeconds,
        onKeepPreview = model::setKeepPreview,
    )
}
