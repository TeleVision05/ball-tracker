package com.balltracker.capture.data

import android.content.Context
import android.graphics.Bitmap
import com.balltracker.capture.geometry.StumpMarks
import java.io.File

data class SavedBall(
    val id: String,
    val video: File,
    val sidecar: File,
    val frame: File,
)

class CaptureStore(context: Context) {
    private val root: File = File(
        context.getExternalFilesDir(null) ?: context.filesDir,
        "captures",
    ).apply { mkdirs() }

    private val ballsDir = File(root, "balls").apply { mkdirs() }
    private val setupFile = File(root, "current-setup.json")
    val setupFrame: File = File(root, "setup-frame.jpg")

    fun saveMarks(marks: StumpMarks) {
        setupFile.writeText(renderJson(linkedMapOf("version" to 1, "markers" to markerMap(marks))))
    }

    fun saveSetup(marks: StumpMarks, frame: Bitmap) {
        setupFrame.outputStream().use { stream ->
            frame.compress(Bitmap.CompressFormat.JPEG, 92, stream)
        }
        saveMarks(marks)
    }

    fun loadMarks(): StumpMarks? {
        if (!setupFile.exists()) return null
        return parseMarks(setupFile.readText())
    }

    fun listBalls(): List<SavedBall> {
        val files = ballsDir.listFiles { file -> file.extension == "mp4" } ?: return emptyList()
        return files
            .sortedBy { it.name }
            .map { video ->
                val id = video.nameWithoutExtension
                SavedBall(
                    id = id,
                    video = video,
                    sidecar = File(ballsDir, "$id.json"),
                    frame = File(ballsDir, "$id-frame.jpg"),
                )
            }
    }

    fun deleteBall(ball: SavedBall) {
        ball.video.delete()
        ball.sidecar.delete()
        if (ball.frame.exists()) ball.frame.delete()
    }

    fun nextVideoFile(): Pair<String, File> {
        val next = ballsDir.listFiles().orEmpty().mapNotNull { file ->
            BALL_INDEX.find(file.name)?.groupValues?.get(1)?.toIntOrNull()
        }.maxOrNull() ?: 0
        val id = "ball-%04d".format(next + 1)
        return id to File(ballsDir, "$id.mp4")
    }

    fun writeSidecar(record: BallRecord) {
        val frameName = if (setupFrame.exists()) {
            val copy = File(ballsDir, "${record.id}-frame.jpg")
            setupFrame.copyTo(copy, overwrite = true)
            copy.name
        } else {
            record.setupFrameFile
        }
        File(ballsDir, "${record.id}.json").writeText(
            ballJson(record.copy(setupFrameFile = frameName)),
        )
    }

    companion object {
        private val BALL_INDEX = Regex("""ball-(\d+)""")

        private fun markerMap(marks: StumpMarks) = linkedMapOf(
            "nearLeft" to listOf(marks.nearLeft.x, marks.nearLeft.y),
            "nearRight" to listOf(marks.nearRight.x, marks.nearRight.y),
            "farLeft" to listOf(marks.farLeft.x, marks.farLeft.y),
            "farRight" to listOf(marks.farRight.x, marks.farRight.y),
        )
    }
}
