package com.balltracker.capture.ui

sealed interface Place {
    data object Film : Place
    data object Balls : Place
    data class Ball(val id: String) : Place
    data object Settings : Place
}
