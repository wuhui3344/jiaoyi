# ONNX Runtime Android
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# kotlinx.serialization
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.**
-keepclassmembers class com.kronos.assistant.** {
    *** Companion;
}
-keepclasseswithmembers class com.kronos.assistant.** {
    kotlinx.serialization.KSerializer serializer(...);
}

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**
