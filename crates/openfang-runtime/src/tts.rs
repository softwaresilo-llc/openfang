//! Text-to-speech engine — synthesize text to audio.
//!
//! Auto-cascades through available providers based on configured API keys.

use openfang_types::config::TtsConfig;
use std::time::Duration;
use tokio::process::Command;

/// Maximum audio response size (10MB).
const MAX_AUDIO_RESPONSE_BYTES: usize = 10 * 1024 * 1024;

/// Result of TTS synthesis.
#[derive(Debug)]
pub struct TtsResult {
    pub audio_data: Vec<u8>,
    pub format: String,
    pub provider: String,
    pub duration_estimate_ms: u64,
}

/// Text-to-speech engine.
pub struct TtsEngine {
    config: TtsConfig,
}

impl TtsEngine {
    pub fn new(config: TtsConfig) -> Self {
        Self { config }
    }

    /// Detect which TTS provider is available based on environment variables.
    fn detect_provider() -> Option<&'static str> {
        if std::env::var("OPENAI_API_KEY").is_ok() {
            return Some("openai");
        }
        if std::env::var("ELEVENLABS_API_KEY").is_ok() {
            return Some("elevenlabs");
        }
        if std::env::var("GROQ_API_KEY").is_ok() {
            return Some("groq");
        }
        if non_empty_env("OPENFANG_TTS_LOCAL_BIN").is_some() {
            return Some("local");
        }
        None
    }

    /// Synthesize text to audio bytes.
    /// Auto-cascade: override -> configured provider -> detected provider.
    /// Optional overrides for voice and format (per-request, from tool input).
    pub async fn synthesize(
        &self,
        text: &str,
        voice_override: Option<&str>,
        format_override: Option<&str>,
    ) -> Result<TtsResult, String> {
        self.synthesize_with_provider(None, text, voice_override, format_override)
            .await
    }

    /// Synthesize text with optional explicit provider override.
    pub async fn synthesize_with_provider(
        &self,
        provider_override: Option<&str>,
        text: &str,
        voice_override: Option<&str>,
        format_override: Option<&str>,
    ) -> Result<TtsResult, String> {
        if !self.config.enabled {
            return Err("TTS is disabled in configuration".into());
        }

        // Validate text length
        if text.is_empty() {
            return Err("Text cannot be empty".into());
        }
        if text.len() > self.config.max_text_length {
            return Err(format!(
                "Text too long: {} chars (max {})",
                text.len(),
                self.config.max_text_length
            ));
        }

        let provider = configured_provider(provider_override)
            .or_else(|| configured_provider(self.config.provider.as_deref()))
            .or_else(|| Self::detect_provider())
            .ok_or(
                "No TTS provider configured. Set GROQ_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY, or OPENFANG_TTS_LOCAL_BIN",
            )?;

        match provider {
            "openai" => {
                self.synthesize_openai(text, voice_override, format_override)
                    .await
            }
            "elevenlabs" => self.synthesize_elevenlabs(text, voice_override).await,
            "groq" => {
                self.synthesize_groq(text, voice_override, format_override)
                    .await
            }
            "local" => {
                self.synthesize_local(text, voice_override, format_override)
                    .await
            }
            other => Err(format!("Unknown TTS provider: {other}")),
        }
    }

    /// Synthesize via OpenAI TTS API.
    async fn synthesize_openai(
        &self,
        text: &str,
        voice_override: Option<&str>,
        format_override: Option<&str>,
    ) -> Result<TtsResult, String> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set")?;

        // Apply per-request overrides or fall back to config defaults
        let voice = voice_override.unwrap_or(&self.config.openai.voice);
        let format = format_override.unwrap_or(&self.config.openai.format);

        let body = serde_json::json!({
            "model": self.config.openai.model,
            "input": text,
            "voice": voice,
            "response_format": format,
            "speed": self.config.openai.speed,
        });

        let client = reqwest::Client::new();
        let response = client
            .post("https://api.openai.com/v1/audio/speech")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .send()
            .await
            .map_err(|e| format!("OpenAI TTS request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let err = response.text().await.unwrap_or_default();
            let truncated = crate::str_utils::safe_truncate_str(&err, 500);
            return Err(format!("OpenAI TTS failed (HTTP {status}): {truncated}"));
        }

        // Check content length before downloading
        if let Some(len) = response.content_length() {
            if len as usize > MAX_AUDIO_RESPONSE_BYTES {
                return Err(format!(
                    "Audio response too large: {len} bytes (max {MAX_AUDIO_RESPONSE_BYTES})"
                ));
            }
        }

        let audio_data = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read audio response: {e}"))?;

        if audio_data.len() > MAX_AUDIO_RESPONSE_BYTES {
            return Err(format!(
                "Audio data exceeds {}MB limit",
                MAX_AUDIO_RESPONSE_BYTES / 1024 / 1024
            ));
        }

        // Rough duration estimate: ~150 words/min at ~12 bytes/ms for MP3
        let word_count = text.split_whitespace().count();
        let duration_ms = (word_count as u64 * 400).max(500); // ~400ms per word, min 500ms

        Ok(TtsResult {
            audio_data: audio_data.to_vec(),
            format: format.to_string(),
            provider: "openai".to_string(),
            duration_estimate_ms: duration_ms,
        })
    }

    /// Synthesize via ElevenLabs TTS API.
    async fn synthesize_elevenlabs(
        &self,
        text: &str,
        voice_override: Option<&str>,
    ) -> Result<TtsResult, String> {
        let api_key =
            std::env::var("ELEVENLABS_API_KEY").map_err(|_| "ELEVENLABS_API_KEY not set")?;

        let voice_id = voice_override.unwrap_or(&self.config.elevenlabs.voice_id);
        let url = format!("https://api.elevenlabs.io/v1/text-to-speech/{}", voice_id);

        let body = serde_json::json!({
            "text": text,
            "model_id": self.config.elevenlabs.model_id,
            "voice_settings": {
                "stability": self.config.elevenlabs.stability,
                "similarity_boost": self.config.elevenlabs.similarity_boost,
            }
        });

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("xi-api-key", &api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .send()
            .await
            .map_err(|e| format!("ElevenLabs TTS request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let err = response.text().await.unwrap_or_default();
            let truncated = crate::str_utils::safe_truncate_str(&err, 500);
            return Err(format!(
                "ElevenLabs TTS failed (HTTP {status}): {truncated}"
            ));
        }

        if let Some(len) = response.content_length() {
            if len as usize > MAX_AUDIO_RESPONSE_BYTES {
                return Err(format!(
                    "Audio response too large: {len} bytes (max {MAX_AUDIO_RESPONSE_BYTES})"
                ));
            }
        }

        let audio_data = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read audio response: {e}"))?;

        if audio_data.len() > MAX_AUDIO_RESPONSE_BYTES {
            return Err(format!(
                "Audio data exceeds {}MB limit",
                MAX_AUDIO_RESPONSE_BYTES / 1024 / 1024
            ));
        }

        let word_count = text.split_whitespace().count();
        let duration_ms = (word_count as u64 * 400).max(500);

        Ok(TtsResult {
            audio_data: audio_data.to_vec(),
            format: "mp3".to_string(),
            provider: "elevenlabs".to_string(),
            duration_estimate_ms: duration_ms,
        })
    }

    /// Synthesize via Groq OpenAI-compatible TTS API.
    async fn synthesize_groq(
        &self,
        text: &str,
        voice_override: Option<&str>,
        format_override: Option<&str>,
    ) -> Result<TtsResult, String> {
        let api_key = std::env::var("GROQ_API_KEY").map_err(|_| "GROQ_API_KEY not set")?;
        let model = non_empty_env("OPENFANG_GROQ_TTS_MODEL")
            .unwrap_or_else(|| "canopylabs/orpheus-v1-english".to_string());
        let voice = voice_override
            .map(|v| v.to_string())
            .or_else(|| non_empty_env("OPENFANG_GROQ_TTS_VOICE"))
            .unwrap_or_else(|| "tara".to_string());
        let format = format_override
            .map(|f| f.to_string())
            .or_else(|| non_empty_env("OPENFANG_GROQ_TTS_FORMAT"))
            .unwrap_or_else(|| "wav".to_string());

        let body = serde_json::json!({
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": format,
        });

        let client = reqwest::Client::new();
        let response = client
            .post("https://api.groq.com/openai/v1/audio/speech")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .timeout(Duration::from_secs(self.config.timeout_secs))
            .send()
            .await
            .map_err(|e| format!("Groq TTS request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let err = response.text().await.unwrap_or_default();
            let truncated = crate::str_utils::safe_truncate_str(&err, 500);
            return Err(format!("Groq TTS failed (HTTP {status}): {truncated}"));
        }

        if let Some(len) = response.content_length() {
            if len as usize > MAX_AUDIO_RESPONSE_BYTES {
                return Err(format!(
                    "Audio response too large: {len} bytes (max {MAX_AUDIO_RESPONSE_BYTES})"
                ));
            }
        }

        let audio_data = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read audio response: {e}"))?;

        if audio_data.len() > MAX_AUDIO_RESPONSE_BYTES {
            return Err(format!(
                "Audio data exceeds {}MB limit",
                MAX_AUDIO_RESPONSE_BYTES / 1024 / 1024
            ));
        }

        let word_count = text.split_whitespace().count();
        let duration_ms = (word_count as u64 * 400).max(500);

        Ok(TtsResult {
            audio_data: audio_data.to_vec(),
            format,
            provider: "groq".to_string(),
            duration_estimate_ms: duration_ms,
        })
    }

    /// Synthesize using local edge-tts CLI.
    async fn synthesize_local(
        &self,
        text: &str,
        voice_override: Option<&str>,
        format_override: Option<&str>,
    ) -> Result<TtsResult, String> {
        let bin = non_empty_env("OPENFANG_TTS_LOCAL_BIN").unwrap_or_else(|| "edge-tts".to_string());
        let voice = voice_override
            .map(|v| v.to_string())
            .or_else(|| non_empty_env("OPENFANG_TTS_LOCAL_VOICE"))
            .unwrap_or_else(|| "de-DE-KatjaNeural".to_string());
        let format = format_override
            .map(|f| f.to_string())
            .or_else(|| non_empty_env("OPENFANG_TTS_LOCAL_FORMAT"))
            .unwrap_or_else(|| "mp3".to_string());
        let rate = non_empty_env("OPENFANG_TTS_LOCAL_RATE");
        let pitch = non_empty_env("OPENFANG_TTS_LOCAL_PITCH");

        let temp_dir = std::env::temp_dir().join(format!("openfang-tts-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&temp_dir)
            .await
            .map_err(|e| format!("Failed to create temp directory for local TTS: {e}"))?;

        let result = async {
            let output_path = temp_dir.join(format!("speech.{format}"));
            let mut cmd = Command::new(&bin);
            cmd.arg("--text")
                .arg(text)
                .arg("--voice")
                .arg(&voice)
                .arg("--write-media")
                .arg(&output_path);
            if let Some(r) = rate.as_deref() {
                cmd.arg("--rate").arg(r);
            }
            if let Some(p) = pitch.as_deref() {
                cmd.arg("--pitch").arg(p);
            }

            let output = tokio::time::timeout(Duration::from_secs(self.config.timeout_secs), cmd.output())
                .await
                .map_err(|_| format!("Local TTS timed out after {}s (binary: '{bin}')", self.config.timeout_secs))?
                .map_err(|e| format!("Failed to start local TTS binary '{bin}'. Install edge-tts or set OPENFANG_TTS_LOCAL_BIN. Error: {e}"))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!(
                    "Local TTS failed (code {:?}): {}",
                    output.status.code(),
                    stderr.trim()
                ));
            }

            let audio_data = tokio::fs::read(&output_path).await.map_err(|e| {
                format!(
                    "Local TTS completed but no audio file found at {}: {e}",
                    output_path.display()
                )
            })?;

            if audio_data.is_empty() {
                return Err("Local TTS produced an empty audio file".to_string());
            }

            if audio_data.len() > MAX_AUDIO_RESPONSE_BYTES {
                return Err(format!(
                    "Audio data exceeds {}MB limit",
                    MAX_AUDIO_RESPONSE_BYTES / 1024 / 1024
                ));
            }

            let word_count = text.split_whitespace().count();
            let duration_ms = (word_count as u64 * 400).max(500);

            Ok(TtsResult {
                audio_data,
                format,
                provider: "local".to_string(),
                duration_estimate_ms: duration_ms,
            })
        }
        .await;

        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
        result
    }
}

fn configured_provider(value: Option<&str>) -> Option<&str> {
    value
        .map(str::trim)
        .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case("auto"))
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> TtsConfig {
        TtsConfig::default()
    }

    #[test]
    fn test_engine_creation() {
        let engine = TtsEngine::new(default_config());
        assert!(!engine.config.enabled);
    }

    #[test]
    fn test_config_defaults() {
        let config = TtsConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_text_length, 4096);
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.openai.voice, "alloy");
        assert_eq!(config.openai.model, "tts-1");
        assert_eq!(config.openai.format, "mp3");
        assert_eq!(config.openai.speed, 1.0);
        assert_eq!(config.elevenlabs.voice_id, "21m00Tcm4TlvDq8ikWAM");
        assert_eq!(config.elevenlabs.model_id, "eleven_monolingual_v1");
    }

    #[tokio::test]
    async fn test_synthesize_disabled() {
        let engine = TtsEngine::new(default_config());
        let result = engine.synthesize("Hello", None, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("disabled"));
    }

    #[tokio::test]
    async fn test_synthesize_empty_text() {
        let mut config = default_config();
        config.enabled = true;
        let engine = TtsEngine::new(config);
        let result = engine.synthesize("", None, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_synthesize_text_too_long() {
        let mut config = default_config();
        config.enabled = true;
        config.max_text_length = 10;
        let engine = TtsEngine::new(config);
        let result = engine
            .synthesize("This text is definitely longer than ten chars", None, None)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too long"));
    }

    #[test]
    fn test_detect_provider_none() {
        // In test env, likely no API keys set
        let _ = TtsEngine::detect_provider(); // Just verify no panic
    }

    #[tokio::test]
    async fn test_synthesize_no_provider() {
        let mut config = default_config();
        config.enabled = true;
        let engine = TtsEngine::new(config);
        // This may or may not error depending on env vars
        let result = engine.synthesize("Hello world", None, None).await;
        // If no API keys are set, should error
        if let Err(err) = result {
            assert!(err.contains("No TTS provider") || err.contains("not set"));
        }
    }

    #[tokio::test]
    async fn test_synthesize_unknown_provider_override() {
        let mut config = default_config();
        config.enabled = true;
        let engine = TtsEngine::new(config);
        let result = engine
            .synthesize_with_provider(Some("definitely-not-real"), "Hello", None, None)
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Unknown TTS provider: definitely-not-real"));
    }

    #[test]
    fn test_configured_provider_filters_auto_and_empty() {
        assert_eq!(configured_provider(Some("auto")), None);
        assert_eq!(configured_provider(Some("  ")), None);
        assert_eq!(configured_provider(Some("groq")), Some("groq"));
    }

    #[test]
    fn test_max_audio_constant() {
        assert_eq!(MAX_AUDIO_RESPONSE_BYTES, 10 * 1024 * 1024);
    }
}
