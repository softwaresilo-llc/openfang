//! Gemini CLI backend driver.
//!
//! Spawns `gemini --prompt ... --output-format json` as a subprocess and
//! converts the JSON result into the OpenFang `LlmDriver` response format.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError};
use async_trait::async_trait;
use openfang_types::message::{ContentBlock, Role, StopReason, TokenUsage};
use serde::Deserialize;
use std::collections::HashMap;
use tracing::debug;

/// LLM driver that delegates to the Gemini CLI.
pub struct GeminiCliDriver {
    cli_path: String,
}

impl GeminiCliDriver {
    /// Create a new Gemini CLI driver.
    ///
    /// `cli_path` overrides the binary path; defaults to `"gemini"` on PATH.
    pub fn new(cli_path: Option<String>) -> Self {
        Self {
            cli_path: cli_path
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "gemini".to_string()),
        }
    }

    /// Detect if the Gemini CLI is available on PATH.
    pub fn detect() -> Option<String> {
        let output = std::process::Command::new("gemini")
            .arg("--version")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
            .ok()?;

        if output.status.success() {
            Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            None
        }
    }

    /// Build a text prompt from completion messages.
    fn build_prompt(request: &CompletionRequest) -> String {
        let mut parts = Vec::new();

        if let Some(ref sys) = request.system {
            parts.push(format!("[System]\n{sys}"));
        }

        for msg in &request.messages {
            let role_label = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
            };
            let text = msg.content.text_content();
            if !text.is_empty() {
                parts.push(format!("[{role_label}]\n{text}"));
            }
        }

        parts.join("\n\n")
    }

    /// Map a model ID like "gemini-cli/gemini-3-flash-preview" to `--model` value.
    fn model_flag(model: &str) -> Option<String> {
        let stripped = model.strip_prefix("gemini-cli/").unwrap_or(model).trim();
        if stripped.is_empty() || stripped == "default" {
            None
        } else {
            Some(stripped.to_string())
        }
    }

    /// Extract the first JSON object payload from mixed log + JSON output.
    fn extract_json_payload(output: &str) -> Option<String> {
        let lines: Vec<&str> = output.lines().collect();
        for i in 0..lines.len() {
            if lines[i].trim_start().starts_with('{') {
                return Some(lines[i..].join("\n"));
            }
        }
        None
    }
}

#[derive(Debug, Deserialize, Default)]
struct GeminiCliOutput {
    #[serde(default)]
    response: String,
    #[serde(default)]
    stats: Option<GeminiCliStats>,
}

#[derive(Debug, Deserialize, Default)]
struct GeminiCliStats {
    #[serde(default)]
    models: HashMap<String, GeminiCliModelStats>,
}

#[derive(Debug, Deserialize, Default)]
struct GeminiCliModelStats {
    #[serde(default)]
    tokens: GeminiCliTokens,
}

#[derive(Debug, Deserialize, Default)]
struct GeminiCliTokens {
    #[serde(default)]
    input: u64,
    #[serde(default)]
    candidates: u64,
}

#[async_trait]
impl LlmDriver for GeminiCliDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let prompt = Self::build_prompt(&request);
        let prompt = if prompt.trim().is_empty() {
            " ".to_string()
        } else {
            prompt
        };

        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.arg("--prompt")
            .arg(prompt)
            .arg("--output-format")
            .arg("json")
            // Keep the CLI read-only from OpenFang's perspective.
            .arg("--approval-mode")
            .arg("plan");

        if let Some(model) = Self::model_flag(&request.model) {
            cmd.arg("--model").arg(model);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        debug!(cli = %self.cli_path, "Spawning Gemini CLI");

        let output = cmd
            .output()
            .await
            .map_err(|e| LlmError::Http(format!("Failed to spawn gemini CLI: {e}")))?;

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !output.status.success() {
            return Err(LlmError::Api {
                status: output.status.code().unwrap_or(1) as u16,
                message: format!("Gemini CLI failed: {stderr}"),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let payload = Self::extract_json_payload(&stdout).unwrap_or_else(|| stdout.to_string());
        let parsed = serde_json::from_str::<GeminiCliOutput>(&payload).unwrap_or_default();

        let mut usage = TokenUsage {
            input_tokens: 0,
            output_tokens: 0,
        };
        if let Some(stats) = parsed.stats {
            for model_stats in stats.models.values() {
                usage.input_tokens = usage.input_tokens.saturating_add(model_stats.tokens.input);
                usage.output_tokens = usage
                    .output_tokens
                    .saturating_add(model_stats.tokens.candidates);
            }
        }

        let text = if parsed.response.trim().is_empty() {
            stdout
                .lines()
                .last()
                .map(str::trim)
                .unwrap_or_default()
                .to_string()
        } else {
            parsed.response
        };

        Ok(CompletionResponse {
            content: vec![ContentBlock::Text {
                text,
                provider_metadata: None,
            }],
            stop_reason: StopReason::EndTurn,
            tool_calls: Vec::new(),
            usage,
        })
    }
}

/// Check if the Gemini CLI is available.
pub fn gemini_cli_available() -> bool {
    GeminiCliDriver::detect().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_flag_mapping() {
        assert_eq!(GeminiCliDriver::model_flag("gemini-cli/default"), None);
        assert_eq!(
            GeminiCliDriver::model_flag("gemini-cli/gemini-3-flash-preview"),
            Some("gemini-3-flash-preview".to_string())
        );
        assert_eq!(
            GeminiCliDriver::model_flag("gemini-2.5-pro"),
            Some("gemini-2.5-pro".to_string())
        );
    }

    #[test]
    fn test_extract_json_payload() {
        let mixed = "Loaded cached credentials.\n{\n  \"response\": \"OK\"\n}";
        let payload = GeminiCliDriver::extract_json_payload(mixed).unwrap();
        assert!(payload.contains("\"response\": \"OK\""));
    }
}
