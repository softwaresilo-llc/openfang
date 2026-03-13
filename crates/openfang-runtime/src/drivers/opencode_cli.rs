//! OpenCode CLI backend driver.
//!
//! Spawns `opencode run --format json` as a subprocess and converts JSONL
//! events into the OpenFang `LlmDriver` response format.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError};
use async_trait::async_trait;
use openfang_types::message::{ContentBlock, Role, StopReason, TokenUsage};
use serde::Deserialize;
use tracing::debug;

/// LLM driver that delegates to the OpenCode CLI.
pub struct OpenCodeCliDriver {
    cli_path: String,
}

impl OpenCodeCliDriver {
    /// Create a new OpenCode CLI driver.
    ///
    /// `cli_path` overrides the binary path; defaults to `"opencode"` on PATH.
    pub fn new(cli_path: Option<String>) -> Self {
        Self {
            cli_path: cli_path
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "opencode".to_string()),
        }
    }

    /// Detect if the OpenCode CLI is available on PATH.
    pub fn detect() -> Option<String> {
        let output = std::process::Command::new("opencode")
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

    /// Map a model ID like "opencode-cli/openai/gpt-4o" to OpenCode `--model` value.
    fn model_flag(model: &str) -> Option<String> {
        let stripped = model.strip_prefix("opencode-cli/").unwrap_or(model).trim();
        if stripped.is_empty() || stripped == "default" {
            None
        } else {
            Some(stripped.to_string())
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenCodeEvent {
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    part: Option<OpenCodePart>,
    #[serde(default)]
    error: String,
}

#[derive(Debug, Deserialize)]
struct OpenCodePart {
    #[serde(default)]
    text: String,
    #[serde(default)]
    tokens: Option<OpenCodeTokens>,
}

#[derive(Debug, Deserialize, Default)]
struct OpenCodeTokens {
    #[serde(default)]
    input: u64,
    #[serde(default)]
    output: u64,
}

#[async_trait]
impl LlmDriver for OpenCodeCliDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let prompt = Self::build_prompt(&request);
        let prompt = if prompt.trim().is_empty() {
            " ".to_string()
        } else {
            prompt
        };

        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.arg("run").arg(prompt).arg("--format").arg("json");

        if let Some(model) = Self::model_flag(&request.model) {
            cmd.arg("--model").arg(model);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        debug!(cli = %self.cli_path, "Spawning OpenCode CLI");

        let output = cmd
            .output()
            .await
            .map_err(|e| LlmError::Http(format!("Failed to spawn opencode CLI: {e}")))?;

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !output.status.success() {
            return Err(LlmError::Api {
                status: output.status.code().unwrap_or(1) as u16,
                message: format!("OpenCode CLI failed: {stderr}"),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut text_chunks = Vec::new();
        let mut usage = TokenUsage {
            input_tokens: 0,
            output_tokens: 0,
        };
        let mut error_messages = Vec::new();

        for line in stdout.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let Ok(event) = serde_json::from_str::<OpenCodeEvent>(line) else {
                continue;
            };

            match event.r#type.as_str() {
                "text" => {
                    if let Some(part) = event.part {
                        if !part.text.trim().is_empty() {
                            text_chunks.push(part.text);
                        }
                    }
                }
                "step_finish" => {
                    if let Some(part) = event.part {
                        if let Some(tokens) = part.tokens {
                            usage = TokenUsage {
                                input_tokens: tokens.input,
                                output_tokens: tokens.output,
                            };
                        }
                    }
                }
                "error" => {
                    if !event.error.trim().is_empty() {
                        error_messages.push(event.error);
                    }
                }
                _ => {}
            }
        }

        let mut text = text_chunks.join("\n").trim().to_string();
        if text.is_empty() {
            if !error_messages.is_empty() {
                return Err(LlmError::Api {
                    status: 500,
                    message: format!(
                        "OpenCode CLI returned errors: {}",
                        error_messages.join("; ")
                    ),
                });
            }
            text = stdout
                .lines()
                .last()
                .map(str::trim)
                .unwrap_or_default()
                .to_string();
        }

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

/// Check if the OpenCode CLI is available.
pub fn opencode_cli_available() -> bool {
    OpenCodeCliDriver::detect().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_flag_mapping() {
        assert_eq!(OpenCodeCliDriver::model_flag("opencode-cli/default"), None);
        assert_eq!(
            OpenCodeCliDriver::model_flag("opencode-cli/openai/gpt-4o"),
            Some("openai/gpt-4o".to_string())
        );
        assert_eq!(
            OpenCodeCliDriver::model_flag("groq/llama-3.3-70b"),
            Some("groq/llama-3.3-70b".to_string())
        );
    }

    #[test]
    fn test_new_defaults_to_opencode() {
        let driver = OpenCodeCliDriver::new(None);
        assert_eq!(driver.cli_path, "opencode");
    }
}
