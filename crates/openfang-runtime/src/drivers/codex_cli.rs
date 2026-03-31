//! Codex CLI backend driver.
//!
//! Spawns `codex exec --json` as a subprocess and converts JSONL events into
//! the OpenFang `LlmDriver` response format.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError};
use async_trait::async_trait;
use openfang_types::message::{ContentBlock, Role, StopReason, TokenUsage};
use serde::Deserialize;
use tracing::debug;

/// LLM driver that delegates to the Codex CLI.
pub struct CodexCliDriver {
    cli_path: String,
}

impl CodexCliDriver {
    /// Create a new Codex CLI driver.
    ///
    /// `cli_path` overrides the binary path; defaults to `"codex"` on PATH.
    pub fn new(cli_path: Option<String>) -> Self {
        Self {
            cli_path: cli_path
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "codex".to_string()),
        }
    }

    /// Detect if the Codex CLI is available on PATH.
    pub fn detect() -> Option<String> {
        let output = std::process::Command::new("codex")
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

    /// Map a model ID like "codex-cli/o4-mini" to Codex `--model` value.
    fn model_flag(model: &str) -> Option<String> {
        let stripped = model.strip_prefix("codex-cli/").unwrap_or(model).trim();
        if stripped.is_empty() || stripped == "default" {
            None
        } else {
            Some(stripped.to_string())
        }
    }
}

#[derive(Debug, Deserialize)]
struct CodexEvent {
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    item: Option<CodexItem>,
    #[serde(default)]
    usage: Option<CodexUsage>,
}

#[derive(Debug, Deserialize)]
struct CodexItem {
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    message: String,
}

#[derive(Debug, Deserialize, Default)]
struct CodexUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

#[async_trait]
impl LlmDriver for CodexCliDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let prompt = Self::build_prompt(&request);
        let prompt = if prompt.trim().is_empty() {
            " ".to_string()
        } else {
            prompt
        };

        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.arg("exec")
            .arg("--json")
            .arg("--skip-git-repo-check")
            .arg("--full-auto");

        if let Some(model) = Self::model_flag(&request.model) {
            cmd.arg("--model").arg(model);
        }

        cmd.arg(prompt);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        debug!(cli = %self.cli_path, "Spawning Codex CLI");

        let output = cmd
            .output()
            .await
            .map_err(|e| LlmError::Http(format!("Failed to spawn codex CLI: {e}")))?;

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !output.status.success() {
            return Err(LlmError::Api {
                status: output.status.code().unwrap_or(1) as u16,
                message: format!("Codex CLI failed: {stderr}"),
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
            let Ok(event) = serde_json::from_str::<CodexEvent>(line) else {
                continue;
            };

            match event.r#type.as_str() {
                "item.completed" => {
                    if let Some(item) = event.item {
                        if item.r#type == "agent_message" && !item.text.trim().is_empty() {
                            text_chunks.push(item.text);
                        } else if item.r#type == "error" && !item.message.trim().is_empty() {
                            error_messages.push(item.message);
                        }
                    }
                }
                "turn.completed" => {
                    if let Some(u) = event.usage {
                        usage = TokenUsage {
                            input_tokens: u.input_tokens,
                            output_tokens: u.output_tokens,
                        };
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
                    message: format!("Codex CLI returned errors: {}", error_messages.join("; ")),
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

/// Check if the Codex CLI is available.
pub fn codex_cli_available() -> bool {
    CodexCliDriver::detect().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_flag_mapping() {
        assert_eq!(CodexCliDriver::model_flag("codex-cli/default"), None);
        assert_eq!(
            CodexCliDriver::model_flag("codex-cli/o4-mini"),
            Some("o4-mini".to_string())
        );
        assert_eq!(
            CodexCliDriver::model_flag("gpt-5"),
            Some("gpt-5".to_string())
        );
    }

    #[test]
    fn test_new_defaults_to_codex() {
        let driver = CodexCliDriver::new(None);
        assert_eq!(driver.cli_path, "codex");
    }
}
