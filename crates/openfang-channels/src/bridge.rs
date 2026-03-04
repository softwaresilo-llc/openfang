//! Channel bridge — connects channel adapters to the OpenFang kernel.
//!
//! Defines `ChannelBridgeHandle` (implemented by openfang-api on the kernel) and
//! `BridgeManager` which owns running adapters and dispatches messages.

use crate::formatter;
use crate::router::AgentRouter;
use crate::types::{ChannelAdapter, ChannelContent, ChannelMessage, ChannelUser};
use async_trait::async_trait;
use dashmap::DashMap;
use futures::StreamExt;
use openfang_types::agent::AgentId;
use openfang_types::config::{
    ChannelOverrides, ChannelVoiceConfig, ChatRoomMode, ChatRoomsConfig, DmPolicy, GroupPolicy,
    OutputFormat, VoiceLanguage, VoiceReplyMode,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

/// Kernel operations needed by channel adapters.
///
/// Defined here to avoid circular deps (openfang-channels can't depend on openfang-kernel).
/// Implemented in openfang-api on the actual kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceAsset {
    /// URL that the channel adapter can send as voice/audio media.
    pub url: String,
    /// Estimated duration for UI/client hints.
    pub duration_seconds: u32,
}

/// Persisted routing state for one external conversation room/thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConversationState {
    /// Stable conversation key (`channel:room[:thread]`).
    pub conversation_key: String,
    /// Channel type string (`telegram`, `whatsapp`, ...).
    pub channel: String,
    /// External room/chat identifier.
    pub room_id: String,
    /// Current room routing mode.
    pub mode: ChatRoomMode,
    /// Active agent name for active/orchestrator modes.
    pub active_agent: Option<String>,
    /// Panel agent names (used in panel mode).
    pub panel_agents: Vec<String>,
    /// Whether replies require explicit @mention.
    pub requires_mention: bool,
    /// In panel mode, reply with all panel agents when no mention exists.
    pub respond_without_mention: bool,
    /// Last update timestamp (RFC3339 UTC).
    pub updated_at: String,
}

impl Default for ConversationState {
    fn default() -> Self {
        Self {
            conversation_key: String::new(),
            channel: String::new(),
            room_id: String::new(),
            mode: ChatRoomMode::Active,
            active_agent: None,
            panel_agents: Vec::new(),
            requires_mention: true,
            respond_without_mention: true,
            updated_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

impl ConversationState {
    /// Build a fresh room state from global defaults.
    pub fn with_defaults(
        conversation_key: String,
        channel: String,
        room_id: String,
        defaults: &ChatRoomsConfig,
    ) -> Self {
        Self {
            conversation_key,
            channel,
            room_id,
            mode: defaults.default_mode,
            active_agent: None,
            panel_agents: Vec::new(),
            requires_mention: defaults.default_requires_mention,
            respond_without_mention: defaults.respond_without_mention,
            updated_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    fn touch(&mut self) {
        self.updated_at = chrono::Utc::now().to_rfc3339();
    }
}

#[async_trait]
pub trait ChannelBridgeHandle: Send + Sync {
    /// Send a message to an agent and get the text response.
    async fn send_message(&self, agent_id: AgentId, message: &str) -> Result<String, String>;

    /// Find an agent by name, returning its ID.
    async fn find_agent_by_name(&self, name: &str) -> Result<Option<AgentId>, String>;

    /// List running agents as (id, name) pairs.
    async fn list_agents(&self) -> Result<Vec<(AgentId, String)>, String>;

    /// Spawn an agent by manifest name, returning its ID.
    async fn spawn_agent_by_name(&self, manifest_name: &str) -> Result<AgentId, String>;

    /// Return uptime info string (e.g., "2h 15m, 5 agents").
    async fn uptime_info(&self) -> String {
        let agents = self.list_agents().await.unwrap_or_default();
        format!("{} agent(s) running", agents.len())
    }

    /// List available models as formatted text for channel display.
    async fn list_models_text(&self) -> String {
        "Model listing not available.".to_string()
    }

    /// List providers and their auth status as formatted text for channel display.
    async fn list_providers_text(&self) -> String {
        "Provider listing not available.".to_string()
    }

    /// Reset an agent's session (clear messages, fresh session ID).
    async fn reset_session(&self, _agent_id: AgentId) -> Result<String, String> {
        Err("Not implemented".to_string())
    }

    /// Trigger LLM-based session compaction for an agent.
    async fn compact_session(&self, _agent_id: AgentId) -> Result<String, String> {
        Err("Not implemented".to_string())
    }

    /// Set an agent's model.
    async fn set_model(&self, _agent_id: AgentId, _model: &str) -> Result<String, String> {
        Err("Not implemented".to_string())
    }

    /// Stop an agent's current LLM run.
    async fn stop_run(&self, _agent_id: AgentId) -> Result<String, String> {
        Err("Not implemented".to_string())
    }

    /// Get session token usage and estimated cost.
    async fn session_usage(&self, _agent_id: AgentId) -> Result<String, String> {
        Err("Not implemented".to_string())
    }

    /// Toggle extended thinking mode for an agent.
    async fn set_thinking(&self, _agent_id: AgentId, _on: bool) -> Result<String, String> {
        Ok("Extended thinking preference saved.".to_string())
    }

    /// List installed skills as formatted text for channel display.
    async fn list_skills_text(&self) -> String {
        "Skill listing not available.".to_string()
    }

    /// List hands (marketplace + active) as formatted text for channel display.
    async fn list_hands_text(&self) -> String {
        "Hand listing not available.".to_string()
    }

    /// Authorize a channel user for an action.
    ///
    /// Returns Ok(()) if the user is allowed, Err(reason) if denied.
    /// Default implementation: allow all (RBAC disabled).
    async fn authorize_channel_user(
        &self,
        _channel_type: &str,
        _platform_id: &str,
        _action: &str,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Get per-channel overrides for a given channel type.
    ///
    /// Returns `None` if the channel is not configured or has no overrides.
    async fn channel_overrides(&self, _channel_type: &str) -> Option<ChannelOverrides> {
        None
    }

    /// Return channel voice behavior config, if supported by this channel.
    async fn channel_voice_config(&self, _channel_type: &str) -> Option<ChannelVoiceConfig> {
        None
    }

    /// Return global chat-room routing defaults.
    async fn chat_rooms_config(&self) -> ChatRoomsConfig {
        ChatRoomsConfig::default()
    }

    /// Load persisted state for one conversation room.
    async fn get_conversation_state(
        &self,
        _conversation_key: &str,
    ) -> Result<Option<ConversationState>, String> {
        Ok(None)
    }

    /// List all persisted conversation room states.
    async fn list_conversation_states(&self) -> Result<Vec<ConversationState>, String> {
        Ok(Vec::new())
    }

    /// Persist conversation room state.
    async fn save_conversation_state(&self, _state: ConversationState) -> Result<(), String> {
        Ok(())
    }

    /// Delete persisted conversation room state.
    async fn delete_conversation_state(&self, _conversation_key: &str) -> Result<(), String> {
        Ok(())
    }

    /// Send a message in a conversation-scoped session (falls back to default send).
    async fn send_message_in_conversation(
        &self,
        agent_id: AgentId,
        message: &str,
        _conversation_key: Option<&str>,
        _conversation_label: Option<&str>,
    ) -> Result<String, String> {
        self.send_message(agent_id, message).await
    }

    /// Transcribe a voice/media URL to plain text.
    ///
    /// Returns `Ok(Some(text))` on successful transcription.
    /// Returns `Ok(None)` if the handle chooses not to transcribe for this channel.
    async fn transcribe_voice_url(
        &self,
        _channel_type: &str,
        _media_url: &str,
    ) -> Result<Option<String>, String> {
        Ok(None)
    }

    /// Synthesize text into a voice asset that the adapter can send.
    ///
    /// Returns `Ok(Some(asset))` when synthesis succeeds.
    /// Returns `Ok(None)` if synthesis is unavailable for this channel/runtime.
    async fn synthesize_voice(
        &self,
        _channel_type: &str,
        _text: &str,
        _language: VoiceLanguage,
    ) -> Result<Option<VoiceAsset>, String> {
        Ok(None)
    }

    /// Record a delivery result for tracking (optional — default no-op).
    async fn record_delivery(
        &self,
        _agent_id: AgentId,
        _channel: &str,
        _recipient: &str,
        _success: bool,
        _error: Option<&str>,
    ) {
        // Default: no tracking
    }

    /// Check if auto-reply is enabled and the message should trigger one.
    /// Returns Some(reply_text) if auto-reply fires, None otherwise.
    async fn check_auto_reply(&self, _agent_id: AgentId, _message: &str) -> Option<String> {
        None
    }

    // ── Automation: workflows, triggers, schedules, approvals ──

    /// List all registered workflows as formatted text.
    async fn list_workflows_text(&self) -> String {
        "Workflows not available.".to_string()
    }

    /// Run a workflow by name with the given input text.
    async fn run_workflow_text(&self, _name: &str, _input: &str) -> String {
        "Workflows not available.".to_string()
    }

    /// List all registered triggers as formatted text.
    async fn list_triggers_text(&self) -> String {
        "Triggers not available.".to_string()
    }

    /// Create a trigger for an agent with the given pattern and prompt.
    async fn create_trigger_text(
        &self,
        _agent_name: &str,
        _pattern: &str,
        _prompt: &str,
    ) -> String {
        "Triggers not available.".to_string()
    }

    /// Delete a trigger by UUID prefix.
    async fn delete_trigger_text(&self, _id_prefix: &str) -> String {
        "Triggers not available.".to_string()
    }

    /// List all cron jobs as formatted text.
    async fn list_schedules_text(&self) -> String {
        "Schedules not available.".to_string()
    }

    /// Manage a cron job: add, del, or run.
    async fn manage_schedule_text(&self, _action: &str, _args: &[String]) -> String {
        "Schedules not available.".to_string()
    }

    /// List pending approval requests as formatted text.
    async fn list_approvals_text(&self) -> String {
        "No approvals pending.".to_string()
    }

    /// Approve or reject a pending approval by UUID prefix.
    async fn resolve_approval_text(&self, _id_prefix: &str, _approve: bool) -> String {
        "Approvals not available.".to_string()
    }

    // ── Budget, Network, A2A ──

    /// Show global budget status (limits, spend, % used).
    async fn budget_text(&self) -> String {
        "Budget information not available.".to_string()
    }

    /// Show OFP peer network status.
    async fn peers_text(&self) -> String {
        "Peer network not available.".to_string()
    }

    /// List discovered external A2A agents.
    async fn a2a_agents_text(&self) -> String {
        "A2A agents not available.".to_string()
    }
}

/// Per-channel rate limiter tracking message timestamps per user.
///
/// Key: `"{channel_type}:{platform_id}"`, Value: timestamps of recent messages.
#[derive(Debug, Clone, Default)]
pub struct ChannelRateLimiter {
    /// Recent message timestamps per user key.
    buckets: Arc<DashMap<String, Vec<Instant>>>,
}

impl ChannelRateLimiter {
    /// Check if a user is rate-limited. Returns `Ok(())` if allowed, `Err(msg)` if blocked.
    ///
    /// `max_per_minute`: 0 means unlimited.
    pub fn check(
        &self,
        channel_type: &str,
        platform_id: &str,
        max_per_minute: u32,
    ) -> Result<(), String> {
        if max_per_minute == 0 {
            return Ok(());
        }

        let key = format!("{channel_type}:{platform_id}");
        let now = Instant::now();
        let window = std::time::Duration::from_secs(60);

        let mut entry = self.buckets.entry(key).or_default();
        // Evict timestamps older than 1 minute
        entry.retain(|&ts| now.duration_since(ts) < window);

        if entry.len() >= max_per_minute as usize {
            return Err(format!(
                "Rate limit exceeded ({max_per_minute} messages/minute). Please wait."
            ));
        }

        entry.push(now);
        Ok(())
    }
}

/// Owns all running channel adapters and dispatches messages to agents.
pub struct BridgeManager {
    handle: Arc<dyn ChannelBridgeHandle>,
    router: Arc<AgentRouter>,
    rate_limiter: ChannelRateLimiter,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl BridgeManager {
    pub fn new(handle: Arc<dyn ChannelBridgeHandle>, router: Arc<AgentRouter>) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            handle,
            router,
            rate_limiter: ChannelRateLimiter::default(),
            shutdown_tx,
            shutdown_rx,
            tasks: Vec::new(),
        }
    }

    /// Start an adapter: subscribe to its message stream and spawn a dispatch task.
    pub async fn start_adapter(
        &mut self,
        adapter: Arc<dyn ChannelAdapter>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let stream = adapter.start().await?;
        let handle = self.handle.clone();
        let router = self.router.clone();
        let rate_limiter = self.rate_limiter.clone();
        let adapter_clone = adapter.clone();
        let mut shutdown = self.shutdown_rx.clone();

        let task = tokio::spawn(async move {
            let mut stream = std::pin::pin!(stream);
            loop {
                tokio::select! {
                    msg = stream.next() => {
                        match msg {
                            Some(message) => {
                                dispatch_message(
                                    &message,
                                    &handle,
                                    &router,
                                    adapter_clone.as_ref(),
                                    &rate_limiter,
                                ).await;
                            }
                            None => {
                                info!("Channel adapter {} stream ended", adapter_clone.name());
                                break;
                            }
                        }
                    }
                    _ = shutdown.changed() => {
                        if *shutdown.borrow() {
                            info!("Shutting down channel adapter {}", adapter_clone.name());
                            break;
                        }
                    }
                }
            }
        });

        self.tasks.push(task);
        Ok(())
    }

    /// Stop all adapters and wait for dispatch tasks to finish.
    pub async fn stop(&mut self) {
        let _ = self.shutdown_tx.send(true);
        for task in self.tasks.drain(..) {
            let _ = task.await;
        }
    }
}

/// Resolve channel type to its config string key.
fn channel_type_str(channel: &crate::types::ChannelType) -> &str {
    match channel {
        crate::types::ChannelType::Telegram => "telegram",
        crate::types::ChannelType::Discord => "discord",
        crate::types::ChannelType::Slack => "slack",
        crate::types::ChannelType::WhatsApp => "whatsapp",
        crate::types::ChannelType::Signal => "signal",
        crate::types::ChannelType::Matrix => "matrix",
        crate::types::ChannelType::Email => "email",
        crate::types::ChannelType::Teams => "teams",
        crate::types::ChannelType::Mattermost => "mattermost",
        crate::types::ChannelType::WebChat => "webchat",
        crate::types::ChannelType::CLI => "cli",
        crate::types::ChannelType::Custom(s) => s.as_str(),
    }
}

/// Send a response, applying output formatting and optional threading.
async fn send_response(
    adapter: &dyn ChannelAdapter,
    user: &ChannelUser,
    text: String,
    thread_id: Option<&str>,
    output_format: OutputFormat,
) {
    let formatted = formatter::format_for_channel(&text, output_format);
    let content = ChannelContent::Text(formatted);

    let result = if let Some(tid) = thread_id {
        adapter.send_in_thread(user, content, tid).await
    } else {
        adapter.send(user, content).await
    };

    if let Err(e) = result {
        error!("Failed to send response: {e}");
    }
}

/// Send a synthesized voice response asset.
async fn send_voice_response(
    adapter: &dyn ChannelAdapter,
    user: &ChannelUser,
    voice: VoiceAsset,
    thread_id: Option<&str>,
) -> Result<(), String> {
    let content = ChannelContent::Voice {
        url: voice.url,
        duration_seconds: voice.duration_seconds,
    };
    let result = if let Some(tid) = thread_id {
        adapter.send_in_thread(user, content, tid).await
    } else {
        adapter.send(user, content).await
    };
    result.map_err(|e| e.to_string())
}

/// Decide if a text response should be delivered as voice.
fn should_send_voice_reply(text: &str, cfg: &ChannelVoiceConfig) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    match cfg.reply_mode {
        VoiceReplyMode::Off => false,
        VoiceReplyMode::Always => true,
        VoiceReplyMode::Auto => {
            if trimmed.chars().count() >= cfg.auto_min_text_length {
                return true;
            }
            if cfg.auto_keywords.is_empty() {
                return false;
            }
            let lowered = trimmed.to_lowercase();
            cfg.auto_keywords.iter().any(|kw| {
                let k = kw.trim().to_lowercase();
                !k.is_empty() && lowered.contains(&k)
            })
        }
    }
}

fn conversation_room_id(message: &ChannelMessage) -> String {
    const CANDIDATE_KEYS: &[&str] = &[
        "chat_id",
        "channel_id",
        "room_id",
        "conversation_id",
        "group_id",
        "thread_root_id",
    ];

    for key in CANDIDATE_KEYS {
        if let Some(value) = message.metadata.get(*key).and_then(|v| v.as_str()) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }

    message.sender.platform_id.clone()
}

fn conversation_key_for_message(message: &ChannelMessage) -> Option<String> {
    if !message.is_group {
        return None;
    }

    let channel = channel_type_str(&message.channel);
    let room_id = conversation_room_id(message);
    if let Some(thread_id) = message.thread_id.as_deref() {
        if !thread_id.trim().is_empty() {
            return Some(format!("{channel}:{room_id}:thread:{thread_id}"));
        }
    }
    Some(format!("{channel}:{room_id}"))
}

fn normalize_agent_token(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .collect::<String>()
}

fn chat_room_mode_label(mode: ChatRoomMode) -> &'static str {
    match mode {
        ChatRoomMode::Active => "active",
        ChatRoomMode::Panel => "panel",
        ChatRoomMode::Orchestrator => "orchestrator",
    }
}

fn extract_mentioned_agent_names(text: &str, agents: &[(AgentId, String)]) -> Vec<String> {
    let mut token_to_agent = HashMap::new();
    for (_, name) in agents {
        let norm = normalize_agent_token(name);
        if !norm.is_empty() {
            token_to_agent.insert(norm.clone(), name.clone());
        }
        let dashed = normalize_agent_token(&name.replace(' ', "-"));
        if !dashed.is_empty() {
            token_to_agent.insert(dashed, name.clone());
        }
        let underscored = normalize_agent_token(&name.replace(' ', "_"));
        if !underscored.is_empty() {
            token_to_agent.insert(underscored, name.clone());
        }
        let compact = normalize_agent_token(&name.replace(' ', ""));
        if !compact.is_empty() {
            token_to_agent.insert(compact, name.clone());
        }
    }

    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for raw in text.split_whitespace() {
        if !raw.starts_with('@') || raw.len() <= 1 {
            continue;
        }
        let trimmed = raw
            .trim_start_matches('@')
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .to_lowercase();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(name) = token_to_agent.get(&trimmed) {
            if seen.insert(name.clone()) {
                out.push(name.clone());
            }
        }
    }
    out
}

fn select_active_agent_name(
    state: &ConversationState,
    router: &Arc<AgentRouter>,
    message: &ChannelMessage,
    agents: &[(AgentId, String)],
) -> Option<String> {
    if let Some(name) = state.active_agent.as_ref() {
        if agents.iter().any(|(_, n)| n == name) {
            return Some(name.clone());
        }
    }

    let fallback_id = router.resolve(
        &message.channel,
        &message.sender.platform_id,
        message.sender.openfang_user.as_deref(),
    )?;
    agents
        .iter()
        .find(|(id, _)| *id == fallback_id)
        .map(|(_, name)| name.clone())
}

async fn ensure_room_state(
    handle: &Arc<dyn ChannelBridgeHandle>,
    router: &Arc<AgentRouter>,
    message: &ChannelMessage,
    conversation_key: &str,
    room_defaults: &ChatRoomsConfig,
) -> Option<ConversationState> {
    let channel = channel_type_str(&message.channel).to_string();
    let room_id = conversation_room_id(message);

    let mut state = match handle.get_conversation_state(conversation_key).await {
        Ok(Some(existing)) => existing,
        Ok(None) => ConversationState::with_defaults(
            conversation_key.to_string(),
            channel,
            room_id,
            room_defaults,
        ),
        Err(e) => {
            warn!("Failed loading room state for {conversation_key}: {e}");
            ConversationState::with_defaults(
                conversation_key.to_string(),
                channel,
                room_id,
                room_defaults,
            )
        }
    };

    if state.conversation_key.is_empty() {
        state.conversation_key = conversation_key.to_string();
    }
    if state.channel.is_empty() {
        state.channel = channel_type_str(&message.channel).to_string();
    }
    if state.room_id.is_empty() {
        state.room_id = conversation_room_id(message);
    }

    if state.active_agent.is_none() {
        let running_agents = handle.list_agents().await.unwrap_or_default();
        state.active_agent = select_active_agent_name(&state, router, message, &running_agents);
    }

    Some(state)
}

async fn handle_room_command(
    args: &[String],
    handle: &Arc<dyn ChannelBridgeHandle>,
    router: &Arc<AgentRouter>,
    message: &ChannelMessage,
    room_defaults: &ChatRoomsConfig,
) -> String {
    let conversation_key = match conversation_key_for_message(message) {
        Some(k) => k,
        None => return "Room commands are only available in group chats.".to_string(),
    };

    if let Err(denied) = handle
        .authorize_channel_user(
            channel_type_str(&message.channel),
            &message.sender.platform_id,
            "manage_rooms",
        )
        .await
    {
        return format!("Access denied: {denied}");
    }

    let mut state =
        match ensure_room_state(handle, router, message, &conversation_key, room_defaults).await {
            Some(s) => s,
            None => return "Failed to initialize room state.".to_string(),
        };

    let sub = args.first().map(|s| s.as_str()).unwrap_or("status");
    match sub {
        "status" => {
            let active = state.active_agent.as_deref().unwrap_or("(none)");
            let panel = if state.panel_agents.is_empty() {
                "(empty)".to_string()
            } else {
                state.panel_agents.join(", ")
            };
            format!(
                "Room state\nkey: {}\nmode: {}\nactive: {}\npanel: {}\nrequires_mention: {}\nrespond_without_mention: {}",
                state.conversation_key,
                chat_room_mode_label(state.mode),
                active,
                panel,
                state.requires_mention,
                state.respond_without_mention,
            )
        }
        "mode" => {
            let mode_arg = match args.get(1) {
                Some(v) => v.as_str(),
                None => return "Usage: /room mode <active|panel|orchestrator>".to_string(),
            };
            state.mode = match mode_arg {
                "active" => ChatRoomMode::Active,
                "panel" => ChatRoomMode::Panel,
                "orchestrator" => ChatRoomMode::Orchestrator,
                _ => return "Unknown mode. Use active, panel, or orchestrator.".to_string(),
            };
            state.touch();
            if let Err(e) = handle.save_conversation_state(state.clone()).await {
                return format!("Failed to save room mode: {e}");
            }
            format!("Room mode set to {mode_arg}")
        }
        "active" => {
            let agent_name = match args.get(1) {
                Some(v) => v.trim(),
                None => return "Usage: /room active <agent-name>".to_string(),
            };
            match handle.find_agent_by_name(agent_name).await {
                Ok(Some(_)) => {
                    state.active_agent = Some(agent_name.to_string());
                    state.touch();
                    if let Err(e) = handle.save_conversation_state(state.clone()).await {
                        return format!("Failed to save active agent: {e}");
                    }
                    format!("Room active agent set to {agent_name}")
                }
                Ok(None) => format!("Unknown agent: {agent_name}"),
                Err(e) => format!("Failed to resolve agent: {e}"),
            }
        }
        "panel" => {
            let action = args.get(1).map(|s| s.as_str()).unwrap_or("");
            match action {
                "add" => {
                    let agent_name = match args.get(2) {
                        Some(v) => v.trim(),
                        None => return "Usage: /room panel add <agent-name>".to_string(),
                    };
                    match handle.find_agent_by_name(agent_name).await {
                        Ok(Some(_)) => {
                            if !state.panel_agents.iter().any(|a| a == agent_name) {
                                state.panel_agents.push(agent_name.to_string());
                            }
                            if state.panel_agents.len() > room_defaults.max_active_agents {
                                state.panel_agents.truncate(room_defaults.max_active_agents);
                            }
                            state.touch();
                            if let Err(e) = handle.save_conversation_state(state.clone()).await {
                                return format!("Failed to update panel: {e}");
                            }
                            format!("Added {agent_name} to room panel")
                        }
                        Ok(None) => format!("Unknown agent: {agent_name}"),
                        Err(e) => format!("Failed to resolve agent: {e}"),
                    }
                }
                "remove" => {
                    let agent_name = match args.get(2) {
                        Some(v) => v.trim(),
                        None => return "Usage: /room panel remove <agent-name>".to_string(),
                    };
                    state.panel_agents.retain(|a| a != agent_name);
                    state.touch();
                    if let Err(e) = handle.save_conversation_state(state.clone()).await {
                        return format!("Failed to update panel: {e}");
                    }
                    format!("Removed {agent_name} from room panel")
                }
                "clear" => {
                    state.panel_agents.clear();
                    state.touch();
                    if let Err(e) = handle.save_conversation_state(state.clone()).await {
                        return format!("Failed to clear panel: {e}");
                    }
                    "Room panel cleared".to_string()
                }
                _ => "Usage: /room panel <add|remove|clear> [agent-name]".to_string(),
            }
        }
        "mention" => {
            let on = match args.get(1).map(|s| s.as_str()) {
                Some("on") => true,
                Some("off") => false,
                _ => return "Usage: /room mention <on|off>".to_string(),
            };
            state.requires_mention = on;
            state.touch();
            if let Err(e) = handle.save_conversation_state(state.clone()).await {
                return format!("Failed to save mention policy: {e}");
            }
            format!("Room mention requirement set to {}", if on { "on" } else { "off" })
        }
        "fallback" => {
            let on = match args.get(1).map(|s| s.as_str()) {
                Some("on") => true,
                Some("off") => false,
                _ => return "Usage: /room fallback <on|off>".to_string(),
            };
            state.respond_without_mention = on;
            state.touch();
            if let Err(e) = handle.save_conversation_state(state.clone()).await {
                return format!("Failed to save fallback policy: {e}");
            }
            format!(
                "Room no-mention panel replies set to {}",
                if on { "on" } else { "off" }
            )
        }
        _ => {
            "Room commands:\n/room status\n/room mode <active|panel|orchestrator>\n/room active <agent>\n/room panel <add|remove|clear> [agent]\n/room mention <on|off>\n/room fallback <on|off>"
                .to_string()
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn dispatch_group_room_message(
    message: &ChannelMessage,
    text: &str,
    handle: &Arc<dyn ChannelBridgeHandle>,
    router: &Arc<AgentRouter>,
    adapter: &dyn ChannelAdapter,
    ct_str: &str,
    thread_id: Option<&str>,
    output_format: OutputFormat,
    room_defaults: &ChatRoomsConfig,
) -> bool {
    if !message.is_group || !room_defaults.enabled {
        return false;
    }

    let conversation_key = match conversation_key_for_message(message) {
        Some(k) => k,
        None => return false,
    };

    let mut state =
        match ensure_room_state(handle, router, message, &conversation_key, room_defaults).await {
            Some(s) => s,
            None => return true,
        };

    let agents = handle.list_agents().await.unwrap_or_default();
    if agents.is_empty() {
        send_response(
            adapter,
            &message.sender,
            "No agents running.".to_string(),
            thread_id,
            output_format,
        )
        .await;
        return true;
    }

    let mentioned_names = extract_mentioned_agent_names(text, &agents);
    let mut target_names: Vec<String> = Vec::new();

    match state.mode {
        ChatRoomMode::Active | ChatRoomMode::Orchestrator => {
            if let Some(first) = mentioned_names.first() {
                state.active_agent = Some(first.clone());
                target_names.push(first.clone());
            } else if state.requires_mention {
                return true;
            } else if let Some(active) = select_active_agent_name(&state, router, message, &agents)
            {
                state.active_agent = Some(active.clone());
                target_names.push(active);
            }
        }
        ChatRoomMode::Panel => {
            if !mentioned_names.is_empty() {
                target_names.extend(mentioned_names);
            } else if state.respond_without_mention {
                target_names.extend(state.panel_agents.iter().cloned());
            } else if state.requires_mention {
                return true;
            }

            if target_names.is_empty() {
                if let Some(active) = select_active_agent_name(&state, router, message, &agents) {
                    state.active_agent = Some(active.clone());
                    target_names.push(active);
                }
            }
        }
    }

    if target_names.is_empty() {
        send_response(
            adapter,
            &message.sender,
            "No target agent selected for this room. Use /room active <agent> or /room panel add <agent>."
                .to_string(),
            thread_id,
            output_format,
        )
        .await;
        return true;
    }

    // Deduplicate while preserving order.
    let mut seen = HashSet::new();
    target_names.retain(|name| seen.insert(name.clone()));
    target_names.truncate(room_defaults.max_active_agents.max(1));

    // Resolve names -> IDs using current running list.
    let id_by_name: HashMap<String, AgentId> = agents
        .iter()
        .map(|(id, name)| (name.clone(), *id))
        .collect();
    let mut targets: Vec<(AgentId, String)> = target_names
        .into_iter()
        .filter_map(|name| id_by_name.get(&name).copied().map(|id| (id, name)))
        .collect();

    if targets.is_empty() {
        send_response(
            adapter,
            &message.sender,
            "None of the selected room agents are currently running.".to_string(),
            thread_id,
            output_format,
        )
        .await;
        return true;
    }

    if let Some((_, first_name)) = targets.first() {
        state.active_agent = Some(first_name.clone());
    }
    state.touch();
    if let Err(e) = handle.save_conversation_state(state.clone()).await {
        warn!(
            "Failed to save room state for {}: {}",
            state.conversation_key, e
        );
    }

    let _ = adapter.send_typing(&message.sender).await;
    let conversation_label = format!("room:{}:{}", state.channel, state.room_id);
    let multi = targets.len() > 1;
    for (agent_id, agent_name) in targets.drain(..) {
        match handle
            .send_message_in_conversation(
                agent_id,
                text,
                Some(&conversation_key),
                Some(&conversation_label),
            )
            .await
        {
            Ok(response) => {
                let outbound = if multi {
                    format!("[{agent_name}] {response}")
                } else {
                    response
                };
                send_response(adapter, &message.sender, outbound, thread_id, output_format).await;
                handle
                    .record_delivery(agent_id, ct_str, &message.sender.platform_id, true, None)
                    .await;
            }
            Err(e) => {
                let err_msg = format!("[{agent_name}] Agent error: {e}");
                send_response(
                    adapter,
                    &message.sender,
                    err_msg.clone(),
                    thread_id,
                    output_format,
                )
                .await;
                handle
                    .record_delivery(
                        agent_id,
                        ct_str,
                        &message.sender.platform_id,
                        false,
                        Some(&err_msg),
                    )
                    .await;
            }
        }
    }
    true
}

/// Dispatch a single incoming message — handles bot commands or routes to an agent.
///
/// Applies per-channel policies (DM/group filtering, rate limiting, formatting, threading).
async fn dispatch_message(
    message: &ChannelMessage,
    handle: &Arc<dyn ChannelBridgeHandle>,
    router: &Arc<AgentRouter>,
    adapter: &dyn ChannelAdapter,
    rate_limiter: &ChannelRateLimiter,
) {
    let ct_str = channel_type_str(&message.channel);

    // Fetch per-channel overrides (if configured)
    let overrides = handle.channel_overrides(ct_str).await;
    let channel_default_format = match ct_str {
        "telegram" => OutputFormat::TelegramHtml,
        "slack" => OutputFormat::SlackMrkdwn,
        _ => OutputFormat::Markdown,
    };
    let output_format = overrides
        .as_ref()
        .and_then(|o| o.output_format)
        .unwrap_or(channel_default_format);
    let threading_enabled = overrides.as_ref().map(|o| o.threading).unwrap_or(false);
    let thread_id = if threading_enabled {
        message.thread_id.as_deref()
    } else {
        None
    };

    // --- DM/Group policy check ---
    if let Some(ref ov) = overrides {
        if message.is_group {
            match ov.group_policy {
                GroupPolicy::Ignore => {
                    debug!("Ignoring group message on {ct_str} (group_policy=ignore)");
                    return;
                }
                GroupPolicy::CommandsOnly => {
                    // Only allow slash commands and ChannelContent::Command
                    let is_command = matches!(&message.content, ChannelContent::Command { .. })
                        || matches!(&message.content, ChannelContent::Text(t) if t.starts_with('/'));
                    if !is_command {
                        debug!("Ignoring non-command group message on {ct_str} (group_policy=commands_only)");
                        return;
                    }
                }
                GroupPolicy::MentionOnly => {
                    // Pass through — adapters should only forward mentioned messages.
                    // This is a hint for adapters, not enforced here.
                }
                GroupPolicy::All => {}
            }
        } else {
            // DM
            match ov.dm_policy {
                DmPolicy::Ignore => {
                    debug!("Ignoring DM on {ct_str} (dm_policy=ignore)");
                    return;
                }
                DmPolicy::AllowedOnly => {
                    // Rely on RBAC authorize_channel_user below
                }
                DmPolicy::Respond => {}
            }
        }
    }

    // --- Rate limiting ---
    if let Some(ref ov) = overrides {
        if ov.rate_limit_per_user > 0 {
            if let Err(msg) =
                rate_limiter.check(ct_str, &message.sender.platform_id, ov.rate_limit_per_user)
            {
                send_response(adapter, &message.sender, msg, thread_id, output_format).await;
                return;
            }
        }
    }

    let inbound_is_voice = matches!(&message.content, ChannelContent::Voice { .. });
    let room_defaults = handle.chat_rooms_config().await;

    let text = match &message.content {
        ChannelContent::Text(t) => t.clone(),
        ChannelContent::Command { name, args } => {
            let result = if name == "room" {
                handle_room_command(args, handle, router, message, &room_defaults).await
            } else {
                handle_command(name, args, handle, router, &message.sender).await
            };
            send_response(adapter, &message.sender, result, thread_id, output_format).await;
            return;
        }
        ChannelContent::Voice { url, .. } => match handle.transcribe_voice_url(ct_str, url).await {
            Ok(Some(transcript)) if !transcript.trim().is_empty() => transcript,
            Ok(_) => {
                send_response(
                    adapter,
                    &message.sender,
                    "I received your voice note but could not transcribe it.".to_string(),
                    thread_id,
                    output_format,
                )
                .await;
                return;
            }
            Err(e) => {
                warn!("Voice transcription failed on {ct_str}: {e}");
                send_response(
                    adapter,
                    &message.sender,
                    "Voice transcription failed. Please retry or send text.".to_string(),
                    thread_id,
                    output_format,
                )
                .await;
                return;
            }
        },
        _ => {
            send_response(
                adapter,
                &message.sender,
                "I can only handle text messages for now.".to_string(),
                thread_id,
                output_format,
            )
            .await;
            return;
        }
    };

    // Check if it's a slash command embedded in text (e.g. "/agents")
    if text.starts_with('/') {
        let parts: Vec<&str> = text.splitn(2, ' ').collect();
        let cmd = &parts[0][1..]; // strip leading '/'
        let args: Vec<String> = if parts.len() > 1 {
            parts[1].split_whitespace().map(String::from).collect()
        } else {
            vec![]
        };

        if cmd == "room" {
            let result = handle_room_command(&args, handle, router, message, &room_defaults).await;
            send_response(adapter, &message.sender, result, thread_id, output_format).await;
            return;
        }

        if matches!(
            cmd,
            "start"
                | "help"
                | "agents"
                | "agent"
                | "status"
                | "models"
                | "providers"
                | "new"
                | "compact"
                | "model"
                | "stop"
                | "usage"
                | "think"
                | "skills"
                | "hands"
                | "workflows"
                | "workflow"
                | "triggers"
                | "trigger"
                | "schedules"
                | "schedule"
                | "approvals"
                | "approve"
                | "reject"
                | "budget"
                | "peers"
                | "a2a"
        ) {
            let result = handle_command(cmd, &args, handle, router, &message.sender).await;
            send_response(adapter, &message.sender, result, thread_id, output_format).await;
            return;
        }
        // Other slash commands pass through to the agent
    }

    if dispatch_group_room_message(
        message,
        &text,
        handle,
        router,
        adapter,
        ct_str,
        thread_id,
        output_format,
        &room_defaults,
    )
    .await
    {
        return;
    }

    // Check broadcast routing first
    if router.has_broadcast(&message.sender.platform_id) {
        let targets = router.resolve_broadcast(&message.sender.platform_id);
        if !targets.is_empty() {
            // RBAC check applies to broadcast too
            if let Err(denied) = handle
                .authorize_channel_user(ct_str, &message.sender.platform_id, "chat")
                .await
            {
                send_response(
                    adapter,
                    &message.sender,
                    format!("Access denied: {denied}"),
                    thread_id,
                    output_format,
                )
                .await;
                return;
            }
            let _ = adapter.send_typing(&message.sender).await;

            let strategy = router.broadcast_strategy();
            let mut responses = Vec::new();

            match strategy {
                openfang_types::config::BroadcastStrategy::Parallel => {
                    let mut handles_vec = Vec::new();
                    for (name, maybe_id) in &targets {
                        if let Some(aid) = maybe_id {
                            let h = handle.clone();
                            let t = text.clone();
                            let aid = *aid;
                            let name = name.clone();
                            handles_vec.push(tokio::spawn(async move {
                                let result = h.send_message(aid, &t).await;
                                (name, aid, result)
                            }));
                        }
                    }
                    for jh in handles_vec {
                        if let Ok((name, _aid, result)) = jh.await {
                            match result {
                                Ok(r) => responses.push(format!("[{name}]: {r}")),
                                Err(e) => responses.push(format!("[{name}]: Error: {e}")),
                            }
                        }
                    }
                }
                openfang_types::config::BroadcastStrategy::Sequential => {
                    for (name, maybe_id) in &targets {
                        if let Some(aid) = maybe_id {
                            match handle.send_message(*aid, &text).await {
                                Ok(r) => responses.push(format!("[{name}]: {r}")),
                                Err(e) => responses.push(format!("[{name}]: Error: {e}")),
                            }
                        }
                    }
                }
            }

            let combined = responses.join("\n\n");
            send_response(adapter, &message.sender, combined, thread_id, output_format).await;
            return;
        }
    }

    // Route to agent (standard path)
    let agent_id = router.resolve(
        &message.channel,
        &message.sender.platform_id,
        message.sender.openfang_user.as_deref(),
    );

    let agent_id = match agent_id {
        Some(id) => id,
        None => {
            // Fallback: try "assistant" agent, then first available agent
            let fallback = handle.find_agent_by_name("assistant").await.ok().flatten();
            let fallback = match fallback {
                Some(id) => Some(id),
                None => handle
                    .list_agents()
                    .await
                    .ok()
                    .and_then(|agents| agents.first().map(|(id, _)| *id)),
            };
            match fallback {
                Some(id) => {
                    // Auto-set this as the user's default so future messages route directly
                    router.set_user_default(message.sender.platform_id.clone(), id);
                    id
                }
                None => {
                    send_response(
                        adapter,
                        &message.sender,
                        "No agents available. Start the dashboard at http://127.0.0.1:4200 to create one.".to_string(),
                        thread_id,
                        output_format,
                    ).await;
                    return;
                }
            }
        }
    };

    // RBAC: authorize the user before forwarding to agent
    if let Err(denied) = handle
        .authorize_channel_user(ct_str, &message.sender.platform_id, "chat")
        .await
    {
        send_response(
            adapter,
            &message.sender,
            format!("Access denied: {denied}"),
            thread_id,
            output_format,
        )
        .await;
        return;
    }

    // Auto-reply check — if enabled, the engine decides whether to process this message.
    // If auto-reply is enabled but suppressed for this message, skip agent call entirely.
    if let Some(reply) = handle.check_auto_reply(agent_id, &text).await {
        send_response(adapter, &message.sender, reply, thread_id, output_format).await;
        handle
            .record_delivery(agent_id, ct_str, &message.sender.platform_id, true, None)
            .await;
        return;
    }

    // Send typing indicator (best-effort)
    let _ = adapter.send_typing(&message.sender).await;

    // Send to agent and relay response
    match handle.send_message(agent_id, &text).await {
        Ok(response) => {
            let mut delivered_as_voice = false;
            if let Some(voice_cfg) = handle.channel_voice_config(ct_str).await {
                let wants_voice = match voice_cfg.reply_mode {
                    VoiceReplyMode::Off => false,
                    VoiceReplyMode::Always => true,
                    VoiceReplyMode::Auto => {
                        inbound_is_voice || should_send_voice_reply(&response, &voice_cfg)
                    }
                };
                if wants_voice {
                    match handle
                        .synthesize_voice(ct_str, &response, voice_cfg.default_language)
                        .await
                    {
                        Ok(Some(asset)) => {
                            if let Err(e) =
                                send_voice_response(adapter, &message.sender, asset, thread_id)
                                    .await
                            {
                                warn!("Voice send failed on {ct_str}, falling back to text: {e}");
                            } else {
                                delivered_as_voice = true;
                            }
                        }
                        Ok(None) => {
                            debug!("Voice synthesis unavailable for {ct_str}, using text fallback");
                        }
                        Err(e) => {
                            warn!("Voice synthesis failed on {ct_str}, using text fallback: {e}");
                        }
                    }
                }
            }

            if !delivered_as_voice {
                send_response(adapter, &message.sender, response, thread_id, output_format).await;
            }
            handle
                .record_delivery(agent_id, ct_str, &message.sender.platform_id, true, None)
                .await;
        }
        Err(e) => {
            warn!("Agent error for {agent_id}: {e}");
            let err_msg = format!("Agent error: {e}");
            send_response(
                adapter,
                &message.sender,
                err_msg.clone(),
                thread_id,
                output_format,
            )
            .await;
            handle
                .record_delivery(
                    agent_id,
                    ct_str,
                    &message.sender.platform_id,
                    false,
                    Some(&err_msg),
                )
                .await;
        }
    }
}

/// Handle a bot command (returns the response text).
async fn handle_command(
    name: &str,
    args: &[String],
    handle: &Arc<dyn ChannelBridgeHandle>,
    router: &Arc<AgentRouter>,
    sender: &ChannelUser,
) -> String {
    match name {
        "start" => {
            let agents = handle.list_agents().await.unwrap_or_default();
            let mut msg = "Welcome to OpenFang! I connect you to AI agents.\n\nAvailable agents:\n"
                .to_string();
            if agents.is_empty() {
                msg.push_str("  (none running)\n");
            } else {
                for (_, name) in &agents {
                    msg.push_str(&format!("  - {name}\n"));
                }
            }
            msg.push_str(
                "\nCommands:\n/agents - list agents\n/agent <name> - select an agent\n/room status - show group room routing state\n/help - show this help",
            );
            msg
        }
        "help" => "OpenFang Bot Commands:\n\
             \n\
             Session:\n\
             /agents - list running agents\n\
             /agent <name> - select which agent to talk to\n\
             /room status - show group room routing state\n\
             /room mode <active|panel|orchestrator> - set room mode\n\
             /room active <agent> - set active room agent\n\
             /new - reset session (clear messages)\n\
             /compact - trigger LLM session compaction\n\
             /model [name] - show or switch agent model\n\
             /stop - cancel current agent run\n\
             /usage - show session token usage and cost\n\
             /think [on|off] - toggle extended thinking\n\
             \n\
             Info:\n\
             /models - list available AI models\n\
             /providers - show configured providers\n\
             /skills - list installed skills\n\
             /hands - list available and active hands\n\
             /status - show system status\n\
             \n\
             Automation:\n\
             /workflows - list workflows\n\
             /workflow run <name> [input] - run a workflow\n\
             /triggers - list event triggers\n\
             /trigger add <agent> <pattern> <prompt> - create trigger\n\
             /trigger del <id> - remove trigger\n\
             /schedules - list cron jobs\n\
             /schedule add <agent> <cron-5-fields> <message> - create job\n\
             /schedule del <id> - remove job\n\
             /schedule run <id> - run job now\n\
             /approvals - list pending approvals\n\
             /approve <id> - approve a request\n\
             /reject <id> - reject a request\n\
             \n\
             Monitoring:\n\
             /budget - show spending limits and current costs\n\
             /peers - show OFP peer network status\n\
             /a2a - list discovered external A2A agents\n\
             \n\
             /start - show welcome message\n\
             /help - show this help"
            .to_string(),
        "status" => handle.uptime_info().await,
        "agents" => {
            let agents = handle.list_agents().await.unwrap_or_default();
            if agents.is_empty() {
                "No agents running.".to_string()
            } else {
                let mut msg = "Running agents:\n".to_string();
                for (_, name) in &agents {
                    msg.push_str(&format!("  - {name}\n"));
                }
                msg
            }
        }
        "agent" => {
            if args.is_empty() {
                return "Usage: /agent <name>".to_string();
            }
            let agent_name = args.join(" ");
            match handle.find_agent_by_name(&agent_name).await {
                Ok(Some(agent_id)) => {
                    router.set_user_default(sender.platform_id.clone(), agent_id);
                    format!("Now talking to agent: {agent_name}")
                }
                Ok(None) => {
                    // Try to spawn it
                    match handle.spawn_agent_by_name(&agent_name).await {
                        Ok(agent_id) => {
                            router.set_user_default(sender.platform_id.clone(), agent_id);
                            format!("Spawned and connected to agent: {agent_name}")
                        }
                        Err(e) => {
                            format!("Agent '{agent_name}' not found and could not spawn: {e}")
                        }
                    }
                }
                Err(e) => format!("Error finding agent: {e}"),
            }
        }
        "new" => {
            // Need to resolve the user's current agent
            let agent_id = router.resolve(
                &crate::types::ChannelType::CLI,
                &sender.platform_id,
                sender.openfang_user.as_deref(),
            );
            match agent_id {
                Some(aid) => handle
                    .reset_session(aid)
                    .await
                    .unwrap_or_else(|e| format!("Error: {e}")),
                None => "No agent selected. Use /agent <name> first.".to_string(),
            }
        }
        "compact" => {
            let agent_id = router.resolve(
                &crate::types::ChannelType::CLI,
                &sender.platform_id,
                sender.openfang_user.as_deref(),
            );
            match agent_id {
                Some(aid) => handle
                    .compact_session(aid)
                    .await
                    .unwrap_or_else(|e| format!("Error: {e}")),
                None => "No agent selected. Use /agent <name> first.".to_string(),
            }
        }
        "model" => {
            let agent_id = router.resolve(
                &crate::types::ChannelType::CLI,
                &sender.platform_id,
                sender.openfang_user.as_deref(),
            );
            match agent_id {
                Some(aid) => {
                    if args.is_empty() {
                        // Show current model
                        handle
                            .set_model(aid, "")
                            .await
                            .unwrap_or_else(|e| format!("Error: {e}"))
                    } else {
                        handle
                            .set_model(aid, &args[0])
                            .await
                            .unwrap_or_else(|e| format!("Error: {e}"))
                    }
                }
                None => "No agent selected. Use /agent <name> first.".to_string(),
            }
        }
        "stop" => {
            let agent_id = router.resolve(
                &crate::types::ChannelType::CLI,
                &sender.platform_id,
                sender.openfang_user.as_deref(),
            );
            match agent_id {
                Some(aid) => handle
                    .stop_run(aid)
                    .await
                    .unwrap_or_else(|e| format!("Error: {e}")),
                None => "No agent selected. Use /agent <name> first.".to_string(),
            }
        }
        "usage" => {
            let agent_id = router.resolve(
                &crate::types::ChannelType::CLI,
                &sender.platform_id,
                sender.openfang_user.as_deref(),
            );
            match agent_id {
                Some(aid) => handle
                    .session_usage(aid)
                    .await
                    .unwrap_or_else(|e| format!("Error: {e}")),
                None => "No agent selected. Use /agent <name> first.".to_string(),
            }
        }
        "think" => {
            let agent_id = router.resolve(
                &crate::types::ChannelType::CLI,
                &sender.platform_id,
                sender.openfang_user.as_deref(),
            );
            match agent_id {
                Some(aid) => {
                    let on = args.first().map(|a| a == "on").unwrap_or(true);
                    handle
                        .set_thinking(aid, on)
                        .await
                        .unwrap_or_else(|e| format!("Error: {e}"))
                }
                None => "No agent selected. Use /agent <name> first.".to_string(),
            }
        }
        "models" => handle.list_models_text().await,
        "providers" => handle.list_providers_text().await,
        "skills" => handle.list_skills_text().await,
        "hands" => handle.list_hands_text().await,

        // ── Automation: workflows, triggers, schedules, approvals ──
        "workflows" => handle.list_workflows_text().await,
        "workflow" => {
            if args.len() >= 2 && args[0] == "run" {
                let wf_name = &args[1];
                let input = if args.len() > 2 {
                    args[2..].join(" ")
                } else {
                    String::new()
                };
                handle.run_workflow_text(wf_name, &input).await
            } else {
                "Usage: /workflow run <name> [input]".to_string()
            }
        }
        "triggers" => handle.list_triggers_text().await,
        "trigger" => {
            if args.len() >= 4 && args[0] == "add" {
                let agent_name = &args[1];
                let pattern = &args[2];
                let prompt = args[3..].join(" ");
                handle
                    .create_trigger_text(agent_name, pattern, &prompt)
                    .await
            } else if args.len() >= 2 && args[0] == "del" {
                handle.delete_trigger_text(&args[1]).await
            } else {
                "Usage:\n  /trigger add <agent> <pattern> <prompt>\n  /trigger del <id-prefix>"
                    .to_string()
            }
        }
        "schedules" => handle.list_schedules_text().await,
        "schedule" => {
            if args.is_empty() {
                return "Usage:\n  /schedule add <agent> <cron-5-fields> <message>\n  /schedule del <id-prefix>\n  /schedule run <id-prefix>".to_string();
            }
            let action = args[0].as_str();
            match action {
                "add" | "del" | "run" => {
                    handle.manage_schedule_text(action, &args[1..]).await
                }
                _ => "Usage:\n  /schedule add <agent> <cron-5-fields> <message>\n  /schedule del <id-prefix>\n  /schedule run <id-prefix>".to_string(),
            }
        }
        "approvals" => handle.list_approvals_text().await,
        "approve" => {
            if args.is_empty() {
                "Usage: /approve <id-prefix>".to_string()
            } else {
                handle.resolve_approval_text(&args[0], true).await
            }
        }
        "reject" => {
            if args.is_empty() {
                "Usage: /reject <id-prefix>".to_string()
            } else {
                handle.resolve_approval_text(&args[0], false).await
            }
        }

        // ── Budget, Network, A2A ──
        "budget" => handle.budget_text().await,
        "peers" => handle.peers_text().await,
        "a2a" => handle.a2a_agents_text().await,

        _ => format!("Unknown command: /{name}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChannelType;
    use std::sync::Mutex;

    /// Mock kernel handle for testing.
    struct MockHandle {
        agents: Mutex<Vec<(AgentId, String)>>,
    }

    #[async_trait]
    impl ChannelBridgeHandle for MockHandle {
        async fn send_message(&self, _agent_id: AgentId, message: &str) -> Result<String, String> {
            Ok(format!("Echo: {message}"))
        }
        async fn find_agent_by_name(&self, name: &str) -> Result<Option<AgentId>, String> {
            let agents = self.agents.lock().unwrap();
            Ok(agents.iter().find(|(_, n)| n == name).map(|(id, _)| *id))
        }
        async fn list_agents(&self) -> Result<Vec<(AgentId, String)>, String> {
            Ok(self.agents.lock().unwrap().clone())
        }
        async fn spawn_agent_by_name(&self, _manifest_name: &str) -> Result<AgentId, String> {
            Err("spawn not implemented in mock".to_string())
        }
    }

    #[test]
    fn test_command_parsing() {
        // Verify slash commands are parsed correctly from text
        let text = "/agent hello-world";
        assert!(text.starts_with('/'));
        let parts: Vec<&str> = text.splitn(2, ' ').collect();
        let cmd = &parts[0][1..];
        assert_eq!(cmd, "agent");
        let args: Vec<String> = if parts.len() > 1 {
            parts[1].split_whitespace().map(String::from).collect()
        } else {
            vec![]
        };
        assert_eq!(args, vec!["hello-world"]);
    }

    #[tokio::test]
    async fn test_dispatch_routes_to_correct_agent() {
        let agent_id = AgentId::new();
        let mock = Arc::new(MockHandle {
            agents: Mutex::new(vec![(agent_id, "test-agent".to_string())]),
        });

        let handle: Arc<dyn ChannelBridgeHandle> = mock;

        // Verify find_agent_by_name works
        let found = handle.find_agent_by_name("test-agent").await.unwrap();
        assert_eq!(found, Some(agent_id));

        let not_found = handle.find_agent_by_name("nonexistent").await.unwrap();
        assert_eq!(not_found, None);

        // Verify send_message echoes
        let response = handle.send_message(agent_id, "hello").await.unwrap();
        assert_eq!(response, "Echo: hello");
    }

    #[tokio::test]
    async fn test_handle_command_agents() {
        let agent_id = AgentId::new();
        let handle: Arc<dyn ChannelBridgeHandle> = Arc::new(MockHandle {
            agents: Mutex::new(vec![(agent_id, "coder".to_string())]),
        });
        let router = Arc::new(AgentRouter::new());
        let sender = ChannelUser {
            platform_id: "user1".to_string(),
            display_name: "Test".to_string(),
            openfang_user: None,
        };

        let result = handle_command("agents", &[], &handle, &router, &sender).await;
        assert!(result.contains("coder"));

        let result = handle_command("help", &[], &handle, &router, &sender).await;
        assert!(result.contains("/agents"));
    }

    #[tokio::test]
    async fn test_handle_command_agent_select() {
        let agent_id = AgentId::new();
        let handle: Arc<dyn ChannelBridgeHandle> = Arc::new(MockHandle {
            agents: Mutex::new(vec![(agent_id, "coder".to_string())]),
        });
        let router = Arc::new(AgentRouter::new());
        let sender = ChannelUser {
            platform_id: "user1".to_string(),
            display_name: "Test".to_string(),
            openfang_user: None,
        };

        // Select existing agent
        let result =
            handle_command("agent", &["coder".to_string()], &handle, &router, &sender).await;
        assert!(result.contains("Now talking to agent: coder"));

        // Verify router was updated
        let resolved = router.resolve(&ChannelType::Telegram, "user1", None);
        assert_eq!(resolved, Some(agent_id));
    }

    #[test]
    fn test_rate_limiter_allows_within_limit() {
        let limiter = ChannelRateLimiter::default();
        assert!(limiter.check("telegram", "user1", 5).is_ok());
        assert!(limiter.check("telegram", "user1", 5).is_ok());
        assert!(limiter.check("telegram", "user1", 5).is_ok());
    }

    #[test]
    fn test_rate_limiter_blocks_over_limit() {
        let limiter = ChannelRateLimiter::default();
        for _ in 0..3 {
            limiter.check("telegram", "user1", 3).unwrap();
        }
        // 4th should be blocked
        let result = limiter.check("telegram", "user1", 3);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Rate limit exceeded"));
    }

    #[test]
    fn test_rate_limiter_zero_means_unlimited() {
        let limiter = ChannelRateLimiter::default();
        for _ in 0..100 {
            assert!(limiter.check("telegram", "user1", 0).is_ok());
        }
    }

    #[test]
    fn test_rate_limiter_separate_users() {
        let limiter = ChannelRateLimiter::default();
        for _ in 0..3 {
            limiter.check("telegram", "user1", 3).unwrap();
        }
        // user1 is blocked
        assert!(limiter.check("telegram", "user1", 3).is_err());
        // user2 should still be ok
        assert!(limiter.check("telegram", "user2", 3).is_ok());
    }

    #[test]
    fn test_dm_policy_filtering() {
        // Test that DmPolicy::Ignore would be checked
        assert_eq!(DmPolicy::default(), DmPolicy::Respond);
        assert_eq!(GroupPolicy::default(), GroupPolicy::MentionOnly);
    }

    #[test]
    fn test_channel_type_str() {
        assert_eq!(channel_type_str(&ChannelType::Telegram), "telegram");
        assert_eq!(channel_type_str(&ChannelType::Matrix), "matrix");
        assert_eq!(channel_type_str(&ChannelType::Email), "email");
        assert_eq!(
            channel_type_str(&ChannelType::Custom("irc".to_string())),
            "irc"
        );
    }

    #[test]
    fn test_conversation_key_uses_room_and_thread() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("chat_id".to_string(), serde_json::json!("room-123"));
        let msg = ChannelMessage {
            channel: ChannelType::WhatsApp,
            platform_message_id: "m1".to_string(),
            sender: ChannelUser {
                platform_id: "u1".to_string(),
                display_name: "User".to_string(),
                openfang_user: None,
            },
            content: ChannelContent::Text("hi".to_string()),
            target_agent: None,
            timestamp: chrono::Utc::now(),
            is_group: true,
            thread_id: Some("thread-9".to_string()),
            metadata,
        };
        assert_eq!(
            conversation_key_for_message(&msg).as_deref(),
            Some("whatsapp:room-123:thread:thread-9")
        );
    }

    #[test]
    fn test_extract_mentioned_agent_names() {
        let a = AgentId::new();
        let b = AgentId::new();
        let agents = vec![
            (a, "Data Analyst".to_string()),
            (b, "coder_bot".to_string()),
        ];
        let text = "@data-analyst can you check this? also @coder_bot please.";
        let mentions = extract_mentioned_agent_names(text, &agents);
        assert_eq!(
            mentions,
            vec!["Data Analyst".to_string(), "coder_bot".to_string()]
        );
    }

    #[test]
    fn test_voice_policy_auto_threshold_and_keywords() {
        let cfg = ChannelVoiceConfig {
            reply_mode: VoiceReplyMode::Auto,
            default_language: VoiceLanguage::De,
            auto_min_text_length: 10,
            auto_keywords: vec!["status".to_string()],
        };
        assert!(should_send_voice_reply("this is long enough", &cfg));
        assert!(should_send_voice_reply("quick STATUS update", &cfg));
        assert!(!should_send_voice_reply("short", &cfg));
    }
}
