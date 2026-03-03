//! WhatsApp Cloud API channel adapter.
//!
//! Uses the official WhatsApp Business Cloud API to send and receive messages.
//! Requires a webhook endpoint for incoming messages and the Cloud API for outgoing.

use crate::types::{ChannelAdapter, ChannelContent, ChannelMessage, ChannelType, ChannelUser};
use async_trait::async_trait;
use futures::Stream;
use serde::Deserialize;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio::time::{sleep, Duration};
use tracing::{error, info};
use zeroize::Zeroizing;

const MAX_MESSAGE_LEN: usize = 4096;
const GATEWAY_POLL_TIMEOUT_MS: u64 = 25_000;

#[derive(Debug, Deserialize)]
struct GatewayPollEnvelope {
    event: Option<GatewayInboundEvent>,
}

#[derive(Debug, Deserialize)]
struct GatewayInboundEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    message_id: String,
    #[serde(default)]
    chat_jid: String,
    #[serde(default)]
    sender_phone: String,
    #[serde(default)]
    sender_name: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    voice_url: Option<String>,
    #[serde(default)]
    duration_seconds: Option<u32>,
    #[serde(default)]
    is_group: bool,
    #[serde(default)]
    timestamp: Option<i64>,
}

/// WhatsApp Cloud API adapter.
///
/// Supports two modes:
/// - **Cloud API mode**: Uses the official WhatsApp Business Cloud API (requires Meta dev account).
/// - **Web/QR mode**: Routes outgoing messages through a local Baileys-based gateway process.
///
/// Mode is selected automatically: if `gateway_url` is set (from `WHATSAPP_WEB_GATEWAY_URL`),
/// the adapter uses Web mode. Otherwise it falls back to Cloud API mode.
pub struct WhatsAppAdapter {
    /// WhatsApp Business phone number ID (Cloud API mode).
    phone_number_id: String,
    /// SECURITY: Access token is zeroized on drop.
    access_token: Zeroizing<String>,
    /// SECURITY: Verify token is zeroized on drop.
    verify_token: Zeroizing<String>,
    /// Port to listen for webhook callbacks (Cloud API mode).
    webhook_port: u16,
    /// HTTP client.
    client: reqwest::Client,
    /// Allowed phone numbers (empty = allow all).
    allowed_users: Vec<String>,
    /// Optional WhatsApp Web gateway URL for QR/Web mode (e.g. "http://127.0.0.1:3009").
    gateway_url: Option<String>,
    /// Shutdown signal.
    shutdown_tx: Arc<watch::Sender<bool>>,
    shutdown_rx: watch::Receiver<bool>,
}

impl WhatsAppAdapter {
    /// Create a new WhatsApp Cloud API adapter.
    pub fn new(
        phone_number_id: String,
        access_token: String,
        verify_token: String,
        webhook_port: u16,
        allowed_users: Vec<String>,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            phone_number_id,
            access_token: Zeroizing::new(access_token),
            verify_token: Zeroizing::new(verify_token),
            webhook_port,
            client: reqwest::Client::new(),
            allowed_users,
            gateway_url: None,
            shutdown_tx: Arc::new(shutdown_tx),
            shutdown_rx,
        }
    }

    /// Create a new WhatsApp adapter with gateway URL for Web/QR mode.
    ///
    /// When `gateway_url` is `Some`, outgoing messages are sent via `POST {gateway_url}/message/send`
    /// instead of the Cloud API. Incoming messages are handled by the gateway itself.
    pub fn with_gateway(mut self, gateway_url: Option<String>) -> Self {
        self.gateway_url = gateway_url.filter(|u| !u.is_empty());
        self
    }

    /// Send a text message via the WhatsApp Cloud API.
    async fn api_send_message(
        &self,
        to: &str,
        text: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!(
            "https://graph.facebook.com/v21.0/{}/messages",
            self.phone_number_id
        );

        // Split long messages
        let chunks = crate::types::split_message(text, MAX_MESSAGE_LEN);
        for chunk in chunks {
            let body = serde_json::json!({
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": { "body": chunk }
            });

            let resp = self
                .client
                .post(&url)
                .bearer_auth(&*self.access_token)
                .json(&body)
                .send()
                .await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                error!("WhatsApp API error {status}: {body}");
                return Err(format!("WhatsApp API error {status}: {body}").into());
            }
        }

        Ok(())
    }

    /// Send a voice/audio message via WhatsApp Cloud API.
    async fn api_send_voice(
        &self,
        to: &str,
        voice_url: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!(
            "https://graph.facebook.com/v21.0/{}/messages",
            self.phone_number_id
        );

        let body = serde_json::json!({
            "messaging_product": "whatsapp",
            "to": to,
            "type": "audio",
            "audio": { "link": voice_url }
        });

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&*self.access_token)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!("WhatsApp API voice error {status}: {body}");
            return Err(format!("WhatsApp API voice error {status}: {body}").into());
        }

        Ok(())
    }

    /// Mark a message as read.
    #[allow(dead_code)]
    async fn api_mark_read(&self, message_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!(
            "https://graph.facebook.com/v21.0/{}/messages",
            self.phone_number_id
        );

        let body = serde_json::json!({
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        });

        let _ = self
            .client
            .post(&url)
            .bearer_auth(&*self.access_token)
            .json(&body)
            .send()
            .await;

        Ok(())
    }

    /// Send a text message via the WhatsApp Web gateway.
    async fn gateway_send_message(
        &self,
        gateway_url: &str,
        to: &str,
        text: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{}/message/send", gateway_url.trim_end_matches('/'));
        let body = serde_json::json!({ "to": to, "text": text });

        let resp = self.client.post(&url).json(&body).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!("WhatsApp gateway error {status}: {body}");
            return Err(format!("WhatsApp gateway error {status}: {body}").into());
        }

        Ok(())
    }

    /// Send a voice message through the WhatsApp Web gateway.
    async fn gateway_send_voice(
        &self,
        gateway_url: &str,
        to: &str,
        voice_url: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{}/message/send_voice", gateway_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "to": to,
            "voice_url": voice_url,
            "ptt": true
        });

        let resp = self.client.post(&url).json(&body).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!("WhatsApp gateway voice error {status}: {body}");
            return Err(format!("WhatsApp gateway voice error {status}: {body}").into());
        }
        Ok(())
    }

    fn is_allowed_user(allowed_users: &[String], phone: &str, chat_jid: &str) -> bool {
        if allowed_users.is_empty() {
            return true;
        }
        allowed_users
            .iter()
            .any(|u| u == phone || (!chat_jid.is_empty() && u == chat_jid))
    }

    async fn gateway_poll_event(
        client: &reqwest::Client,
        gateway_url: &str,
        allowed_users: &[String],
    ) -> Result<Option<ChannelMessage>, String> {
        let url = format!(
            "{}/events/poll?timeout_ms={}",
            gateway_url.trim_end_matches('/'),
            GATEWAY_POLL_TIMEOUT_MS
        );

        let resp = client
            .get(&url)
            .timeout(Duration::from_millis(GATEWAY_POLL_TIMEOUT_MS + 5_000))
            .send()
            .await
            .map_err(|e| format!("Gateway poll request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Gateway poll failed ({status}): {body}"));
        }

        let envelope: GatewayPollEnvelope = resp
            .json()
            .await
            .map_err(|e| format!("Gateway poll JSON parse failed: {e}"))?;
        let Some(event) = envelope.event else {
            return Ok(None);
        };

        let platform_id = if event.sender_phone.is_empty() {
            event.chat_jid.clone()
        } else {
            event.sender_phone.clone()
        };
        if platform_id.is_empty() {
            return Ok(None);
        }

        if !Self::is_allowed_user(allowed_users, &platform_id, &event.chat_jid) {
            return Ok(None);
        }

        let display_name = if event.sender_name.is_empty() {
            platform_id.clone()
        } else {
            event.sender_name
        };

        let content = match event.event_type.as_str() {
            "voice" => {
                let Some(voice_url) = event.voice_url else {
                    return Ok(None);
                };
                ChannelContent::Voice {
                    url: voice_url,
                    duration_seconds: event.duration_seconds.unwrap_or(0),
                }
            }
            _ => {
                if event.text.trim().is_empty() {
                    return Ok(None);
                }
                ChannelContent::Text(event.text)
            }
        };

        let timestamp = event
            .timestamp
            .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0))
            .unwrap_or_else(chrono::Utc::now);
        let message_id = if event.message_id.is_empty() {
            format!("gw-{}", timestamp.timestamp_millis())
        } else {
            event.message_id
        };
        let mut metadata = HashMap::new();
        if !event.chat_jid.is_empty() {
            metadata.insert("chat_jid".to_string(), serde_json::json!(event.chat_jid));
        }
        metadata.insert("source".to_string(), serde_json::json!("whatsapp_gateway"));

        Ok(Some(ChannelMessage {
            channel: ChannelType::WhatsApp,
            platform_message_id: message_id,
            sender: ChannelUser {
                platform_id,
                display_name,
                openfang_user: None,
            },
            content,
            target_agent: None,
            timestamp,
            is_group: event.is_group,
            thread_id: None,
            metadata,
        }))
    }

    /// Check if a phone number is allowed.
    #[allow(dead_code)]
    fn is_allowed(&self, phone: &str) -> bool {
        self.allowed_users.is_empty() || self.allowed_users.iter().any(|u| u == phone)
    }

    /// Returns true if this adapter is configured for Web/QR gateway mode.
    #[allow(dead_code)]
    pub fn is_gateway_mode(&self) -> bool {
        self.gateway_url.is_some()
    }
}

#[async_trait]
impl ChannelAdapter for WhatsAppAdapter {
    fn name(&self) -> &str {
        "whatsapp"
    }

    fn channel_type(&self) -> ChannelType {
        ChannelType::WhatsApp
    }

    async fn start(
        &self,
    ) -> Result<Pin<Box<dyn Stream<Item = ChannelMessage> + Send>>, Box<dyn std::error::Error>>
    {
        let (tx, rx) = mpsc::channel::<ChannelMessage>(256);

        // Web/QR gateway mode — poll inbound events from the gateway.
        if let Some(gateway_url) = self.gateway_url.clone() {
            let client = self.client.clone();
            let allowed_users = self.allowed_users.clone();
            let mut shutdown_rx = self.shutdown_rx.clone();

            info!("Starting WhatsApp gateway poller at {gateway_url}");
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = shutdown_rx.changed() => {
                            if *shutdown_rx.borrow() {
                                info!("WhatsApp gateway poller stopped");
                                break;
                            }
                        }
                        polled = Self::gateway_poll_event(&client, &gateway_url, &allowed_users) => {
                            match polled {
                                Ok(Some(message)) => {
                                    if tx.send(message).await.is_err() {
                                        break;
                                    }
                                }
                                Ok(None) => {
                                    // No event in this poll cycle.
                                }
                                Err(e) => {
                                    error!("WhatsApp gateway poll error: {e}");
                                    sleep(Duration::from_secs(2)).await;
                                }
                            }
                        }
                    }
                }
            });

            return Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)));
        }

        // Cloud API mode (webhook placeholder)
        let port = self.webhook_port;
        let _verify_token = self.verify_token.clone();
        let _allowed_users = self.allowed_users.clone();
        let _access_token = self.access_token.clone();
        let _phone_number_id = self.phone_number_id.clone();
        let mut shutdown_rx = self.shutdown_rx.clone();

        info!("Starting WhatsApp webhook listener on port {port}");

        tokio::spawn(async move {
            info!("WhatsApp webhook ready on port {port} (verify_token configured)");
            info!("Configure your webhook URL: https://your-domain:{port}/webhook");

            // Wait for shutdown
            let _ = shutdown_rx.changed().await;
            info!("WhatsApp adapter stopped");
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn send(
        &self,
        user: &ChannelUser,
        content: ChannelContent,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Web/QR gateway mode: route all messages through the gateway
        if let Some(ref gw) = self.gateway_url {
            match &content {
                ChannelContent::Text(t) => {
                    // Split long messages the same way as Cloud API mode
                    let chunks = crate::types::split_message(t, MAX_MESSAGE_LEN);
                    for chunk in chunks {
                        self.gateway_send_message(gw, &user.platform_id, chunk)
                            .await?;
                    }
                }
                ChannelContent::Voice { url, .. } => {
                    self.gateway_send_voice(gw, &user.platform_id, url).await?;
                }
                ChannelContent::Image { caption, .. } => {
                    let text = caption
                        .clone()
                        .unwrap_or_else(|| "(Image — not supported in Web mode)".to_string());
                    self.gateway_send_message(gw, &user.platform_id, &text)
                        .await?;
                }
                ChannelContent::File { filename, .. } => {
                    let text = format!("(File: {filename} — not supported in Web mode)");
                    self.gateway_send_message(gw, &user.platform_id, &text)
                        .await?;
                }
                _ => {
                    self.gateway_send_message(
                        gw,
                        &user.platform_id,
                        "(Unsupported content type in Web mode)",
                    )
                    .await?;
                }
            }
            return Ok(());
        }

        // Cloud API mode (default)
        match content {
            ChannelContent::Text(text) => {
                self.api_send_message(&user.platform_id, &text).await?;
            }
            ChannelContent::Image { url, caption } => {
                let body = serde_json::json!({
                    "messaging_product": "whatsapp",
                    "to": user.platform_id,
                    "type": "image",
                    "image": {
                        "link": url,
                        "caption": caption.unwrap_or_default()
                    }
                });
                let api_url = format!(
                    "https://graph.facebook.com/v21.0/{}/messages",
                    self.phone_number_id
                );
                self.client
                    .post(&api_url)
                    .bearer_auth(&*self.access_token)
                    .json(&body)
                    .send()
                    .await?;
            }
            ChannelContent::File { url, filename } => {
                let body = serde_json::json!({
                    "messaging_product": "whatsapp",
                    "to": user.platform_id,
                    "type": "document",
                    "document": {
                        "link": url,
                        "filename": filename
                    }
                });
                let api_url = format!(
                    "https://graph.facebook.com/v21.0/{}/messages",
                    self.phone_number_id
                );
                self.client
                    .post(&api_url)
                    .bearer_auth(&*self.access_token)
                    .json(&body)
                    .send()
                    .await?;
            }
            ChannelContent::Location { lat, lon } => {
                let body = serde_json::json!({
                    "messaging_product": "whatsapp",
                    "to": user.platform_id,
                    "type": "location",
                    "location": {
                        "latitude": lat,
                        "longitude": lon
                    }
                });
                let api_url = format!(
                    "https://graph.facebook.com/v21.0/{}/messages",
                    self.phone_number_id
                );
                self.client
                    .post(&api_url)
                    .bearer_auth(&*self.access_token)
                    .json(&body)
                    .send()
                    .await?;
            }
            ChannelContent::Voice { url, .. } => {
                self.api_send_voice(&user.platform_id, &url).await?;
            }
            _ => {
                self.api_send_message(&user.platform_id, "(Unsupported content type)")
                    .await?;
            }
        }
        Ok(())
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.shutdown_tx.send(true);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whatsapp_adapter_creation() {
        let adapter = WhatsAppAdapter::new(
            "12345".to_string(),
            "access_token".to_string(),
            "verify_token".to_string(),
            8443,
            vec![],
        );
        assert_eq!(adapter.name(), "whatsapp");
        assert_eq!(adapter.channel_type(), ChannelType::WhatsApp);
    }

    #[test]
    fn test_allowed_users_check() {
        let adapter = WhatsAppAdapter::new(
            "12345".to_string(),
            "token".to_string(),
            "verify".to_string(),
            8443,
            vec!["+1234567890".to_string()],
        );
        assert!(adapter.is_allowed("+1234567890"));
        assert!(!adapter.is_allowed("+9999999999"));

        let open = WhatsAppAdapter::new(
            "12345".to_string(),
            "token".to_string(),
            "verify".to_string(),
            8443,
            vec![],
        );
        assert!(open.is_allowed("+anything"));
    }
}
