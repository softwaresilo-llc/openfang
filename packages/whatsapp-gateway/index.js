#!/usr/bin/env node
'use strict';

const http = require('node:http');
const { randomUUID } = require('node:crypto');

// ---------------------------------------------------------------------------
// Config from environment
// ---------------------------------------------------------------------------
const PORT = parseInt(process.env.WHATSAPP_GATEWAY_PORT || '3009', 10);
const OPENFANG_URL = (process.env.OPENFANG_URL || 'http://127.0.0.1:4200').replace(/\/+$/, '');
const SELF_CHAT_MODE = /^(1|true|yes|on)$/i.test(
  String(process.env.OPENFANG_WHATSAPP_SELF_CHAT_MODE || 'false')
);

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let sock = null;          // Baileys socket
let sessionId = '';       // current session identifier
let qrDataUrl = '';       // latest QR code as data:image/png;base64,...
let connStatus = 'disconnected'; // disconnected | qr_ready | connected
let qrExpired = false;
let statusMessage = 'Not started';
const inboundEvents = []; // queued inbound events for the Rust channel adapter
const MAX_INBOUND_EVENTS = 200;
const mediaStore = new Map(); // mediaId -> { data: Buffer, mimeType: string, expiresAt: number }
const MEDIA_TTL_MS = 5 * 60 * 1000;
const outboundMessageIds = new Map(); // messageId -> expiresAt
const OUTBOUND_ID_TTL_MS = 10 * 60 * 1000;

function enqueueInboundEvent(event) {
  if (!event) return;
  if (inboundEvents.length >= MAX_INBOUND_EVENTS) inboundEvents.shift();
  inboundEvents.push(event);
}

function dequeueInboundEvent() {
  if (!inboundEvents.length) return null;
  return inboundEvents.shift() || null;
}

function saveMedia(buffer, mimeType) {
  const mediaId = randomUUID();
  mediaStore.set(mediaId, {
    data: Buffer.from(buffer),
    mimeType: mimeType || 'application/octet-stream',
    expiresAt: Date.now() + MEDIA_TTL_MS,
  });
  return mediaId;
}

function readMedia(mediaId) {
  const item = mediaStore.get(mediaId);
  if (!item) return null;
  if (Date.now() > item.expiresAt) {
    mediaStore.delete(mediaId);
    return null;
  }
  return item;
}

function trackOutboundMessageId(messageId) {
  if (!messageId) return;
  outboundMessageIds.set(messageId, Date.now() + OUTBOUND_ID_TTL_MS);
}

function consumeTrackedOutboundMessageId(messageId) {
  if (!messageId) return false;
  const expiresAt = outboundMessageIds.get(messageId);
  if (!expiresAt) return false;
  outboundMessageIds.delete(messageId);
  if (Date.now() > expiresAt) return false;
  return true;
}

function normalizedWaIdNumber(raw) {
  if (!raw) return '';
  return String(raw)
    .split('@')[0]
    .split(':')[0]
    .replace(/[^\d]/g, '');
}

function isSelfChat(remoteJid) {
  const remoteHost = String(remoteJid || '').split('@')[1] || '';
  const remoteNumber = normalizedWaIdNumber(remoteJid);
  if (!remoteNumber) return false;
  const myPhoneNumber = normalizedWaIdNumber(sock?.user?.id || '');
  const myLidNumber = normalizedWaIdNumber(sock?.user?.lid || '');
  if (remoteHost === 'lid') {
    if (myLidNumber) return remoteNumber === myLidNumber;
    return myPhoneNumber ? remoteNumber === myPhoneNumber : false;
  }
  return myPhoneNumber ? remoteNumber === myPhoneNumber : false;
}

setInterval(() => {
  const now = Date.now();
  for (const [id, item] of mediaStore.entries()) {
    if (now > item.expiresAt) mediaStore.delete(id);
  }
  for (const [messageId, expiresAt] of outboundMessageIds.entries()) {
    if (now > expiresAt) outboundMessageIds.delete(messageId);
  }
}, 30_000);

// ---------------------------------------------------------------------------
// Baileys connection
// ---------------------------------------------------------------------------
async function startConnection() {
  // Dynamic imports — Baileys is ESM-only in v6+
  const { default: makeWASocket, useMultiFileAuthState, DisconnectReason, fetchLatestBaileysVersion, downloadMediaMessage } =
    await import('@whiskeysockets/baileys');
  const QRCode = (await import('qrcode')).default || await import('qrcode');
  const pino = (await import('pino')).default || await import('pino');

  const logger = pino({ level: 'warn' });
  const authDir = require('node:path').join(__dirname, 'auth_store');

  const { state, saveCreds } = await useMultiFileAuthState(
    require('node:path').join(__dirname, 'auth_store')
  );
  const { version } = await fetchLatestBaileysVersion();

  sessionId = randomUUID();
  qrDataUrl = '';
  qrExpired = false;
  connStatus = 'disconnected';
  statusMessage = 'Connecting...';

  sock = makeWASocket({
    version,
    auth: state,
    logger,
    printQRInTerminal: true,
    browser: ['OpenFang', 'Desktop', '1.0.0'],
  });

  // Save credentials whenever they update
  sock.ev.on('creds.update', saveCreds);

  // Connection state changes (QR code, connected, disconnected)
  sock.ev.on('connection.update', async (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      // New QR code generated — convert to data URL
      try {
        qrDataUrl = await QRCode.toDataURL(qr, { width: 256, margin: 2 });
        connStatus = 'qr_ready';
        qrExpired = false;
        statusMessage = 'Scan this QR code with WhatsApp → Linked Devices';
        console.log('[gateway] QR code ready — waiting for scan');
      } catch (err) {
        console.error('[gateway] QR generation failed:', err.message);
      }
    }

    if (connection === 'close') {
      const statusCode = lastDisconnect?.error?.output?.statusCode;
      const reason = lastDisconnect?.error?.output?.payload?.message || 'unknown';
      console.log(`[gateway] Connection closed: ${reason} (${statusCode})`);

      if (statusCode === DisconnectReason.loggedOut) {
        // User logged out from phone — clear auth and stop
        connStatus = 'disconnected';
        statusMessage = 'Logged out. Generate a new QR code to reconnect.';
        qrDataUrl = '';
        sock = null;
        // Remove auth store so next connect gets a fresh QR
        const fs = require('node:fs');
        const path = require('node:path');
        const authPath = path.join(__dirname, 'auth_store');
        if (fs.existsSync(authPath)) {
          fs.rmSync(authPath, { recursive: true, force: true });
        }
      } else if (statusCode === DisconnectReason.restartRequired ||
                 statusCode === DisconnectReason.timedOut) {
        // Recoverable — reconnect automatically
        console.log('[gateway] Reconnecting...');
        statusMessage = 'Reconnecting...';
        setTimeout(() => startConnection(), 2000);
      } else {
        // QR expired or other non-recoverable close
        qrExpired = true;
        connStatus = 'disconnected';
        statusMessage = 'QR code expired. Click "Generate New QR" to retry.';
        qrDataUrl = '';
      }
    }

    if (connection === 'open') {
      connStatus = 'connected';
      qrExpired = false;
      qrDataUrl = '';
      statusMessage = 'Connected to WhatsApp';
      console.log('[gateway] Connected to WhatsApp!');
    }
  });

  // Incoming messages → queue for the Rust WhatsApp adapter poll loop
  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    // Self-chat / own-device messages can arrive as "append" instead of "notify".
    if (type !== 'notify' && type !== 'append') return;

    for (const msg of messages) {
      const remoteJid = msg.key.remoteJid || '';
      if (remoteJid === 'status@broadcast') continue;

      const messageId = msg.key.id || '';
      if (consumeTrackedOutboundMessageId(messageId)) {
        // This is our own outbound echo; ignore it to prevent loops.
        continue;
      }

      const selfChat = isSelfChat(remoteJid);
      if (msg.key.fromMe && (!SELF_CHAT_MODE || !selfChat)) {
        continue;
      }

      const sender = remoteJid;
      const text = msg.message?.conversation
        || msg.message?.extendedTextMessage?.text
        || msg.message?.imageMessage?.caption
        || '';
      const normalizedSender = normalizedWaIdNumber(sender)
        || normalizedWaIdNumber(msg.key?.participant || '')
        || '';
      const phone = normalizedSender ? ('+' + normalizedSender) : ('+' + sender.replace(/@.*$/, ''));
      const pushName = msg.pushName || phone;
      const tsSeconds = Number(msg.messageTimestamp || 0) || Math.floor(Date.now() / 1000);
      const baseEvent = {
        message_id: msg.key.id || randomUUID(),
        chat_jid: sender,
        sender_phone: phone,
        sender_name: pushName,
        is_group: sender.endsWith('@g.us'),
        timestamp: tsSeconds,
      };

      if (text && text.trim()) {
        enqueueInboundEvent({
          type: 'text',
          ...baseEvent,
          text,
        });
        console.log(`[gateway] Queued text from ${pushName} (${phone}): ${text.substring(0, 80)}`);
        continue;
      }

      // Voice note / audio fallback path
      if (msg.message?.audioMessage || msg.message?.pttMessage) {
        try {
          const audio = msg.message?.audioMessage || msg.message?.pttMessage;
          const mediaBuffer = await downloadMediaMessage(msg, 'buffer', {});
          if (mediaBuffer && mediaBuffer.length > 0) {
            const mimeType = audio?.mimetype || 'audio/ogg';
            const mediaId = saveMedia(mediaBuffer, mimeType);
            enqueueInboundEvent({
              type: 'voice',
              ...baseEvent,
              voice_url: `http://127.0.0.1:${PORT}/media/${mediaId}`,
              duration_seconds: Number(audio?.seconds || 0) || 0,
            });
            console.log(`[gateway] Queued voice from ${pushName} (${phone})`);
          }
        } catch (err) {
          console.error('[gateway] Failed to decode inbound audio:', err.message);
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Send a message via Baileys (called by OpenFang for outgoing)
// ---------------------------------------------------------------------------
async function sendMessage(to, text) {
  if (!sock || connStatus !== 'connected') {
    throw new Error('WhatsApp not connected');
  }

  // Normalize phone/JID/lid form → JID: "+1234567890" → "1234567890@s.whatsapp.net"
  const number = normalizedWaIdNumber(to);
  if (!number) {
    throw new Error(`Invalid recipient identifier: ${to}`);
  }
  const jid = number + '@s.whatsapp.net';

  const sent = await sock.sendMessage(jid, { text });
  const sentId = sent?.key?.id;
  if (sentId) {
    trackOutboundMessageId(sentId);
  }
}

async function sendVoiceMessage(to, voiceUrl, ptt = true) {
  if (!sock || connStatus !== 'connected') {
    throw new Error('WhatsApp not connected');
  }
  if (!voiceUrl) {
    throw new Error('voiceUrl is required');
  }

  const normalizedUrl = /^https?:\/\//i.test(voiceUrl)
    ? voiceUrl
    : `${OPENFANG_URL}${voiceUrl.startsWith('/') ? '' : '/'}${voiceUrl}`;

  const response = await fetch(normalizedUrl, { signal: AbortSignal.timeout(20_000) });
  if (!response.ok) {
    throw new Error(`Voice download failed (${response.status})`);
  }

  const mimeType = (response.headers.get('content-type') || 'audio/ogg').split(';')[0].trim();
  const arrayBuf = await response.arrayBuffer();
  const audioBuffer = Buffer.from(arrayBuf);
  if (!audioBuffer.length) {
    throw new Error('Downloaded voice payload is empty');
  }

  const number = normalizedWaIdNumber(to);
  if (!number) {
    throw new Error(`Invalid recipient identifier: ${to}`);
  }
  const jid = number + '@s.whatsapp.net';
  const sent = await sock.sendMessage(jid, {
    audio: audioBuffer,
    mimetype: mimeType || 'audio/ogg',
    ptt: !!ptt,
  });
  const sentId = sent?.key?.id;
  if (sentId) {
    trackOutboundMessageId(sentId);
  }
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------
function parseBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (e) {
        reject(new Error('Invalid JSON'));
      }
    });
    req.on('error', reject);
  });
}

function jsonResponse(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
    'Access-Control-Allow-Origin': '*',
  });
  res.end(body);
}

const server = http.createServer(async (req, res) => {
  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    return res.end();
  }

  const url = new URL(req.url, `http://localhost:${PORT}`);
  const path = url.pathname;

  try {
    // POST /login/start — start Baileys connection, return QR
    if (req.method === 'POST' && path === '/login/start') {
      // If already connected, just return success
      if (connStatus === 'connected') {
        return jsonResponse(res, 200, {
          qr_data_url: '',
          session_id: sessionId,
          message: 'Already connected to WhatsApp',
          connected: true,
        });
      }

      // Start a new connection (resets any existing)
      await startConnection();

      // Wait briefly for QR to generate (Baileys emits it quickly)
      let waited = 0;
      while (!qrDataUrl && connStatus !== 'connected' && waited < 15_000) {
        await new Promise((r) => setTimeout(r, 300));
        waited += 300;
      }

      return jsonResponse(res, 200, {
        qr_data_url: qrDataUrl,
        session_id: sessionId,
        message: statusMessage,
        connected: connStatus === 'connected',
      });
    }

    // GET /login/status — poll for connection status
    if (req.method === 'GET' && path === '/login/status') {
      return jsonResponse(res, 200, {
        connected: connStatus === 'connected',
        message: statusMessage,
        expired: qrExpired,
      });
    }

    // GET /events/poll — long-poll next inbound WhatsApp event for Rust adapter
    if (req.method === 'GET' && path === '/events/poll') {
      const rawTimeout = Number.parseInt(url.searchParams.get('timeout_ms') || '25000', 10);
      const timeoutMs = Number.isFinite(rawTimeout) ? Math.max(1000, Math.min(60000, rawTimeout)) : 25000;
      let event = dequeueInboundEvent();
      let waited = 0;
      while (!event && waited < timeoutMs) {
        await new Promise((r) => setTimeout(r, 200));
        waited += 200;
        event = dequeueInboundEvent();
      }
      return jsonResponse(res, 200, { event });
    }

    // GET /media/:id — serve short-lived media cached from inbound voice notes
    if (req.method === 'GET' && path.startsWith('/media/')) {
      const mediaId = path.slice('/media/'.length).trim();
      if (!mediaId) return jsonResponse(res, 400, { error: 'Missing media id' });
      const item = readMedia(mediaId);
      if (!item) return jsonResponse(res, 404, { error: 'Media not found or expired' });
      res.writeHead(200, {
        'Content-Type': item.mimeType || 'application/octet-stream',
        'Content-Length': item.data.length,
        'Access-Control-Allow-Origin': '*',
      });
      return res.end(item.data);
    }

    // POST /message/send — send outgoing message via Baileys
    if (req.method === 'POST' && path === '/message/send') {
      const body = await parseBody(req);
      const { to, text } = body;

      if (!to || !text) {
        return jsonResponse(res, 400, { error: 'Missing "to" or "text" field' });
      }

      await sendMessage(to, text);
      return jsonResponse(res, 200, { success: true, message: 'Sent' });
    }

    // POST /message/send_voice — send outgoing audio/voice via Baileys
    if (req.method === 'POST' && path === '/message/send_voice') {
      const body = await parseBody(req);
      const { to, voice_url: voiceUrl, ptt } = body;
      if (!to || !voiceUrl) {
        return jsonResponse(res, 400, { error: 'Missing "to" or "voice_url" field' });
      }
      await sendVoiceMessage(to, voiceUrl, ptt !== false);
      return jsonResponse(res, 200, { success: true, message: 'Voice sent' });
    }

    // GET /health — health check
    if (req.method === 'GET' && path === '/health') {
      return jsonResponse(res, 200, {
        status: 'ok',
        connected: connStatus === 'connected',
        session_id: sessionId || null,
        self_chat_mode: SELF_CHAT_MODE,
      });
    }

    // 404
    jsonResponse(res, 404, { error: 'Not found' });
  } catch (err) {
    console.error(`[gateway] ${req.method} ${path} error:`, err.message);
    jsonResponse(res, 500, { error: err.message });
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[gateway] WhatsApp Web gateway listening on http://127.0.0.1:${PORT}`);
  console.log(`[gateway] OpenFang URL: ${OPENFANG_URL}`);
  console.log(`[gateway] Self-chat mode: ${SELF_CHAT_MODE ? 'enabled' : 'disabled'}`);
  console.log('[gateway] Waiting for POST /login/start and /events/poll ...');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n[gateway] Shutting down...');
  if (sock) sock.end();
  server.close(() => process.exit(0));
});

process.on('SIGTERM', () => {
  if (sock) sock.end();
  server.close(() => process.exit(0));
});
