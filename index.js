import express from 'express';
import cors from 'cors';
import http from 'http';
import { WebSocketServer } from 'ws';
import fs from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import dotenv from 'dotenv';

dotenv.config(); // Load environment variables from .env

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

const ELEVEN_LABS_API_KEY = process.env.ELEVEN_LABS_API_KEY;
const ELEVEN_LABS_URL = 'https://api.elevenlabs.io/v1';

if (!ELEVEN_LABS_API_KEY) {
  console.error('❌ ELEVEN_LABS_API_KEY is missing. Set it in your .env file.');
  process.exit(1);
}

// Directories
const publicDir = path.join(__dirname, 'public');
const uploadDir = path.join(__dirname, 'uploads');

await fs.mkdir(publicDir, { recursive: true });
await fs.mkdir(uploadDir, { recursive: true });

app.use(cors({ origin: 'http://localhost:3000' })); // Allow frontend requests
app.use(express.json());
app.use('/public', express.static(publicDir)); // Serve generated audio files

/** 🎤 WebSocket for Real-time Transcription */
wss.on('connection', (ws) => {
  console.log('✅ WebSocket connected');

  ws.on('message', async (message) => {
    try {
      const { audioChunk, diarization } = JSON.parse(message);
      if (!audioChunk) throw new Error('Invalid audio data received.');

      const audioBuffer = Buffer.from(audioChunk, 'base64');
      const audioFilePath = path.join(uploadDir, `audio_${Date.now()}.wav`);
      await fs.writeFile(audioFilePath, audioBuffer);
      console.log(`📁 Saved audio chunk: ${audioFilePath}`);

      const transcription = await processAudioWithWhisper(audioFilePath, diarization);
      console.log(`📝 Transcription: ${transcription}`);
      ws.send(JSON.stringify({ transcription }));

      // Delete file after processing
      setTimeout(async () => {
        try {
          await fs.unlink(audioFilePath);
          console.log(`🗑️ Deleted file: ${audioFilePath}`);
        } catch (err) {
          console.error(`⚠️ Failed to delete file: ${err.message}`);
        }
      }, 5000);
    } catch (error) {
      console.error(`❌ Error processing message: ${error.message}`);
      ws.send(JSON.stringify({ error: error.message }));
    }
  });

  ws.on('close', () => console.log('⚡ WebSocket disconnected'));
});

/** 🎙️ Fetch Available Voices from Eleven Labs */
app.get('/voices', async (req, res) => {
  try {
    const response = await fetch(`${ELEVEN_LABS_URL}/voices`, {
      headers: { 'xi-api-key': ELEVEN_LABS_API_KEY },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error: ${errorText}`);
    }

    const data = await response.json();
    res.json(data.voices);
  } catch (error) {
    console.error(`❌ Failed to fetch voices: ${error.message}`);
    res.status(500).json({ error: 'Failed to fetch voices' });
  }
});

/** 🗣️ AI Voice Generation using Eleven Labs */
app.post('/generate-voice', async (req, res) => {
  try {
    const { text, voiceId } = req.body;
    if (!text || !voiceId) throw new Error('Text and voiceId are required');

    const response = await fetch(`${ELEVEN_LABS_URL}/text-to-speech/${voiceId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'accept': 'audio/mpeg',
      },
      body: JSON.stringify({
        text,
        model_id: "eleven_monolingual_v1",
        voice_settings: { stability: 0.75, similarity_boost: 0.9 },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`AI voice API error: ${errorText}`);
    }

    const audioFileName = `output_${Date.now()}.mp3`;
    const audioFilePath = path.join(publicDir, audioFileName);
    const audioBuffer = await response.arrayBuffer();
    await fs.writeFile(audioFilePath, Buffer.from(audioBuffer));

    res.json({ audioUrl: `/public/${audioFileName}` });
  } catch (error) {
    console.error(`❌ Error generating voice: ${error.message}`);
    res.status(500).json({ error: error.message });
  }
});

/** 🎧 Process Audio with Whisper */
const processAudioWithWhisper = (audioFilePath, useDiarization) => {
  return new Promise((resolve, reject) => {
    console.log(`🔄 Processing audio file: ${audioFilePath}`);

    const pythonScript = useDiarization ? 'transcribe_with_diarization.py' : 'transcribe.py';
    const pythonProcess = spawn('python3', [path.join(__dirname, pythonScript), audioFilePath]);

    let transcriptionResult = '';
    let errorOccurred = false;

    pythonProcess.stdout.on('data', (data) => (transcriptionResult += data.toString()));

    pythonProcess.stderr.on('data', (data) => {
      errorOccurred = true;
      console.error(`🐍 Python error: ${data.toString()}`);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0 || errorOccurred) {
        return reject(new Error(`Python process failed with exit code ${code}`));
      }

      try {
        const parsedResult = JSON.parse(transcriptionResult.trim());
        resolve(parsedResult.transcription || '');
      } catch (error) {
        reject(new Error('Failed to parse Whisper output'));
      }
    });
  });
};

// 🚀 Start Server
server.listen(5001, () => console.log('✅ Server running at http://localhost:5001'));
