import asyncio
import websockets
import pyaudio
import json
import threading

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4000  # Send 4000 frames (0.25 seconds) at a time

async def mic_stream_client():
    uri = "ws://localhost:8000/ws/stream"
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Connecting to Voice AI Backend...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Start speaking (Press Ctrl+C to stop).")
            
            # Start a thread/task to receive messages from the server
            async def receive_messages():
                try:
                    while True:
                        response = await websocket.recv()
                        data = json.loads(response)
                        if data.get("type") == "transcript":
                            print(f"\n[You]: {data.get('text')}")
                        elif data.get("type") == "ai_response":
                            print(f"[AI]:  {data.get('text')}")
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by server.")
            
            asyncio.create_task(receive_messages())
            
            # Main loop: send audio
            while True:
                # Read audio chunk
                # exception_on_overflow=False prevents crashes if we don't read fast enough
                data = stream.read(CHUNK, exception_on_overflow=False)
                await websocket.send(data)
                # Small sleep to yield to the event loop
                await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(mic_stream_client())
