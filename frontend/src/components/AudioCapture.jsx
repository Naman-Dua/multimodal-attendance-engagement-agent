import React, { useEffect, useRef, useState } from 'react';

const AudioCapture = ({ active, onMetrics }) => {
  const [isRecording, setIsRecording] = useState(false);
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (active) {
      startCapture();
    } else {
      stopCapture();
    }
    return () => stopCapture();
  }, [active]);

  const startCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsRecording(true);

      // Connect WS
      const ws = new WebSocket('ws://localhost:8000/ws/audio');
      wsRef.current = ws;
