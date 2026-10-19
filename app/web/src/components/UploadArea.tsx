'use client';

import { useCallback, useRef, useState } from 'react';
import Image from 'next/image';

interface UploadAreaProps {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function UploadArea({ onFile, disabled }: UploadAreaProps) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    onFile(file);
  }, [onFile]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  }, []);

  const onClick = () => inputRef.current?.click();
  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="relative w-full">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={onChange}
        disabled={disabled}
        className="hidden"
        id="image-upload"
      />
      <label
        htmlFor="image-upload"
        onClick={onClick}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        className={`block w-full cursor-pointer transition-all ${
          dragActive ? 'ring-4 ring-green-500 bg-green-50' : 'hover:bg-gray-50'
        } border-2 border-dashed rounded-2xl p-8 text-center ${
          disabled ? 'opacity-50 cursor-not-allowed' : ''
        }`}
      >
        {preview ? (
          <div className="relative aspect-video rounded-lg overflow-hidden bg-gray-100">
            <Image
              src={preview}
              alt="Preview"
              fill
              className="object-cover"
              sizes="100vw"
            />
          </div>
        ) : (
          <div className="space-y-3">
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <div className="text-lg font-medium text-gray-900">Drop a leaf photo or click to browse</div>
            <div className="text-sm text-gray-500">
              Supports JPG, PNG, WebP • Max 10MB • Camera capture on mobile
            </div>
          </div>
        )}
      </label>
    </div>
  );
}