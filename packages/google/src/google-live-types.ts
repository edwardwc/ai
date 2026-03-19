import {
  Experimental_LiveModelV1,
  Experimental_LiveSessionV1,
  LanguageModelV4CallOptions,
  SharedV4Warning,
} from '@ai-sdk/provider';

export type GoogleLiveTool =
  | {
      functionDeclarations: Array<{
        name: string;
        description: string;
        parameters: unknown;
      }>;
    }
  | Record<string, any>;

export type GoogleLiveConfig = {
  generationConfig?: Record<string, unknown>;
  responseModalities?: string[];
  temperature?: number;
  topP?: number;
  topK?: number;
  maxOutputTokens?: number;
  seed?: number;
  mediaResolution?: string;
  speechConfig?: Record<string, unknown>;
  thinkingConfig?: Record<string, unknown>;
  enableAffectiveDialog?: boolean;
  systemInstruction?: string | Record<string, unknown>;
  tools?: GoogleLiveTool[];
  sessionResumption?: {
    handle?: string;
  };
  inputAudioTranscription?: Record<string, unknown>;
  outputAudioTranscription?: Record<string, unknown>;
  realtimeInputConfig?: Record<string, unknown>;
  contextWindowCompression?: Record<string, unknown>;
  proactivity?: Record<string, unknown>;
};

export type GoogleLiveConnectOptions = {
  authToken?: string;
  config?: GoogleLiveConfig;
  abortSignal?: AbortSignal;
};

export type GoogleLiveAuthToken = {
  name: string;
  expireTime?: string;
  newSessionExpireTime?: string;
};

export type GoogleLiveCreateAuthTokenOptions = {
  uses?: number;
  expireTime?: string;
  newSessionExpireTime?: string;
  config?: GoogleLiveConfig;
  lockedFields?: string[];
  abortSignal?: AbortSignal;
};

export type GoogleLivePreparedToolsResult = {
  tools: GoogleLiveTool[] | undefined;
  warnings: SharedV4Warning[];
};

export type GoogleLiveWebSocketLike = {
  onopen: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onerror: ((event: { error?: unknown }) => void) | null;
  onclose: ((event: { code?: number; reason?: string }) => void) | null;
  send(data: string): void;
  close(code?: number, reason?: string): void;
};

export type GoogleLiveWebSocketFactory = (
  url: string,
) => GoogleLiveWebSocketLike;

export interface GoogleLiveSession extends Experimental_LiveSessionV1 {}

export interface GoogleLiveModel extends Experimental_LiveModelV1 {
  connect(options?: GoogleLiveConnectOptions): Promise<GoogleLiveSession>;
  createAuthToken(
    options?: GoogleLiveCreateAuthTokenOptions,
  ): Promise<GoogleLiveAuthToken>;
  prepareTools(options: {
    tools: LanguageModelV4CallOptions['tools'];
  }): GoogleLivePreparedToolsResult;
}
