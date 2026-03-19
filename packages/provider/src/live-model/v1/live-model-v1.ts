import { SharedV4ProviderMetadata } from '../../shared/v4/shared-v4-provider-metadata';

export type LiveModelV1MediaData = string | Uint8Array;

export type LiveModelV1TranscriptMessage = {
  type: 'transcript';
  role: 'user' | 'assistant';
  text: string;
  format: 'snapshot' | 'delta';
  source: 'input-audio' | 'output-audio' | 'model';
  providerMetadata?: SharedV4ProviderMetadata;
};

export type LiveModelV1AudioMessage = {
  type: 'audio';
  role: 'assistant';
  data: LiveModelV1MediaData;
  mimeType?: string;
  providerMetadata?: SharedV4ProviderMetadata;
};

export type LiveModelV1ToolCall = {
  toolCallId: string;
  toolName: string;
  input: unknown;
  providerExecuted?: boolean;
  providerMetadata?: SharedV4ProviderMetadata;
};

export type LiveModelV1ServerMessage =
  | LiveModelV1TranscriptMessage
  | LiveModelV1AudioMessage
  | {
      type: 'tool-call';
      toolCalls: LiveModelV1ToolCall[];
      providerMetadata?: SharedV4ProviderMetadata;
    }
  | {
      type: 'tool-call-cancel';
      toolCallIds: string[];
      providerMetadata?: SharedV4ProviderMetadata;
    }
  | {
      type: 'turn-complete';
      providerMetadata?: SharedV4ProviderMetadata;
    }
  | {
      type: 'interrupted';
      providerMetadata?: SharedV4ProviderMetadata;
    }
  | {
      type: 'usage';
      usage: {
        inputTokens?: number;
        outputTokens?: number;
        totalTokens?: number;
      };
      providerMetadata?: SharedV4ProviderMetadata;
    }
  | {
      type: 'session-resumption';
      resumable: boolean;
      handle?: string;
      providerMetadata?: SharedV4ProviderMetadata;
    }
  | {
      type: 'go-away';
      timeLeft?: string;
      providerMetadata?: SharedV4ProviderMetadata;
    }
  | {
      type: 'warning';
      warning: string;
      providerMetadata?: SharedV4ProviderMetadata;
    }
  | {
      type: 'raw';
      value: unknown;
    };

export type LiveModelV1ClientContent = {
  turns: Array<{
    role: 'user' | 'assistant';
    content: Array<
      | {
          type: 'text';
          text: string;
        }
      | {
          type: 'media';
          mediaType: string;
          data: LiveModelV1MediaData;
        }
    >;
  }>;
  turnComplete?: boolean;
};

export type LiveModelV1RealtimeInput = {
  text?: string;
  audio?: {
    data: LiveModelV1MediaData;
    mimeType: string;
  };
  video?: {
    data: LiveModelV1MediaData;
    mimeType: string;
  };
  audioStreamEnd?: boolean;
};

export type LiveModelV1ToolResponse = {
  toolResults: Array<{
    toolCallId: string;
    toolName: string;
    output?: unknown;
    error?: string;
    providerMetadata?: SharedV4ProviderMetadata;
  }>;
};

export type LiveSessionV1 = {
  readonly specificationVersion: 'v1';
  readonly provider: string;
  readonly modelId: string;
  readonly messages: AsyncIterable<LiveModelV1ServerMessage>;
  readonly closed: PromiseLike<{
    code?: number;
    reason?: string;
    error?: unknown;
  }>;
  sendClientContent(
    content: LiveModelV1ClientContent,
  ): PromiseLike<void> | void;
  sendRealtimeInput(
    input: LiveModelV1RealtimeInput,
  ): PromiseLike<void> | void;
  sendToolResponse(
    response: LiveModelV1ToolResponse,
  ): PromiseLike<void> | void;
  close(): PromiseLike<void> | void;
};

export type LiveModelV1 = {
  readonly specificationVersion: 'v1';
  readonly provider: string;
  readonly modelId: string;
  connect(options?: {
    config?: unknown;
    abortSignal?: AbortSignal;
  }): PromiseLike<LiveSessionV1>;
};
