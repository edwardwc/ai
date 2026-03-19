import type { ModelMessage } from '@ai-sdk/provider-utils';
import type { TypedToolCall } from '../generate-text/tool-call';
import type { TypedToolResult } from '../generate-text/tool-result';
import type { ToolSet } from '../generate-text/tool-set';
import type { LanguageModelUsage } from '../types/usage';

export interface OnLiveStartEvent<TOOLS extends ToolSet = ToolSet> {
  readonly callId: string;
  readonly provider: string;
  readonly modelId: string;
  readonly tools: TOOLS | undefined;
  readonly functionId: string | undefined;
  readonly metadata: Record<string, unknown> | undefined;
  readonly experimental_context: unknown;
  readonly abortSignal: AbortSignal | undefined;
  readonly config: unknown;
  readonly isEnabled: boolean | undefined;
  readonly recordInputs: boolean | undefined;
  readonly recordOutputs: boolean | undefined;
}

export interface OnLiveTurnStartEvent {
  readonly callId: string;
  readonly turnNumber: number;
  readonly provider: string;
  readonly modelId: string;
  readonly userTranscript: string | undefined;
  readonly functionId: string | undefined;
  readonly metadata: Record<string, unknown> | undefined;
  readonly experimental_context: unknown;
}

export interface OnLiveTurnFinishEvent<TOOLS extends ToolSet = ToolSet> {
  readonly callId: string;
  readonly turnNumber: number;
  readonly provider: string;
  readonly modelId: string;
  readonly userTranscript: string | undefined;
  readonly assistantTranscript: string | undefined;
  readonly toolCalls: Array<TypedToolCall<TOOLS>>;
  readonly toolResults: Array<TypedToolResult<TOOLS>>;
  readonly messages: Array<ModelMessage>;
  readonly usage: LanguageModelUsage;
  readonly functionId: string | undefined;
  readonly metadata: Record<string, unknown> | undefined;
  readonly experimental_context: unknown;
  readonly finishReason: 'stop' | 'error' | 'other';
}

export interface OnLiveFinishEvent<TOOLS extends ToolSet = ToolSet> {
  readonly callId: string;
  readonly provider: string;
  readonly modelId: string;
  readonly turnCount: number;
  readonly usage: LanguageModelUsage;
  readonly lastUserTranscript: string | undefined;
  readonly lastAssistantTranscript: string | undefined;
  readonly functionId: string | undefined;
  readonly metadata: Record<string, unknown> | undefined;
  readonly experimental_context: unknown;
  readonly toolCalls: Array<TypedToolCall<TOOLS>>;
  readonly toolResults: Array<TypedToolResult<TOOLS>>;
}
