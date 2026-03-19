import {
  Experimental_LiveModelV1ServerMessage,
  Experimental_LiveSessionV1,
} from '@ai-sdk/provider';
import { createIdGenerator } from '@ai-sdk/provider-utils';
import { executeToolCall } from '../generate-text/execute-tool-call';
import { ToolSet } from '../generate-text/tool-set';
import { TypedToolCall } from '../generate-text/tool-call';
import { getGlobalTelemetryIntegration } from '../telemetry/get-global-telemetry-integration';
import { TelemetrySettings } from '../telemetry/telemetry-settings';
import {
  addLanguageModelUsage,
  createNullLanguageModelUsage,
  LanguageModelUsage,
} from '../types/usage';
import { ModelMessage } from '../prompt';

const createCallId = createIdGenerator({ prefix: 'ailive', size: 24 });

export type ObservedLiveSessionMessage<TOOLS extends ToolSet = ToolSet> =
  | Experimental_LiveModelV1ServerMessage
  | {
      type: 'tool-output-available';
      toolCall: TypedToolCall<TOOLS>;
      output: unknown;
      preliminary?: boolean;
    }
  | {
      type: 'tool-output-error';
      toolCall: TypedToolCall<TOOLS>;
      error: unknown;
      errorText: string;
    };

export type ObservedLiveSession<TOOLS extends ToolSet = ToolSet> =
  Experimental_LiveSessionV1 & {
    readonly callId: string;
    readonly observedMessages: AsyncIterable<ObservedLiveSessionMessage<TOOLS>>;
  };

class AsyncMessageQueue<T> implements AsyncIterable<T> {
  private readonly values: T[] = [];
  private readonly waiters: Array<{
    resolve: (value: IteratorResult<T>) => void;
    reject: (error: unknown) => void;
  }> = [];
  private closed = false;
  private error: unknown;

  push(value: T): void {
    if (this.closed) {
      return;
    }

    const waiter = this.waiters.shift();
    if (waiter) {
      waiter.resolve({ done: false, value });
      return;
    }

    this.values.push(value);
  }

  close(error?: unknown): void {
    if (this.closed) {
      return;
    }

    this.closed = true;
    this.error = error;

    while (this.waiters.length > 0) {
      const waiter = this.waiters.shift()!;
      if (error != null) {
        waiter.reject(error);
      } else {
        waiter.resolve({ done: true, value: undefined as never });
      }
    }
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: () => {
        if (this.values.length > 0) {
          return Promise.resolve({
            done: false,
            value: this.values.shift()!,
          });
        }

        if (this.closed) {
          if (this.error != null) {
            return Promise.reject(this.error);
          }

          return Promise.resolve({
            done: true,
            value: undefined as never,
          });
        }

        return new Promise<IteratorResult<T>>((resolve, reject) => {
          this.waiters.push({ resolve, reject });
        });
      },
    };
  }
}

function mergeTranscript(
  existingText: string | undefined,
  incomingText: string,
  format: 'snapshot' | 'delta',
): string {
  const current = existingText ?? '';
  if (format === 'delta') {
    return `${current}${incomingText}`;
  }

  if (incomingText.startsWith(current)) {
    return incomingText;
  }

  if (current.endsWith(incomingText)) {
    return current;
  }

  return incomingText;
}

function toLanguageModelUsage(
  usage: Experimental_LiveModelV1ServerMessage extends infer _T
    ? { inputTokens?: number; outputTokens?: number; totalTokens?: number }
    : never,
): LanguageModelUsage {
  return {
    inputTokens: usage.inputTokens,
    inputTokenDetails: {
      noCacheTokens: undefined,
      cacheReadTokens: undefined,
      cacheWriteTokens: undefined,
    },
    outputTokens: usage.outputTokens,
    outputTokenDetails: {
      textTokens: usage.outputTokens,
      reasoningTokens: undefined,
    },
    totalTokens: usage.totalTokens,
    reasoningTokens: undefined,
    cachedInputTokens: undefined,
    raw: undefined,
  };
}

function toTypedToolCall<TOOLS extends ToolSet>({
  toolCallId,
  toolName,
  input,
  tools,
}: {
  toolCallId: string;
  toolName: string;
  input: unknown;
  tools: TOOLS | undefined;
}): TypedToolCall<TOOLS> {
  if (tools?.[toolName] != null) {
    return {
      type: 'tool-call',
      toolCallId,
      toolName: toolName as keyof TOOLS & string,
      input: input as any,
      dynamic: false,
    } as TypedToolCall<TOOLS>;
  }

  return {
    type: 'tool-call',
    toolCallId,
    toolName,
    input,
    dynamic: true,
    invalid: true,
    error: new Error(`Unknown live tool: ${toolName}`),
  } as TypedToolCall<TOOLS>;
}

export function experimental_observeLiveSession<TOOLS extends ToolSet>({
  session,
  tools,
  experimental_context,
  experimental_telemetry: telemetry,
  config,
  abortSignal,
}: {
  session: Experimental_LiveSessionV1;
  tools?: TOOLS;
  experimental_context?: unknown;
  experimental_telemetry?: TelemetrySettings;
  config?: unknown;
  abortSignal?: AbortSignal;
}): ObservedLiveSession<TOOLS> {
  const callId = createCallId();
  const observedQueue = new AsyncMessageQueue<ObservedLiveSessionMessage<TOOLS>>();
  const globalTelemetry = getGlobalTelemetryIntegration<TOOLS>()({
    integrations: telemetry?.integrations,
  });

  const conversationMessages: ModelMessage[] = [];
  let turnNumber = 0;
  let turnActive = false;
  let userTranscript: string | undefined;
  let assistantTranscript: string | undefined;
  let currentUsage = createNullLanguageModelUsage();
  let totalUsage = createNullLanguageModelUsage();
  let currentToolCalls: Array<TypedToolCall<TOOLS>> = [];
  let currentToolResults: any[] = [];
  let lastUserTranscript: string | undefined;
  let lastAssistantTranscript: string | undefined;

  void globalTelemetry.onLiveStart?.({
    callId,
    provider: session.provider,
    modelId: session.modelId,
    tools,
    functionId: telemetry?.functionId,
    metadata: telemetry?.metadata as Record<string, unknown> | undefined,
    experimental_context,
    abortSignal,
    config,
    isEnabled: telemetry?.isEnabled,
    recordInputs: telemetry?.recordInputs,
    recordOutputs: telemetry?.recordOutputs,
  });

  function ensureTurnStarted() {
    if (turnActive) {
      return;
    }

    turnActive = true;
    turnNumber += 1;

    void globalTelemetry.onLiveTurnStart?.({
      callId,
      turnNumber,
      provider: session.provider,
      modelId: session.modelId,
      userTranscript,
      functionId: telemetry?.functionId,
      metadata: telemetry?.metadata as Record<string, unknown> | undefined,
      experimental_context,
    });
  }

  function finalizeTurn(finishReason: 'stop' | 'error' | 'other') {
    if (!turnActive) {
      return;
    }

    const turnMessages: ModelMessage[] = [];
    if (userTranscript) {
      turnMessages.push({
        role: 'user',
        content: [{ type: 'text', text: userTranscript }],
      });
      conversationMessages.push(turnMessages[turnMessages.length - 1]);
      lastUserTranscript = userTranscript;
    }

    if (assistantTranscript) {
      turnMessages.push({
        role: 'assistant',
        content: [{ type: 'text', text: assistantTranscript }],
      });
      conversationMessages.push(turnMessages[turnMessages.length - 1]);
      lastAssistantTranscript = assistantTranscript;
    }

    void globalTelemetry.onLiveTurnFinish?.({
      callId,
      turnNumber,
      provider: session.provider,
      modelId: session.modelId,
      userTranscript,
      assistantTranscript,
      toolCalls: currentToolCalls,
      toolResults: currentToolResults,
      messages: conversationMessages,
      usage: currentUsage,
      functionId: telemetry?.functionId,
      metadata: telemetry?.metadata as Record<string, unknown> | undefined,
      experimental_context,
      finishReason,
    });

    totalUsage = addLanguageModelUsage(totalUsage, currentUsage);
    userTranscript = undefined;
    assistantTranscript = undefined;
    currentUsage = createNullLanguageModelUsage();
    currentToolCalls = [];
    currentToolResults = [];
    turnActive = false;
  }

  void (async () => {
    try {
      for await (const message of session.messages) {
        observedQueue.push(message);

        switch (message.type) {
          case 'transcript':
            ensureTurnStarted();
            if (message.role === 'user') {
              userTranscript = mergeTranscript(
                userTranscript,
                message.text,
                message.format,
              );
            } else {
              assistantTranscript = mergeTranscript(
                assistantTranscript,
                message.text,
                message.format,
              );
            }
            break;

          case 'usage':
            currentUsage = addLanguageModelUsage(
              currentUsage,
              toLanguageModelUsage(message.usage),
            );
            break;

          case 'tool-call': {
            ensureTurnStarted();
            const toolResults = await Promise.all(
              message.toolCalls.map(async toolCall => {
                const typedToolCall = toTypedToolCall({
                  ...toolCall,
                  tools,
                });
                currentToolCalls.push(typedToolCall);

                const result = await executeToolCall({
                  toolCall: typedToolCall,
                  tools,
                  telemetry,
                  callId,
                  messages: conversationMessages,
                  abortSignal,
                  experimental_context,
                  stepNumber: turnNumber,
                  provider: session.provider,
                  modelId: session.modelId,
                  onPreliminaryToolResult: preliminaryResult => {
                    observedQueue.push({
                      type: 'tool-output-available',
                      toolCall: typedToolCall,
                      output: preliminaryResult.output,
                      preliminary: true,
                    });
                  },
                  onToolCallStart: globalTelemetry.onToolCallStart as any,
                  onToolCallFinish: globalTelemetry.onToolCallFinish as any,
                  executeToolInTelemetryContext: globalTelemetry.executeTool,
                });

                if (result == null) {
                  observedQueue.push({
                    type: 'tool-output-error',
                    toolCall: typedToolCall,
                    error: new Error(`Tool ${toolCall.toolName} is not executable.`),
                    errorText: `Tool ${toolCall.toolName} is not executable.`,
                  });

                  return {
                    toolCallId: toolCall.toolCallId,
                    toolName: toolCall.toolName,
                    error: `Tool ${toolCall.toolName} is not executable.`,
                  };
                }

                currentToolResults.push(result as any);

                if (result.type === 'tool-result') {
                  observedQueue.push({
                    type: 'tool-output-available',
                    toolCall: typedToolCall,
                    output: result.output,
                    preliminary: result.preliminary,
                  });

                  return {
                    toolCallId: toolCall.toolCallId,
                    toolName: toolCall.toolName,
                    output: result.output,
                  };
                }

                observedQueue.push({
                  type: 'tool-output-error',
                  toolCall: typedToolCall,
                  error: result.error,
                  errorText:
                    result.error instanceof Error
                      ? result.error.message
                      : 'Tool execution failed.',
                });

                return {
                  toolCallId: toolCall.toolCallId,
                  toolName: toolCall.toolName,
                  error:
                    result.error instanceof Error
                      ? result.error.message
                      : 'Tool execution failed.',
                };
              }),
            );

            session.sendToolResponse({
              toolResults,
            });
            break;
          }

          case 'turn-complete':
            finalizeTurn('stop');
            break;

          case 'interrupted':
            finalizeTurn('other');
            break;
        }
      }

      finalizeTurn('stop');

      await session.closed;

      void globalTelemetry.onLiveFinish?.({
        callId,
        provider: session.provider,
        modelId: session.modelId,
        turnCount: turnNumber,
        usage: totalUsage,
        lastUserTranscript,
        lastAssistantTranscript,
        functionId: telemetry?.functionId,
        metadata: telemetry?.metadata as Record<string, unknown> | undefined,
        experimental_context,
        toolCalls: currentToolCalls,
        toolResults: currentToolResults,
      });

      observedQueue.close();
    } catch (error) {
      finalizeTurn('error');
      void globalTelemetry.onError?.({ callId, error });
      observedQueue.close(error);
    }
  })();

  return {
    ...session,
    callId,
    observedMessages: observedQueue,
  };
}
