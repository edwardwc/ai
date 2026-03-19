import {
  Experimental_LiveModelV1ServerMessage,
  Experimental_LiveSessionV1,
  LanguageModelV4CallOptions,
} from '@ai-sdk/provider';
import {
  createJsonResponseHandler,
  FetchFunction,
  postJsonToApi,
  Resolvable,
  resolve,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';
import { googleFailedResponseHandler } from './google-error';
import {
  GoogleLiveAuthToken,
  GoogleLiveConfig,
  GoogleLiveConnectOptions,
  GoogleLiveCreateAuthTokenOptions,
  GoogleLiveModel,
  GoogleLivePreparedToolsResult,
  GoogleLiveTool,
  GoogleLiveWebSocketFactory,
  GoogleLiveWebSocketLike,
} from './google-live-types';
import { GoogleGenerativeAIModelId } from './google-generative-ai-options';
import { prepareTools } from './google-prepare-tools';

const authTokenResponseSchema = z.object({
  name: z.string(),
  expireTime: z.string().optional(),
  newSessionExpireTime: z.string().optional(),
});

type GoogleLiveModelConfig = {
  provider: string;
  modelId: GoogleGenerativeAIModelId;
  liveBaseURL: string;
  liveApiVersion: string;
  headers: Resolvable<Record<string, string | undefined>>;
  apiKey: () => string | undefined;
  fetch?: FetchFunction;
  createWebSocket?: GoogleLiveWebSocketFactory;
  generateId: () => string;
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

  end(error?: unknown): void {
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

function getWebSocketFactory(
  factory: GoogleLiveWebSocketFactory | undefined,
): GoogleLiveWebSocketFactory {
  if (factory) {
    return factory;
  }

  const WebSocketCtor = (globalThis as any).WebSocket as
    | (new (url: string) => GoogleLiveWebSocketLike)
    | undefined;

  if (WebSocketCtor == null) {
    throw new Error(
      'No WebSocket implementation available. Provide createWebSocket in createGoogleGenerativeAI({ ... }).',
    );
  }

  return url => new WebSocketCtor(url);
}

function normalizeModelId(modelId: string): string {
  return modelId.startsWith('models/') ? modelId : `models/${modelId}`;
}

function toWsBaseUrl(baseURL: string): string {
  if (baseURL.startsWith('https://')) {
    return `wss://${baseURL.slice('https://'.length)}`;
  }

  if (baseURL.startsWith('http://')) {
    return `ws://${baseURL.slice('http://'.length)}`;
  }

  return baseURL;
}

function maybeToBase64(data: string | Uint8Array): string {
  if (typeof data === 'string') {
    return data;
  }

  const BufferCtor = (globalThis as any).Buffer as
    | {
        from(value: Uint8Array): { toString(encoding: string): string };
      }
    | undefined;

  if (BufferCtor != null) {
    return BufferCtor.from(data).toString('base64');
  }

  let binary = '';
  for (const byte of data) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

function toGoogleContent(value: string | Record<string, unknown>) {
  if (typeof value === 'string') {
    return {
      parts: [{ text: value }],
    };
  }

  return value;
}

function mapGenerationConfig(config: GoogleLiveConfig) {
  const generationConfig =
    config.generationConfig != null ? { ...config.generationConfig } : {};

  const generationFields = {
    responseModalities: config.responseModalities,
    temperature: config.temperature,
    topP: config.topP,
    topK: config.topK,
    maxOutputTokens: config.maxOutputTokens,
    seed: config.seed,
    mediaResolution: config.mediaResolution,
    speechConfig: config.speechConfig,
    thinkingConfig: config.thinkingConfig,
    enableAffectiveDialog: config.enableAffectiveDialog,
  };

  for (const [key, value] of Object.entries(generationFields)) {
    if (value !== undefined) {
      generationConfig[key] = value;
    }
  }

  return Object.keys(generationConfig).length > 0 ? generationConfig : undefined;
}

function buildSetupPayload(
  modelId: string,
  config: GoogleLiveConfig | undefined,
) {
  const setup: Record<string, unknown> = {
    model: normalizeModelId(modelId),
  };

  if (config == null) {
    return { setup };
  }

  const generationConfig = mapGenerationConfig(config);
  if (generationConfig != null) {
    setup.generationConfig = generationConfig;
  }

  if (config.systemInstruction != null) {
    setup.systemInstruction = toGoogleContent(config.systemInstruction);
  }

  if (config.tools != null) {
    setup.tools = config.tools;
  }

  if (config.sessionResumption != null) {
    setup.sessionResumption = config.sessionResumption;
  }

  if (config.inputAudioTranscription != null) {
    setup.inputAudioTranscription = config.inputAudioTranscription;
  }

  if (config.outputAudioTranscription != null) {
    setup.outputAudioTranscription = config.outputAudioTranscription;
  }

  if (config.realtimeInputConfig != null) {
    setup.realtimeInputConfig = config.realtimeInputConfig;
  }

  if (config.contextWindowCompression != null) {
    setup.contextWindowCompression = config.contextWindowCompression;
  }

  if (config.proactivity != null) {
    setup.proactivity = config.proactivity;
  }

  return { setup };
}

function configFieldMask(config: GoogleLiveConfig): string[] {
  const mask: string[] = [];

  const generationFieldMap: Record<string, string> = {
    responseModalities: 'generationConfig.responseModalities',
    temperature: 'generationConfig.temperature',
    topP: 'generationConfig.topP',
    topK: 'generationConfig.topK',
    maxOutputTokens: 'generationConfig.maxOutputTokens',
    seed: 'generationConfig.seed',
    mediaResolution: 'generationConfig.mediaResolution',
    speechConfig: 'generationConfig.speechConfig',
    thinkingConfig: 'generationConfig.thinkingConfig',
    enableAffectiveDialog: 'generationConfig.enableAffectiveDialog',
  };

  for (const [key, value] of Object.entries(generationFieldMap)) {
    if ((config as Record<string, unknown>)[key] !== undefined) {
      mask.push(value);
    }
  }

  if (config.systemInstruction != null) {
    mask.push('systemInstruction');
  }

  if (config.tools != null) {
    mask.push('tools');
  }

  if (config.sessionResumption != null) {
    mask.push('sessionResumption');
  }

  if (config.inputAudioTranscription != null) {
    mask.push('inputAudioTranscription');
  }

  if (config.outputAudioTranscription != null) {
    mask.push('outputAudioTranscription');
  }

  if (config.realtimeInputConfig != null) {
    mask.push('realtimeInputConfig');
  }

  if (config.contextWindowCompression != null) {
    mask.push('contextWindowCompression');
  }

  if (config.proactivity != null) {
    mask.push('proactivity');
  }

  if (config.generationConfig != null) {
    for (const key of Object.keys(config.generationConfig)) {
      mask.push(`generationConfig.${key}`);
    }
  }

  return Array.from(new Set(mask));
}

function additionalFieldMask(fields: string[] | undefined): string[] {
  if (fields == null) {
    return [];
  }

  return fields.map(field => {
    switch (field) {
      case 'responseModalities':
      case 'temperature':
      case 'topP':
      case 'topK':
      case 'maxOutputTokens':
      case 'seed':
      case 'speechConfig':
      case 'mediaResolution':
      case 'thinkingConfig':
      case 'enableAffectiveDialog':
        return `generationConfig.${field}`;
      default:
        return field;
    }
  });
}

function buildAuthTokenPayload(
  modelId: string,
  options: GoogleLiveCreateAuthTokenOptions | undefined,
) {
  const payload: Record<string, unknown> = {};

  if (options?.uses != null) {
    payload.uses = options.uses;
  }

  if (options?.expireTime != null) {
    payload.expireTime = options.expireTime;
  }

  if (options?.newSessionExpireTime != null) {
    payload.newSessionExpireTime = options.newSessionExpireTime;
  }

  if (options?.config != null) {
    payload.bidiGenerateContentSetup = buildSetupPayload(
      modelId,
      options.config,
    );

    if (options.lockedFields != null) {
      const fieldMask = [
        ...configFieldMask(options.config),
        ...additionalFieldMask(options.lockedFields),
      ];
      if (fieldMask.length > 0) {
        payload.fieldMask = fieldMask.join(',');
      }
    }
  } else if (options?.lockedFields != null) {
    const fieldMask = additionalFieldMask(options.lockedFields);
    if (fieldMask.length > 0) {
      payload.fieldMask = fieldMask.join(',');
    }
  }

  return payload;
}

function parseUsage(value: Record<string, unknown>) {
  const inputTokens =
    typeof value.promptTokenCount === 'number'
      ? value.promptTokenCount
      : typeof value.inputTokenCount === 'number'
        ? value.inputTokenCount
        : undefined;

  const outputTokens =
    typeof value.candidatesTokenCount === 'number'
      ? value.candidatesTokenCount
      : typeof value.outputTokenCount === 'number'
        ? value.outputTokenCount
        : undefined;

  const totalTokens =
    typeof value.totalTokenCount === 'number'
      ? value.totalTokenCount
      : undefined;

  return { inputTokens, outputTokens, totalTokens };
}

function normalizeServerMessage(
  raw: Record<string, any>,
  generateId: () => string,
): Experimental_LiveModelV1ServerMessage[] {
  const messages: Experimental_LiveModelV1ServerMessage[] = [];

  if (typeof raw.data === 'string' && raw.data.length > 0) {
    messages.push({
      type: 'audio',
      role: 'assistant',
      data: raw.data,
      mimeType: 'audio/pcm',
    });
  }

  const serverContent = raw.serverContent;
  if (typeof serverContent?.inputTranscription?.text === 'string') {
    messages.push({
      type: 'transcript',
      role: 'user',
      text: serverContent.inputTranscription.text,
      format: 'snapshot',
      source: 'input-audio',
    });
  }

  const modelTurnText = Array.isArray(serverContent?.modelTurn?.parts)
    ? serverContent.modelTurn.parts
        .filter(
          (part: any) =>
            typeof part?.text === 'string' &&
            part.text.length > 0 &&
            part.thought !== true,
        )
        .map((part: any) => part.text)
        .join('')
    : '';

  if (modelTurnText.length > 0) {
    messages.push({
      type: 'transcript',
      role: 'assistant',
      text: modelTurnText,
      format: 'snapshot',
      source: 'model',
    });
  } else if (typeof serverContent?.outputTranscription?.text === 'string') {
    messages.push({
      type: 'transcript',
      role: 'assistant',
      text: serverContent.outputTranscription.text,
      format: 'snapshot',
      source: 'output-audio',
    });
  }

  if (Array.isArray(raw.toolCall?.functionCalls)) {
    messages.push({
      type: 'tool-call',
      toolCalls: raw.toolCall.functionCalls
        .filter((call: any) => typeof call?.name === 'string')
        .map((call: any) => ({
          toolCallId: call.id ?? generateId(),
          toolName: call.name,
          input: call.args ?? {},
        })),
    });
  }

  if (Array.isArray(raw.toolCallCancellation?.ids)) {
    messages.push({
      type: 'tool-call-cancel',
      toolCallIds: raw.toolCallCancellation.ids,
    });
  }

  if (raw.sessionResumptionUpdate != null) {
    messages.push({
      type: 'session-resumption',
      resumable: raw.sessionResumptionUpdate.resumable === true,
      handle:
        typeof raw.sessionResumptionUpdate.newHandle === 'string'
          ? raw.sessionResumptionUpdate.newHandle
          : undefined,
    });
  }

  if (raw.goAway != null) {
    messages.push({
      type: 'go-away',
      timeLeft:
        typeof raw.goAway.timeLeft === 'string' ? raw.goAway.timeLeft : undefined,
    });
  }

  if (serverContent?.interrupted === true) {
    messages.push({ type: 'interrupted' });
  }

  if (serverContent?.turnComplete === true) {
    messages.push({ type: 'turn-complete' });
  }

  if (raw.usageMetadata != null && typeof raw.usageMetadata === 'object') {
    messages.push({
      type: 'usage',
      usage: parseUsage(raw.usageMetadata),
    });
  }

  return messages;
}

class GoogleLiveSessionImpl implements Experimental_LiveSessionV1 {
  readonly specificationVersion = 'v1' as const;
  readonly messages: AsyncIterable<Experimental_LiveModelV1ServerMessage>;
  readonly closed: Promise<{
    code?: number;
    reason?: string;
    error?: unknown;
  }>;

  private closedResolve!: (value: {
    code?: number;
    reason?: string;
    error?: unknown;
  }) => void;

  constructor(
    readonly provider: string,
    readonly modelId: string,
    private readonly socket: GoogleLiveWebSocketLike,
    private readonly queue: AsyncMessageQueue<Experimental_LiveModelV1ServerMessage>,
  ) {
    this.messages = queue;
    this.closed = new Promise(resolve => {
      this.closedResolve = resolve;
    });
  }

  sendClientContent(content: {
    turns: Array<{
      role: 'user' | 'assistant';
      content: Array<
        | { type: 'text'; text: string }
        | { type: 'media'; mediaType: string; data: string | Uint8Array }
      >;
    }>;
    turnComplete?: boolean;
  }): void {
    this.socket.send(
      JSON.stringify({
        clientContent: {
          turns: content.turns.map(turn => ({
            role: turn.role,
            parts: turn.content.map(part =>
              part.type === 'text'
                ? { text: part.text }
                : {
                    inlineData: {
                      mimeType: part.mediaType,
                      data: maybeToBase64(part.data),
                    },
                  },
            ),
          })),
          turnComplete: content.turnComplete ?? true,
        },
      }),
    );
  }

  sendRealtimeInput(input: {
    text?: string;
    audio?: { data: string | Uint8Array; mimeType: string };
    video?: { data: string | Uint8Array; mimeType: string };
    audioStreamEnd?: boolean;
  }): void {
    const realtimeInput: Record<string, unknown> = {};

    if (input.text != null) {
      realtimeInput.text = input.text;
    }

    if (input.audio != null) {
      realtimeInput.audio = {
        data: maybeToBase64(input.audio.data),
        mimeType: input.audio.mimeType,
      };
    }

    if (input.video != null) {
      realtimeInput.video = {
        data: maybeToBase64(input.video.data),
        mimeType: input.video.mimeType,
      };
    }

    if (input.audioStreamEnd != null) {
      realtimeInput.audioStreamEnd = input.audioStreamEnd;
    }

    this.socket.send(JSON.stringify({ realtimeInput }));
  }

  sendToolResponse(response: {
    toolResults: Array<{
      toolCallId: string;
      toolName: string;
      output?: unknown;
      error?: string;
    }>;
  }): void {
    this.socket.send(
      JSON.stringify({
        toolResponse: {
          functionResponses: response.toolResults.map(toolResult => ({
            id: toolResult.toolCallId,
            name: toolResult.toolName,
            response: toolResult.error
              ? { error: toolResult.error }
              : { output: toolResult.output ?? null },
          })),
        },
      }),
    );
  }

  close(): void {
    this.socket.close();
  }

  handleClose(event: { code?: number; reason?: string; error?: unknown }) {
    this.queue.end(event.error);
    this.closedResolve(event);
  }
}

export class GoogleGenerativeAILiveModel implements GoogleLiveModel {
  readonly specificationVersion = 'v1' as const;

  constructor(private readonly config: GoogleLiveModelConfig) {}

  get provider(): string {
    return this.config.provider;
  }

  get modelId(): string {
    return this.config.modelId;
  }

  prepareTools(options: {
    tools: LanguageModelV4CallOptions['tools'];
  }): GoogleLivePreparedToolsResult {
    const result = prepareTools({
      tools: options.tools,
      toolChoice: undefined,
      modelId: this.config.modelId,
    });

    return {
      tools: result.tools as GoogleLiveTool[] | undefined,
      warnings: result.toolWarnings,
    };
  }

  async createAuthToken(
    options: GoogleLiveCreateAuthTokenOptions = {},
  ): Promise<GoogleLiveAuthToken> {
    const apiKey = this.config.apiKey();
    if (apiKey == null) {
      throw new Error(
        'Google Live auth token creation requires GOOGLE_GENERATIVE_AI_API_KEY.',
      );
    }

    const headers = await resolve(this.config.headers);
    const response = await postJsonToApi({
      url: `${this.config.liveBaseURL}/auth_tokens`,
      headers: {
        ...headers,
        'x-goog-api-key': apiKey,
      },
      body: buildAuthTokenPayload(this.config.modelId, options),
      failedResponseHandler: googleFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        authTokenResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    return response.value;
  }

  async connect(
    options: GoogleLiveConnectOptions = {},
  ): Promise<Experimental_LiveSessionV1> {
    const authToken = options.authToken;
    const apiKey = authToken == null ? this.config.apiKey() : undefined;

    if (authToken == null && apiKey == null) {
      throw new Error(
        'Google Live connection requires either authToken or GOOGLE_GENERATIVE_AI_API_KEY.',
      );
    }

    const wsBaseUrl = toWsBaseUrl(this.config.liveBaseURL);
    const method =
      authToken != null ? 'BidiGenerateContentConstrained' : 'BidiGenerateContent';
    const keyName = authToken != null ? 'access_token' : 'key';
    const credential = encodeURIComponent(authToken ?? apiKey!);
    const url =
      `${wsBaseUrl}/ws/google.ai.generativelanguage.${this.config.liveApiVersion}` +
      `.GenerativeService.${method}?${keyName}=${credential}`;

    const socket = getWebSocketFactory(this.config.createWebSocket)(url);
    const queue = new AsyncMessageQueue<Experimental_LiveModelV1ServerMessage>();
    const session = new GoogleLiveSessionImpl(
      this.provider,
      this.modelId,
      socket,
      queue,
    );

    let resolveReady!: () => void;
    let rejectReady!: (error: unknown) => void;
    const ready = new Promise<void>((resolve, reject) => {
      resolveReady = resolve;
      rejectReady = reject;
    });
    let setupResolved = false;

    if (options.abortSignal != null) {
      if (options.abortSignal.aborted) {
        session.close();
        throw new Error('Google Live connection aborted before opening.');
      }

      options.abortSignal.addEventListener(
        'abort',
        () => {
          session.close();
        },
        { once: true },
      );
    }

    socket.onopen = () => {
      socket.send(
        JSON.stringify(buildSetupPayload(this.config.modelId, options.config)),
      );
    };

    socket.onmessage = async event => {
      try {
        const text =
          typeof event.data === 'string'
            ? event.data
            : event.data instanceof ArrayBuffer
              ? new TextDecoder().decode(event.data)
              : typeof Blob !== 'undefined' && event.data instanceof Blob
                ? await event.data.text()
                : String(event.data);

        const raw = JSON.parse(text);

        if (!setupResolved && raw.setupComplete != null) {
          setupResolved = true;
          resolveReady();
        }

        const normalized = normalizeServerMessage(raw, this.config.generateId);
        for (const message of normalized) {
          queue.push(message);
        }
      } catch (error) {
        if (!setupResolved) {
          rejectReady(error);
        }
        queue.end(error);
      }
    };

    socket.onerror = event => {
      const error =
        event.error ?? new Error('Google Live WebSocket encountered an error.');
      if (!setupResolved) {
        rejectReady(error);
      }
      session.handleClose({ error });
    };

    socket.onclose = event => {
      if (!setupResolved) {
        rejectReady(
          new Error(
            event.reason
              ? `Google Live WebSocket closed before setup completed: ${event.reason}`
              : 'Google Live WebSocket closed before setup completed.',
          ),
        );
      }

      session.handleClose({
        code: event.code,
        reason: event.reason,
      });
    };

    await ready;
    return session;
  }
}
