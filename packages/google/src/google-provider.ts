import {
  EmbeddingModelV4,
  Experimental_LiveModelV1,
  Experimental_VideoModelV4,
  ImageModelV4,
  LanguageModelV4,
  ProviderV4,
} from '@ai-sdk/provider';
import {
  FetchFunction,
  generateId,
  loadApiKey,
  loadOptionalSetting,
  withoutTrailingSlash,
  withUserAgentSuffix,
} from '@ai-sdk/provider-utils';
import { VERSION } from './version';
import { GoogleGenerativeAIEmbeddingModel } from './google-generative-ai-embedding-model';
import { GoogleGenerativeAIEmbeddingModelId } from './google-generative-ai-embedding-options';
import { GoogleGenerativeAILanguageModel } from './google-generative-ai-language-model';
import { GoogleGenerativeAIModelId } from './google-generative-ai-options';
import { googleTools } from './google-tools';

import {
  GoogleGenerativeAIImageSettings,
  GoogleGenerativeAIImageModelId,
} from './google-generative-ai-image-settings';
import { GoogleGenerativeAIImageModel } from './google-generative-ai-image-model';
import { GoogleGenerativeAILiveModel } from './google-live-model';
import { GoogleLiveModel, GoogleLiveWebSocketFactory } from './google-live-types';
import { GoogleGenerativeAIVideoModel } from './google-generative-ai-video-model';
import { GoogleGenerativeAIVideoModelId } from './google-generative-ai-video-settings';

export interface GoogleGenerativeAIProvider extends ProviderV4 {
  (modelId: GoogleGenerativeAIModelId): LanguageModelV4;

  languageModel(modelId: GoogleGenerativeAIModelId): LanguageModelV4;

  chat(modelId: GoogleGenerativeAIModelId): LanguageModelV4;

  /**
   * Creates a model for image generation.
   */
  image(
    modelId: GoogleGenerativeAIImageModelId,
    settings?: GoogleGenerativeAIImageSettings,
  ): ImageModelV4;

  /**
   * @deprecated Use `chat()` instead.
   */
  generativeAI(modelId: GoogleGenerativeAIModelId): LanguageModelV4;

  /**
   * Creates a model for text embeddings.
   */
  embedding(modelId: GoogleGenerativeAIEmbeddingModelId): EmbeddingModelV4;

  /**
   * Creates a model for text embeddings.
   */
  embeddingModel(modelId: GoogleGenerativeAIEmbeddingModelId): EmbeddingModelV4;

  /**
   * @deprecated Use `embedding` instead.
   */
  textEmbedding(modelId: GoogleGenerativeAIEmbeddingModelId): EmbeddingModelV4;

  /**
   * @deprecated Use `embeddingModel` instead.
   */
  textEmbeddingModel(
    modelId: GoogleGenerativeAIEmbeddingModelId,
  ): EmbeddingModelV4;

  /**
   * Creates a model for video generation.
   */
  video(modelId: GoogleGenerativeAIVideoModelId): Experimental_VideoModelV4;

  /**
   * Creates a model for video generation.
   */
  videoModel(
    modelId: GoogleGenerativeAIVideoModelId,
  ): Experimental_VideoModelV4;

  /**
   * Creates a model for Gemini Live sessions.
   */
  live(modelId: GoogleGenerativeAIModelId): GoogleLiveModel;

  /**
   * Creates a model for Gemini Live sessions.
   */
  liveModel(modelId: GoogleGenerativeAIModelId): GoogleLiveModel;

  tools: typeof googleTools;
}

export interface GoogleGenerativeAIProviderSettings {
  /**
   * Use a different URL prefix for API calls, e.g. to use proxy servers.
   * The default prefix is `https://generativelanguage.googleapis.com/v1beta`.
   */
  baseURL?: string;

  /**
   * API key that is being send using the `x-goog-api-key` header.
   * It defaults to the `GOOGLE_GENERATIVE_AI_API_KEY` environment variable.
   */
  apiKey?: string;

  /**
   * Custom headers to include in the requests.
   */
  headers?: Record<string, string | undefined>;

  /**
   * Custom fetch implementation. You can use it as a middleware to intercept requests,
   * or to provide a custom fetch implementation for e.g. testing.
   */
  fetch?: FetchFunction;

  /**
   * Optional base URL for Gemini Live token creation and WebSocket connections.
   * Defaults to `https://generativelanguage.googleapis.com`.
   */
  liveBaseURL?: string;

  /**
   * Live API version. Defaults to `v1alpha`.
   */
  liveApiVersion?: string;

  /**
   * Optional WebSocket factory used for Gemini Live connections.
   */
  createWebSocket?: GoogleLiveWebSocketFactory;

  /**
   * Optional function to generate a unique ID for each request.
   */
  generateId?: () => string;

  /**
   * Custom provider name
   * Defaults to 'google.generative-ai'.
   */
  name?: string;
}

/**
 * Create a Google Generative AI provider instance.
 */
export function createGoogleGenerativeAI(
  options: GoogleGenerativeAIProviderSettings = {},
): GoogleGenerativeAIProvider {
  const baseURL =
    withoutTrailingSlash(options.baseURL) ??
    'https://generativelanguage.googleapis.com/v1beta';
  const liveBaseURL =
    withoutTrailingSlash(options.liveBaseURL) ??
    'https://generativelanguage.googleapis.com';

  const providerName = options.name ?? 'google.generative-ai';
  const getOptionalApiKey = () =>
    loadOptionalSetting({
      settingValue: options.apiKey,
      environmentVariableName: 'GOOGLE_GENERATIVE_AI_API_KEY',
    });

  const getHeaders = () =>
    withUserAgentSuffix(
      {
        'x-goog-api-key': loadApiKey({
          apiKey: options.apiKey,
          environmentVariableName: 'GOOGLE_GENERATIVE_AI_API_KEY',
          description: 'Google Generative AI',
        }),
        ...options.headers,
      },
      `ai-sdk/google/${VERSION}`,
    );

  const getLiveHeaders = () =>
    withUserAgentSuffix(
      {
        ...options.headers,
      },
      `ai-sdk/google/${VERSION}`,
    );

  const createChatModel = (modelId: GoogleGenerativeAIModelId) =>
    new GoogleGenerativeAILanguageModel(modelId, {
      provider: providerName,
      baseURL,
      headers: getHeaders,
      generateId: options.generateId ?? generateId,
      supportedUrls: () => ({
        '*': [
          // Google Generative Language "files" endpoint
          // e.g. https://generativelanguage.googleapis.com/v1beta/files/...
          new RegExp(`^${baseURL}/files/.*$`),
          // YouTube URLs (public or unlisted videos)
          new RegExp(
            `^https://(?:www\\.)?youtube\\.com/watch\\?v=[\\w-]+(?:&[\\w=&.-]*)?$`,
          ),
          new RegExp(`^https://youtu\\.be/[\\w-]+(?:\\?[\\w=&.-]*)?$`),
        ],
      }),
      fetch: options.fetch,
    });

  const createEmbeddingModel = (modelId: GoogleGenerativeAIEmbeddingModelId) =>
    new GoogleGenerativeAIEmbeddingModel(modelId, {
      provider: providerName,
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const createImageModel = (
    modelId: GoogleGenerativeAIImageModelId,
    settings: GoogleGenerativeAIImageSettings = {},
  ) =>
    new GoogleGenerativeAIImageModel(modelId, settings, {
      provider: providerName,
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const createVideoModel = (modelId: GoogleGenerativeAIVideoModelId) =>
    new GoogleGenerativeAIVideoModel(modelId, {
      provider: providerName,
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
      generateId: options.generateId ?? generateId,
    });

  const createLiveModel = (modelId: GoogleGenerativeAIModelId) =>
    new GoogleGenerativeAILiveModel({
      provider: `${providerName}.live`,
      modelId,
      liveBaseURL,
      liveApiVersion: options.liveApiVersion ?? 'v1alpha',
      headers: getLiveHeaders,
      apiKey: getOptionalApiKey,
      fetch: options.fetch,
      createWebSocket: options.createWebSocket,
      generateId: options.generateId ?? generateId,
    });

  const provider = function (modelId: GoogleGenerativeAIModelId) {
    if (new.target) {
      throw new Error(
        'The Google Generative AI model function cannot be called with the new keyword.',
      );
    }

    return createChatModel(modelId);
  };

  provider.specificationVersion = 'v4' as const;
  provider.languageModel = createChatModel;
  provider.chat = createChatModel;
  provider.generativeAI = createChatModel;
  provider.embedding = createEmbeddingModel;
  provider.embeddingModel = createEmbeddingModel;
  provider.textEmbedding = createEmbeddingModel;
  provider.textEmbeddingModel = createEmbeddingModel;
  provider.image = createImageModel;
  provider.imageModel = createImageModel;
  provider.live = createLiveModel;
  provider.liveModel = createLiveModel;
  provider.video = createVideoModel;
  provider.videoModel = createVideoModel;
  provider.tools = googleTools;

  return provider as GoogleGenerativeAIProvider;
}

/**
 * Default Google Generative AI provider instance.
 */
export const google = createGoogleGenerativeAI();
