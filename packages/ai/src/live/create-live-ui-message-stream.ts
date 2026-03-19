import { UIMessageChunk } from '../ui-message-stream/ui-message-chunks';
import { createUIMessageStream } from '../ui-message-stream/create-ui-message-stream';
import { UIMessage } from '../ui/ui-messages';
import {
  ObservedLiveSession,
  ObservedLiveSessionMessage,
} from './observe-live-session';

function pushTranscriptChunk({
  writer,
  textBlockId,
  previousText,
  nextText,
}: {
  writer: { write: (chunk: any) => void };
  textBlockId: string;
  previousText: string;
  nextText: string;
}) {
  if (previousText.length === 0) {
    writer.write({ type: 'text-start', id: textBlockId });
  }

  const delta = nextText.startsWith(previousText)
    ? nextText.slice(previousText.length)
    : nextText;

  if (delta.length > 0) {
    writer.write({
      type: 'text-delta',
      id: textBlockId,
      delta,
    });
  }
}

export function experimental_createLiveUIMessageStream<
  UI_MESSAGE extends UIMessage = UIMessage,
>({
  session,
  originalMessages,
}: {
  session: ObservedLiveSession<any>;
  originalMessages?: UI_MESSAGE[];
}): ReadableStream<any> {
  return createUIMessageStream<UI_MESSAGE>({
    originalMessages,
    execute: async ({ writer }) => {
      const textBlockId = `live-text-${session.callId}`;
      let assistantTranscript = '';

      for await (const message of session.observedMessages) {
        switch (message.type) {
          case 'transcript':
            if (message.role === 'assistant') {
              pushTranscriptChunk({
                writer,
                textBlockId,
                previousText: assistantTranscript,
                nextText: message.text,
              });
              assistantTranscript = message.text;
            }
            break;

          case 'tool-call':
            for (const toolCall of message.toolCalls) {
              writer.write({
                type: 'tool-input-available',
                toolCallId: toolCall.toolCallId,
                toolName: toolCall.toolName,
                input: toolCall.input,
                dynamic: true,
              });
            }
            break;

          case 'tool-call-cancel':
            // UI protocol has no direct cancel chunk. Let the consumer handle this
            // via observedMessages if it needs stricter cancellation handling.
            break;

          case 'tool-output-available':
            writer.write({
              type: 'tool-output-available',
              toolCallId: message.toolCall.toolCallId,
              output: message.output,
              dynamic: message.toolCall.dynamic === true,
              preliminary: message.preliminary,
            });
            break;

          case 'tool-output-error':
            writer.write({
              type: 'tool-output-error',
              toolCallId: message.toolCall.toolCallId,
              errorText: message.errorText,
              dynamic: message.toolCall.dynamic === true,
            });
            break;

          case 'turn-complete':
            if (assistantTranscript.length > 0) {
              writer.write({ type: 'text-end', id: textBlockId });
            }
            assistantTranscript = '';
            break;
        }
      }
    },
  });
}
