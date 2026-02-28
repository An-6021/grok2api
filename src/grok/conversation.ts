import type { GrokSettings } from "../settings";
import { getDynamicHeaders } from "./headers";
import { getModelInfo, toGrokModel } from "./models";

export interface OpenAIChatMessage {
  role: string;
  content: unknown;
}

export interface OpenAIChatRequestBody {
  model: string;
  messages: OpenAIChatMessage[];
  stream?: boolean;
  customPersonality?: unknown;
  systemPrompt?: unknown;
  video_config?: {
    aspect_ratio?: string;
    video_length?: number;
    resolution?: string;
    preset?: string;
  };
}

export const CONVERSATION_API = "https://grok.com/rest/app-chat/conversations/new";

const INVALID_TEXT = new Set(["", "undefined", "[undefined]", "null", "none"]);

function normalizeText(value: unknown): string {
  if (typeof value !== "string") return "";
  const text = value.trim();
  if (!text) return "";
  if (INVALID_TEXT.has(text.toLowerCase())) return "";
  return text;
}

function coerceText(value: unknown, depth = 0): string {
  if (depth > 4 || value === null || value === undefined) return "";

  const normalized = normalizeText(value);
  if (normalized) return normalized;

  if (Array.isArray(value)) {
    const parts: string[] = [];
    for (const item of value) {
      const text = coerceText(item, depth + 1);
      if (text) parts.push(text);
    }
    return parts.join("\n").trim();
  }

  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    if (obj.enabled === false) return "";

    const itemType = normalizeText(obj.type).toLowerCase();
    if (itemType === "text" || itemType === "input_text") {
      const text = coerceText(obj.text, depth + 1);
      if (text) return text;
    }

    for (const key of [
      "text",
      "content",
      "prompt",
      "message",
      "instruction",
      "instructions",
      "system",
      "system_prompt",
      "systemPrompt",
      "value",
    ]) {
      if (!(key in obj)) continue;
      const text = coerceText(obj[key], depth + 1);
      if (text) return text;
    }
  }

  return "";
}

export function extractPersonality(messages: OpenAIChatMessage[]): string {
  const parts: string[] = [];
  for (const msg of messages || []) {
    const role = normalizeText((msg as any)?.role ?? "").toLowerCase();
    if (role !== "system" && role !== "developer") continue;
    const text = coerceText((msg as any)?.content);
    if (text) parts.push(text);
  }
  return parts.join("\n\n").trim();
}

function resolvePersonalityFromBody(body: Record<string, unknown>): string {
  const direct = coerceText(body.customPersonality ?? body.custom_personality);
  if (direct) return direct;

  for (const key of [
    "system",
    "system_prompt",
    "systemPrompt",
    "system_message",
    "systemMessage",
    "instructions",
    "custom_instructions",
    "customInstructions",
    "persona",
    "personality",
  ]) {
    if (!(key in body)) continue;
    const text = coerceText(body[key]);
    if (text) return text;
  }

  for (const [key, value] of Object.entries(body)) {
    const keyL = String(key ?? "").toLowerCase();
    if (keyL === "system_fingerprint") continue;
    if (!["system", "instruction", "persona", "personality"].some((t) => keyL.includes(t)))
      continue;
    const text = coerceText(value);
    if (text) return text;
  }

  return "";
}

export function resolveCustomPersonality(args: {
  body: Record<string, unknown>;
  messages: OpenAIChatMessage[];
  defaultPersonality?: string;
}): string | null {
  const personalityFromField = resolvePersonalityFromBody(args.body);
  const personalityFromMessages = extractPersonality(args.messages);

  const parts: string[] = [];
  if (personalityFromField) parts.push(personalityFromField);
  if (personalityFromMessages && !parts.includes(personalityFromMessages)) parts.push(personalityFromMessages);

  const resolved = parts.join("\n\n").trim();
  if (resolved) return resolved;

  const fallback = normalizeText(args.defaultPersonality);
  return fallback || null;
}

export function extractContent(
  messages: OpenAIChatMessage[],
  opts?: { excludeRoles?: Set<string> },
): { content: string; images: string[] } {
  const images: string[] = [];
  const extracted: Array<{ role: string; text: string }> = [];
  const excludeRoles = opts?.excludeRoles;

  for (const msg of messages) {
    const role = normalizeText((msg as any)?.role ?? "user").toLowerCase() || "user";
    if (excludeRoles?.has(role)) continue;
    const content = (msg as any)?.content ?? "";

    const parts: string[] = [];
    const blocks = Array.isArray(content) ? content : [content];
    for (const item of blocks) {
      if (!item) continue;
      if (typeof item === "string") {
        const t = normalizeText(item);
        if (t) parts.push(t);
        continue;
      }
      if (typeof item !== "object") continue;
      const obj = item as Record<string, unknown>;
      const itemType = normalizeText(obj.type).toLowerCase();
      if (itemType === "text" || itemType === "input_text") {
        const t = normalizeText(obj.text);
        if (t) parts.push(t);
      } else if (itemType === "image_url") {
        const imageUrl = obj.image_url;
        if (imageUrl && typeof imageUrl === "object") {
          const url = normalizeText((imageUrl as Record<string, unknown>).url);
          if (url) images.push(url);
        }
      }
    }

    if (parts.length) extracted.push({ role, text: parts.join("\n") });
  }

  let lastUserIndex: number | null = null;
  for (let i = extracted.length - 1; i >= 0; i--) {
    if (extracted[i]!.role === "user") {
      lastUserIndex = i;
      break;
    }
  }

  const out: string[] = [];
  for (let i = 0; i < extracted.length; i++) {
    const role = extracted[i]!.role || "user";
    const text = extracted[i]!.text;
    if (i === lastUserIndex) out.push(text);
    else out.push(`${role}: ${text}`);
  }

  return { content: out.join("\n\n"), images };
}

export function buildConversationPayload(args: {
  requestModel: string;
  content: string;
  imgIds: string[];
  imgUris: string[];
  disableSearch?: boolean;
  postId?: string;
  videoConfig?: {
    aspect_ratio?: string;
    video_length?: number;
    resolution?: string;
    preset?: string;
  };
  customPersonality?: string | null;
  settings: GrokSettings;
}): { payload: Record<string, unknown>; referer?: string; isVideoModel: boolean } {
  const { requestModel, content, imgIds, imgUris, postId, settings } = args;
  const cfg = getModelInfo(requestModel);
  const { grokModel, mode, isVideoModel } = toGrokModel(requestModel);

  if (cfg?.is_video_model) {
    if (!postId) throw new Error("视频模型缺少 postId（需要先创建 media post）");

    const aspectRatio = (args.videoConfig?.aspect_ratio ?? "").trim() || "3:2";
    const videoLengthRaw = Number(args.videoConfig?.video_length ?? 6);
    const videoLength = Number.isFinite(videoLengthRaw) ? Math.max(1, Math.floor(videoLengthRaw)) : 6;
    const resolution = (args.videoConfig?.resolution ?? "SD") === "HD" ? "HD" : "SD";
    const preset = (args.videoConfig?.preset ?? "normal").trim();

    let modeFlag = "--mode=custom";
    if (preset === "fun") modeFlag = "--mode=extremely-crazy";
    else if (preset === "normal") modeFlag = "--mode=normal";
    else if (preset === "spicy") modeFlag = "--mode=extremely-spicy-or-crazy";

    const prompt = `${String(content || "").trim()} ${modeFlag}`.trim();

    return {
      isVideoModel: true,
      referer: "https://grok.com/imagine",
      payload: {
        temporary: true,
        modelName: "grok-3",
        message: prompt,
        toolOverrides: { videoGen: true },
        enableSideBySide: true,
        responseMetadata: {
          experiments: [],
          modelConfigOverride: {
            modelMap: {
              videoGenModelConfig: {
                parentPostId: postId,
                aspectRatio,
                videoLength,
                videoResolution: resolution,
              },
            },
          },
        },
      },
    };
  }

  return {
    isVideoModel,
    payload: {
      temporary: settings.temporary ?? true,
      modelName: grokModel,
      message: content,
      fileAttachments: imgIds,
      imageAttachments: [],
      disableSearch: args.disableSearch === true,
      enableImageGeneration: true,
      returnImageBytes: false,
      returnRawGrokInXaiRequest: false,
      enableImageStreaming: true,
      imageGenerationCount: 2,
      forceConcise: false,
      ...(normalizeText(args.customPersonality) ? { customPersonality: normalizeText(args.customPersonality) } : {}),
      toolOverrides: {},
      enableSideBySide: true,
      sendFinalMetadata: true,
      isReasoning: false,
      webpageUrls: [],
      disableTextFollowUps: true,
      responseMetadata: { requestModelDetails: { modelId: grokModel } },
      disableMemory: false,
      forceSideBySide: false,
      modelMode: mode,
      isAsyncChat: false,
    },
  };
}

export async function sendConversationRequest(args: {
  payload: Record<string, unknown>;
  cookie: string;
  settings: GrokSettings;
  referer?: string;
}): Promise<Response> {
  const { payload, cookie, settings, referer } = args;
  const headers = getDynamicHeaders(settings, "/rest/app-chat/conversations/new");
  headers.Cookie = cookie;
  if (referer) headers.Referer = referer;
  const body = JSON.stringify(payload);

  return fetch(CONVERSATION_API, { method: "POST", headers, body });
}
