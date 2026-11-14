import { ApiError } from '../api/client'

export const GENERIC_ERROR = 'Something went wrong. Try again.'

export const BACKEND_UNREACHABLE =
  "Can't reach the BallTrace backend. Make sure it is running, then reload this page."

/** Where the failure happened; only used to tag the log line, never shown. */
export type ErrorContext =
  | 'load analyses'
  | 'load analysis'
  | 'create analysis'
  | 'upload video'
  | 'start analysis'
  | 'cancel analysis'
  | 'export video'
  | 'load settings'
  | 'save settings'
  | 'progress stream'

export function uploadTooLargeMessage(limitMb: number): string {
  return `That video is larger than the ${limitMb} MB limit. Trim or compress it, then upload it again.`
}

/** Plain-language text for an HTTP status when the server gave no worded message. */
function messageForStatus(status: number): string {
  if (status === 0) return BACKEND_UNREACHABLE
  if (status === 413) {
    return 'That video is too large to upload. Trim or compress it, then upload it again.'
  }
  if (status === 404) return "We couldn't find that. Refresh the page and try again."
  if (status === 400 || status === 422) {
    return "Some of the information sent wasn't valid. Check it and try again."
  }
  if (status >= 500) return 'BallTrace had a problem handling that. Try again in a moment.'
  return GENERIC_ERROR
}

/**
 * The only way a failure becomes text on screen.
 *
 * Known backend failures carry a worded `detail` and a `code`, so that text is
 * shown. Everything else (no JSON body, proxy errors, thrown exceptions) maps
 * to a generic sentence. The raw error goes to the console for debugging and
 * is never part of the returned string.
 */
export function friendlyError(err: unknown, context: ErrorContext): string {
  console.error(`[BallTrace] could not ${context}:`, err)
  if (err instanceof ApiError) {
    if (err.code && err.detail) return err.detail
    return messageForStatus(err.status)
  }
  return GENERIC_ERROR
}
