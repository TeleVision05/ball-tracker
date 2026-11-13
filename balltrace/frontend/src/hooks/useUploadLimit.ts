import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { uploadTooLargeMessage } from '../lib/friendlyError'

/**
 * The backend's upload size limit in MB, or null until known (or if it can't be
 * read). The backend enforces the limit either way; this lets the page refuse
 * a too-large file before sending it.
 */
export function useUploadLimit(): number | null {
  const [limitMb, setLimitMb] = useState<number | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .limits()
      .then((limits) => {
        if (!cancelled) setLimitMb(limits.max_upload_mb)
      })
      .catch(() => {
        // Not fatal: the backend still rejects an oversized file with a clear message.
      })
    return () => {
      cancelled = true
    }
  }, [])

  return limitMb
}

/** Message to show if `file` is over the limit, otherwise null. */
export function checkUploadSize(file: File, limitMb: number | null): string | null {
  if (limitMb == null) return null
  return file.size > limitMb * 1024 * 1024 ? uploadTooLargeMessage(limitMb) : null
}
