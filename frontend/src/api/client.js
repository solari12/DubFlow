async function request(url, options) {
  const response = await fetch(url, options)
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try { message = (await response.json()).detail || message } catch { /* response was not JSON */ }
    throw new Error(message)
  }
  return response.json()
}

export const healthCheck = () => request('/api/v1/health')

export function createJob(file, options) {
  const body = new FormData()
  body.append('video', file)
  body.append('source_language', options.sourceLanguage)
  body.append('target_language', options.targetLanguage)
  body.append('voice', options.voice)
  return request('/api/v1/jobs', { method: 'POST', body })
}

export const getJob = (jobId) => request(`/api/v1/jobs/${encodeURIComponent(jobId)}`)
export const getArtifactUrl = (jobId) => `/api/v1/artifacts/${encodeURIComponent(jobId)}/video`
