/* Optional integration; demo/ is maintained separately. Every stream can pause for approval. */
window.demoAgentStream = async function (path, body, onEvent = () => {}) {
  const response = await fetch(`http://127.0.0.1:8770${path}`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  if (!response.ok) throw new Error((await response.json()).error || 'Agent request failed');
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let pending = ''; const events = [];
  const emit = line => {
    if (!line.trim()) return;
    const event = JSON.parse(line); events.push(event); onEvent(event);
    window.dispatchEvent(new CustomEvent('demoAgentEvent', {detail: event}));
  };
  for (;;) {
    const {value, done} = await reader.read();
    pending += decoder.decode(value, {stream: !done});
    const lines = pending.split('\n'); pending = lines.pop(); lines.forEach(emit);
    if (done) { emit(pending); break; }
  }
  return events;
};
window.demoAgent = (question, onEvent, profile_id = 'default') =>
  window.demoAgentStream('/ask', {question, profile_id}, onEvent);
window.demoAgentApprove = (thread, approval_id, decision, edit = {}, onEvent) =>
  window.demoAgentStream('/approve', {thread, approval_id, decision, ...edit}, onEvent);
window.demoAgentFeedback = (needs, feedback, profile_id = 'default', onEvent) =>
  window.demoAgentStream('/feedback', {needs, feedback, profile_id}, onEvent);
