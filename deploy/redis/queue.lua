-- Five explicit keys, bounded 128 jobs/workers, Redis server time, no external side effects.
-- All access to this namespace must use this protocol, standalone Redis 7+ only.
local jobs, pending, leases, meta, workers = KEYS[1], KEYS[2], KEYS[3], KEYS[4], KEYS[5]
local expected = {'hash', 'list', 'zset', 'hash', 'zset'}
for i=1,5 do
  local kind=redis.call('TYPE',KEYS[i]).ok
  if kind~='none' and kind~=expected[i] then return redis.error_reply('QUEUE_WRONG_TYPE') end
end
local c=cjson.decode(ARGV[2])
local op=c.op
local context=redis.call('HGET',meta,'context')
if context and context~=ARGV[1] then return redis.error_reply('QUEUE_CONTEXT_MISMATCH') end
if not context and redis.call('EXISTS',jobs,pending,leases,meta,workers)>0 then
  return redis.error_reply('QUEUE_MISSING_CONTEXT')
end
if not context and op~='submit' and op~='worker_ping' and op~='workers' then return 'null' end
local function output(j) return j and cjson.encode(j) or 'null' end
local function get(id)
  local raw=redis.call('HGET',jobs,id)
  return raw and cjson.decode(raw) or nil
end
local function save(j) redis.call('HSET',jobs,j.id,cjson.encode(j)) end
local clock=redis.call('TIME')
local now=tonumber(clock[1])*1000+math.floor(tonumber(clock[2])/1000)
if op=='worker_ping' or op=='workers' then
  -- Observational liveness only, independent of job leases and model readiness.
  redis.call('ZREMRANGEBYSCORE',workers,'-inf',now)
  if op=='worker_ping' then
    if not redis.call('ZSCORE',workers,c.owner) and redis.call('ZCARD',workers)>=128 then
      return redis.error_reply('QUEUE_WORKERS_FULL')
    end
    if not context then redis.call('HSET',meta,'context',ARGV[1]) end
    redis.call('ZADD',workers,now+5000,c.owner)
    redis.call('PEXPIRE',workers,5000)
  end
  return cjson.encode({worker_count=redis.call('ZCARD',workers),heartbeat_ttl_ms=5000})
elseif op=='submit' then
  local old=get(c.id)
  local payload=ARGV[3]
  if old then
    -- Canonical payload bytes are supplied separately by the C++ client.
    if old.request_json~=payload then return redis.error_reply('QUEUE_ID_CONFLICT') end
    return output(old)
  end
  if redis.call('HLEN',jobs)>=128 then return redis.error_reply('QUEUE_FULL') end
  if not context then redis.call('HSET',meta,'context',ARGV[1]) end
  local j={id=c.id,request_json=payload,state='queued',attempts=0,
    token='0',owner='',deadline_ms=0,result=cjson.null,updated_ms=now}
  save(j); redis.call('RPUSH',pending,c.id)
  return output(j)
elseif op=='get' then return output(get(c.id))
elseif op=='claim' then
  -- No scan of unbounded history; at most 128 current leases and 128 jobs.
  local expired=redis.call('ZRANGEBYSCORE',leases,'-inf',now,'LIMIT',0,128)
  for _,id in ipairs(expired) do
    local j=get(id)
    if j and j.state=='running' and j.deadline_ms<=now then
      j.owner=''; j.deadline_ms=0; j.updated_ms=now
      if j.attempts>=3 then j.state='failed'; j.result={error='lease retry limit exceeded'}
      else j.state='queued'; redis.call('RPUSH',pending,id) end
      save(j)
    end
    redis.call('ZREM',leases,id)
  end
  local id=redis.call('LINDEX',pending,0)
  if not id then return 'null' end
  local j=get(id)
  if not j or j.state~='queued' then return redis.error_reply('QUEUE_CORRUPT_PENDING') end
  local seq=tonumber(redis.call('HGET',meta,'sequence') or '0')
  if seq>=999999999999 then return redis.error_reply('QUEUE_TOKEN_LIMIT') end
  local token=redis.call('HINCRBY',meta,'sequence',1)
  j.state='running'; j.owner=c.owner; j.token=string.format('%.0f',token)
  j.attempts=j.attempts+1; j.deadline_ms=now+c.lease_ms; j.updated_ms=now
  save(j); redis.call('LPOP',pending); redis.call('ZADD',leases,j.deadline_ms,id)
  return output(j)
elseif op=='cancel' then
  local j=get(c.id)
  if not j then return 'null' end
  if j.state=='queued' or j.state=='running' then
    j.state='cancelled'; j.deadline_ms=0; j.updated_ms=now
    save(j); redis.call('LREM',pending,0,c.id); redis.call('ZREM',leases,c.id)
  end
  return output(j)
elseif op=='renew' or op=='complete' or op=='fail' then
  local j=get(c.id)
  if not j or j.state~='running' or j.owner~=c.owner or j.token~=c.token or j.deadline_ms<=now then
    return 'null'
  end
  j.updated_ms=now
  if op=='renew' then
    j.deadline_ms=now+c.lease_ms; save(j); redis.call('ZADD',leases,j.deadline_ms,c.id)
  else
    j.state=op=='complete' and 'succeeded' or 'failed'; j.result_json=ARGV[4]; j.deadline_ms=0
    save(j); redis.call('ZREM',leases,c.id)
  end
  return output(j)
end
return redis.error_reply('QUEUE_UNKNOWN_OPERATION')
