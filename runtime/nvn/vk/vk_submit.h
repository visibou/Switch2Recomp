// Submissions: one NVN QueueSubmitCommands = one recorded submission (an upload command buffer, then the main one), batched into few vkQueueSubmit2
// calls; CPU-visible effects of GPU completion (events, counters, readbacks) are applied in order by the completion thread, after the GPU is done.
#pragma once
#include "vk_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

int vks_init(void);
void vks_shutdown(void);
void vks_begin(void);                         // gfx_sub_begin (the caller holds the NVN submit lock)
VkCommandBuffer vks_cb(void);                 // the main command buffer of the submission being recorded (the replay's)
VkCommandBuffer vks_upload_cb(void);          // its upload command buffer: runs first, with a transfer -> everything barrier at its end
uint64_t vks_take_seq(void);                   // slots taken for recording so far (vk_retire)
uint64_t vks_oldest_recording(uint64_t *assigned);   // the take number of the oldest slot still recording (UINT64_MAX: none), and the value assigned so far
uint64_t vks_recording_value(void);           // the timeline value the submission being recorded will signal (0 = none)
void vks_post_event(volatile uint32_t *val, uint32_t mode, uint32_t v);   // written after completion: mode 0 write, 1 add
void vks_post_counter(uint64_t addr, uint32_t type);                       // the 16-byte counter record written after completion
void vks_post_fn(void (*fn)(void *), void *arg);                           // run after completion, before waiters return
void vks_split(void);
uint64_t vks_commit(void);                    // gfx_sub_commit: ends the submission, returns its fence value
void vks_flush(void);                         // hands every committed submission to the queue (one vkQueueSubmit2)
int vks_wait(uint64_t fence, uint64_t timeout_ns);   // 1 when the GPU has completed fence and its CPU effects are applied (0 timeout_ns = poll)
void vks_finish(void);                        // flush, then wait for all of it
int vks_owner(void);                          // the slot this thread is recording into, -1 = none (staging ownership)
int vks_recording(void);                      // an NVN submission is being recorded (its value is not assigned yet)
void vks_stats(uint64_t *subs, uint64_t *queue_submits, uint64_t *empty, uint64_t *posted);

#ifdef __cplusplus
}
#endif
