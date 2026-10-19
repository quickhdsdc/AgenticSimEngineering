<template>
  <span class="help-tip"><button type="button" :aria-label="text" @mouseenter="show" @mouseleave="open=false" @focus="show" @blur="open=false" @click="show" @keydown.esc="open=false" :aria-expanded="open">?</button><Teleport to="body"><span v-if="open" class="help-content" role="tooltip" :style="position">{{ text }}</span></Teleport></span>
</template>
<script setup lang="ts">
import {ref} from 'vue'
defineProps<{text:string}>()
const open = ref(false)
const position = ref({left:'8px', top:'8px'})
function show(event:Event) {
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
  position.value = {left:`${Math.max(8, Math.min(rect.left, window.innerWidth - 310))}px`, top:`${Math.max(8, Math.min(rect.bottom + 6, window.innerHeight - 150))}px`}
  open.value = true
}
</script>
<style scoped>
.help-tip{display:inline-flex;position:relative;margin-left:6px;vertical-align:middle}
.help-tip button{border:1px solid #8395a7;border-radius:50%;background:transparent;color:inherit;width:20px;height:20px;padding:0;font:13px sans-serif;cursor:help}
.help-content{position:fixed;z-index:5000;width:280px;max-width:calc(100vw - 40px);max-height:130px;overflow:auto;padding:12px;background:#203044;color:#edf5ff;border:1px solid #53708c;border-radius:6px;box-shadow:0 4px 14px #0005;font:14px/1.5 sans-serif;white-space:normal;text-transform:none;letter-spacing:normal;pointer-events:none}
</style>
