import { createApp } from 'vue'
import App from './App.vue'
import HelpTip from './components/HelpTip.vue'
import './style.css'
import {createPinia} from 'pinia';
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate';

const app = createApp(App);
app.component('HelpTip', HelpTip);

const pinia = createPinia();
pinia.use(piniaPluginPersistedstate);
app.use(pinia);

app.mount('#app')
