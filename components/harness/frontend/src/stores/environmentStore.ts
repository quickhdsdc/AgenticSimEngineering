import { defineStore } from 'pinia';
import { computed, ref } from 'vue';

const isProduction = import.meta.env.MODE === 'production';

export const useEnvStore = defineStore('envStore', () => {
    // States
    const _appVersion = ref(import.meta.env.VITE_APP_VERSION || (isProduction ? '/__ENV_APP_VERSION__PLACEHOLDER__/' : ''));

    const _backendUrl = ref(import.meta.env.VITE_APP_BACKEND_URL || (isProduction ? '/__ENV_BACKEND_URL__PLACEHOLDER__/' : ''));
    const _aasServerUrl = ref(import.meta.env.VITE_APP_AAS_SERVER_URL || (isProduction ? '/__ENV_AAS_SERVER_URL__PLACEHOLDER__/' : ''));
    const _aasUiUrl = ref(import.meta.env.VITE_APP_AAS_UI_URL || (isProduction ? '/__ENV_AAS_UI_URL__PLACEHOLDER__/' : ''));

    const mesUiUrl = ref(import.meta.env.VITE_APP_MES_UI_URL || '')
    // Getters
    const appVersion = computed(() => _appVersion.value);

    const backendUrl = computed(() => _backendUrl.value);
    const aasServerUrl = computed(() => _aasServerUrl.value);
    const aasUiUrl = computed(() => _aasUiUrl.value);

    return {
        mesUiUrl,
        appVersion,
        backendUrl,
        aasServerUrl,
        aasUiUrl,
    };
});