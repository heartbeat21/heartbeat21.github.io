<!-- .vitepress/theme/MyLayout.vue -->
<script setup>
import DefaultTheme from 'vitepress/theme'
import HeroContent from './HeroContent.vue'

const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <!-- 覆盖首页英雄区域的默认内容 -->
    <template #home-hero-info>
      <HeroContent />
    </template>
  </Layout>
</template>