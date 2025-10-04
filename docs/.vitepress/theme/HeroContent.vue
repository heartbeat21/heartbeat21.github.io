<!-- .vitepress/theme/HeroContent.vue -->
<script setup lang="ts">
import { useData } from 'vitepress'
import DefaultTheme from 'vitepress/theme'

const { frontmatter } = useData()
const hero = frontmatter.value.hero
const bgImage = hero.background // fallback
const name = hero.name
const text = hero.text
const tagline = hero.tagline
</script>

<template>
  <div class="custom-home-hero-info" :style="{ backgroundImage: `url(${bgImage})` }">
    <!-- 这里可以根据需要添加遮罩层 -->
    <div class="overlay"></div>
    
    <!-- 官方 home-hero-info 内容 -->
    <slot name="home-hero-info-before" />
      <div class="hero-text-container">
        <h1 class="heading">
          <span v-if="name" v-html="name" class="name clip"></span>
        </h1>
          <p v-if="tagline" v-html="tagline" class="tagline"></p>
      </div>
    <slot name="home-hero-info-after" />
  </div>
</template>

<style scoped>
/* ============ 核心容器与布局 ============ */

/* 仿照 .VPHero 的基础样式 */
.custom-home-hero-info {
  margin-top: calc((var(--vp-nav-height) + var(--vp-layout-top-height, 0px)) * -1);
  padding: calc(var(--vp-nav-height) + var(--vp-layout-top-height, 0px) + 80px) 24px calc(var(--vp-nav-height) + var(--vp-layout-top-height, 0px) + 80px);
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  position: relative; /* 为遮罩层定位 */
  filter: brightness(1.2);
  min-height: 300px; 
}

.hero-text-container {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  z-index: 2; /* 确保在 .overlay 之上 */
  transform: none !important; /* 强制移除 VitePress 动画位移 */
}

@media (min-width: 640px) {
  .custom-home-hero-info {
    padding: calc(var(--vp-nav-height) + var(--vp-layout-top-height, 0px) + 100px) 48px calc(var(--vp-nav-height) + var(--vp-layout-top-height, 0px) + 100px);
  }
}

@media (min-width: 960px) {
  .custom-home-hero-info {
    align-items: flex-start;
    padding: calc(var(--vp-nav-height) + var(--vp-layout-top-height, 0px) + 120px) 64px calc(var(--vp-nav-height) + var(--vp-layout-top-height, 0px) + 120px); 
  }
}

/* 仿照 .container 的布局 */
.container {
  display: flex;
  flex-direction: column;
  margin: 0 auto;
  max-width: 1152px;
}

@media (min-width: 960px) {
  .container {
    flex-direction: row;
  }
}

/* 仿照 .main 的定位和尺寸 */
.main {
  position: relative;
  z-index: 10;
  order: 2;
  flex-grow: 1;
  flex-shrink: 0;
}

/* 在有背景图的场景下，内容居中 */
.custom-home-hero-info .container {
  text-align: center;
}

@media (min-width: 960px) {
  .custom-home-hero-info .container {
    text-align: left;
  }
}

@media (min-width: 960px) {
  .main {
    order: 1;
    width: calc((100% / 3) * 2);
  }
  /* 限制最大宽度，与官方一致 */
  .custom-home-hero-info .main {
    max-width: 592px;
  }
}

/* ============ 遮罩层 ============ */

.overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(to bottom, rgba(0, 0, 0, 0.2), rgba(255, 255, 255, 0.0)); /* 渐变色遮罩层 */
  z-index: 1;
}

/* ============ 标题与副标题样式 ============ */

/* 仿照 .heading */
.heading {
  display: flex;
  flex-direction: column;
}

/* 仿照 .name 和 .text */
.name,
.text {
  margin: 0 auto;
  width: fit-content;
  max-width: 392px;
  letter-spacing: -0.4px;
  line-height: 40px;
  font-size: 32px;
  font-weight: 700;
  white-space: pre-wrap;
  z-index: 2;
}

/* 在有背景图时居中 */
.custom-home-hero-info .name,
.custom-home-hero-info .text {
  transform: translateY(var(--vp-home-hero-name-text-transform));
}

/* 仿照 .name 的颜色 */
.name {
  color: var(--vp-home-hero-name-color);
}

/* 仿照 .clip 的文字渐变效果 */
.clip {
  background: var(--vp-home-hero-name-background);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* 响应式字体大小 */
@media (min-width: 640px) {
  .name,
  .text {
    max-width: 576px;
    line-height: 56px;
    font-size: 48px;
  }
}

@media (min-width: 960px) {
  .name,
  .text {
    line-height: 64px;
    font-size: 56px;
  }
  /* 在大屏时取消居中 */
  .custom-home-hero-info .name,
  .custom-home-hero-info .text {
    margin: 0;
  }
}

/* 仿照 .tagline */
.tagline {
  margin: 0 auto; 
  padding-top: 8px;
  max-width: 392px;
  line-height: 28px;
  font-size: 18px;
  font-weight: 500;
  white-space: pre-wrap;
  color: var(--vp-c-text-2);
}

/* 在有背景图时居中 */
.custom-home-hero-info .tagline {
  transform: translateY(var(--vp-home-hero-name-text-transform));
}

@media (min-width: 640px) {
  .tagline {
    padding-top: 12px;
    max-width: 576px;
    line-height: 32px;
    font-size: 20px;
  }
}

@media (min-width: 960px) {
  .tagline {
    line-height: 36px;
    font-size: 24px;
  }
  /* 在大屏时取消居中 */
  .custom-home-hero-info .tagline {
    margin: 0;
  }
}
</style>