<template>
  <div id="app">
   <Report />
  </div>
</template>

<script>
import Report from "./components/Report";

export default {
  name: 'app',
  components: {
    Report
  },
  data () {
    return {}
  },
  methods: {}
}
</script>

<style lang="scss">
@import './assets/styles/scss/app.scss';
@import'~bootstrap/dist/css/bootstrap.css';
</style>
