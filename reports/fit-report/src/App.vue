<template>
  <div id="app">
   <Report />
    {{ myJson }}
  </div>
</template>

<script>
import Report from "./components/Report";
import json from './01_optimization_result.json'

export default {
  name: 'app',
  components: {
    Report
  },
  data () {
    return {
      myJson: json
    }
  },
  methods: {}
}
</script>

<style lang="scss">
@import './assets/styles/scss/app.scss';
@import'~bootstrap/dist/css/bootstrap.css';
</style>
