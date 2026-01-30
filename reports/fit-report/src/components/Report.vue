<template>
  <div id="report">
    <h1>Parameter optimization report</h1>
  </div>
</template>

<script>

export default {
  components: {
  },

  data() {
    return {
      json_data: {
      },
    }
  },

  props: {
    modelSID: {
      type: String,
      default: "Model-SID"
    },
    modelName: {
      type: String,
      default: "Model-Name"
    },
  },

  methods: {}
}
</script>

<style lang="scss" scoped>
@import '../assets/styles/scss/report.scss';
</style>
