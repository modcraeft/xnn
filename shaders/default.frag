#version 330 core
in vec4 fragColor;
in vec2 texCoord;

out vec4 FragColor;

uniform sampler2D tex0;
uniform int useTexture = 0;

void main() {
    if (useTexture == 1) {
        FragColor = texture(tex0, texCoord) * fragColor;
    } else {
        FragColor = fragColor;
    }
}
