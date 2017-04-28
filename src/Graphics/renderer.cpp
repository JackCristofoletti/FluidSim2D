#include "renderer.h"

Renderer::Renderer()
{

}

void Renderer::InitGL()
{
    initializeOpenGLFunctions();

    float quadVertices[] = {
        // Positions        // Texture Coords
        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    };
    // Setup plane VAO
    glGenVertexArrays(1, &quad_vao_);
    glGenBuffers(1, &quad_vbo_);
    glBindVertexArray(quad_vao_);
    {
        glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)(3 * sizeof(float) ) );
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);


    texture_renderer_.reset( new QOpenGLShaderProgram() );
    texture_renderer_->addShaderFromSourceFile( QOpenGLShader::Vertex, "C:/Users/Jack/Desktop/FluidSimulation/shaders/quad.vert" );
    texture_renderer_->addShaderFromSourceFile( QOpenGLShader::Fragment, "C:/Users/Jack/Desktop/FluidSimulation/shaders/quad.frag" );
    texture_renderer_->link();
}

void Renderer::RenderTexture( QOpenGLTexture &texture )
{
    glBindVertexArray(quad_vao_);
    texture_renderer_->bind();
    texture.bind();

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glBindVertexArray(0);
    texture.release();
    texture_renderer_->release();
}

void Renderer::RenderTexture(unsigned texture)
{
	glBindVertexArray(quad_vao_);
	texture_renderer_->bind();
	GLint id = glGetUniformLocation(texture, "ourTexture");
	glUniform1i(id, 0); // texture unit 0 to "texImage"

	glBindTexture(GL_TEXTURE_2D, texture);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
	texture_renderer_->release();
}
