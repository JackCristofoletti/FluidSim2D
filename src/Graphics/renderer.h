#ifndef RENDERER_H
#define RENDERER_H

#include <QOpenGLFunctions_4_5_Compatibility>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <memory>

class Renderer : public QOpenGLFunctions_4_5_Compatibility
{
public:
   //initialization functions
   Renderer();
   void InitGL();

   //Rendering
   void RenderTexture( QOpenGLTexture& texture );
   void RenderTexture( unsigned texture);
private:
   std::unique_ptr<QOpenGLShaderProgram> texture_renderer_;
   unsigned quad_vbo_, quad_vao_;

};

#endif // RENDERER_H
