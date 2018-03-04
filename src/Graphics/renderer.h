#ifndef RENDERER_H
#define RENDERER_H

#include <QOpenGLFunctions_4_5_Compatibility>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <memory>

#include "../Fluid/SolidBody.h"

class Renderer : public QOpenGLFunctions_4_5_Compatibility
{
public:
   //initialization functions
   Renderer();
   void InitGL();
   void DestroyGL();

   //Rendering
   void RenderTexture( QOpenGLTexture& texture );
   void RenderTexture( unsigned texture);
   void RenderSolidBodies( const std::vector<SolidBody>& solidBodies );

private:
   std::unique_ptr<QOpenGLShaderProgram> textureRenderer_;
   std::unique_ptr<QOpenGLShaderProgram> solidQuadRenderer_;
   std::unique_ptr<QOpenGLTexture>		 solidBodyTexture_;

   unsigned screenQuadVbo_, screenQuadVao_;
   unsigned solidbodyQuadVbo_, solidbodyQuadVao_;
};

#endif // RENDERER_H
