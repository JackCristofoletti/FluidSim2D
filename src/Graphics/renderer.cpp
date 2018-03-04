#include "renderer.h"

#include <iostream>
#include <QMatrix4x4>

Renderer::Renderer()
	: screenQuadVao_( 0 )
	, screenQuadVbo_( 0 )
	, solidbodyQuadVao_( 0 )
	, solidbodyQuadVbo_( 0 )
{

}

void Renderer::InitGL()
{
	initializeOpenGLFunctions();

	float screenQuadVertices[] = {
		// Positions        // Texture Coords
		-1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	};
	// Setup screen VAO
	glGenVertexArrays( 1, &screenQuadVao_ );
	glGenBuffers( 1, &screenQuadVbo_ );
	glBindVertexArray( screenQuadVao_ );
	{
		glBindBuffer( GL_ARRAY_BUFFER, screenQuadVbo_ );
		glBufferData( GL_ARRAY_BUFFER, sizeof( screenQuadVertices ), &screenQuadVertices, GL_STATIC_DRAW );
		glEnableVertexAttribArray( 0 );
		glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof( float ), (GLvoid*)0 );
		glEnableVertexAttribArray( 1 );
		glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof( float ), (GLvoid*)(3 * sizeof( float )) );
	}
	glBindVertexArray( 0 );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	//setup buffers for solidbody quads. Simulation Domain is [ (0,0) - (1,1) ] Will be scaled as needed
	float solidQuadVertices[] = {
		// Positions        // Texture Coords
		-0.5f,  0.5f, 0.0f, 0.0f, 1.0f,
		-0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
		 0.5f,  0.5f, 0.0f, 1.0f, 1.0f,
		 0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
	};

	glGenVertexArrays( 1, &solidbodyQuadVao_ );
	glGenBuffers( 1, &solidbodyQuadVbo_ );
	glBindVertexArray( solidbodyQuadVao_ );
	{
		glBindBuffer( GL_ARRAY_BUFFER, solidbodyQuadVbo_ );
		glBufferData( GL_ARRAY_BUFFER, sizeof( solidQuadVertices ), &solidQuadVertices, GL_STATIC_DRAW );
		glEnableVertexAttribArray( 0 );
		glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof( float ), (GLvoid*)0 );
		glEnableVertexAttribArray( 1 );
		glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof( float ), (GLvoid*)(3 * sizeof( float )) );
	}
	glBindVertexArray( 0 );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	textureRenderer_.reset( new QOpenGLShaderProgram() );
	textureRenderer_->addShaderFromSourceFile( QOpenGLShader::Vertex, "C:/Users/Jack/Desktop/FluidSimulation/shaders/quad.vert" );
	textureRenderer_->addShaderFromSourceFile( QOpenGLShader::Fragment, "C:/Users/Jack/Desktop/FluidSimulation/shaders/quad.frag" );
	textureRenderer_->link();

	solidQuadRenderer_.reset( new QOpenGLShaderProgram() );
	solidQuadRenderer_->addShaderFromSourceFile( QOpenGLShader::Vertex, "C:/Users/Jack/Desktop/FluidSimulation/shaders/solidBodyQuad.vert" );
	solidQuadRenderer_->addShaderFromSourceFile( QOpenGLShader::Fragment, "C:/Users/Jack/Desktop/FluidSimulation/shaders/solidBodyQuad.frag" );
	solidQuadRenderer_->link();

	//generate solidbody texture
	QImage image( "C:/Users/Jack/Desktop/FluidSimulation/assets/metal050.jpg" );
	solidBodyTexture_.reset( new QOpenGLTexture( image.mirrored() ) );
	solidBodyTexture_->setMinificationFilter( QOpenGLTexture::LinearMipMapLinear );
	solidBodyTexture_->setMagnificationFilter( QOpenGLTexture::Linear );
}

void Renderer::DestroyGL()
{
}

void Renderer::RenderTexture( QOpenGLTexture &texture )
{
	glBindVertexArray( screenQuadVao_ );
	textureRenderer_->bind();
	texture.bind();

	glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );

	glBindVertexArray( 0 );
	texture.release();
	textureRenderer_->release();
}

void Renderer::RenderTexture( unsigned texture )
{
	glBindVertexArray( screenQuadVao_ );
	textureRenderer_->bind();
	glUniform1i( textureRenderer_->uniformLocation( "ourTexture" ), 0 ); // texture unit 0 to "texImage"

	glBindTexture( GL_TEXTURE_2D, texture );

	glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );

	glBindVertexArray( 0 );
	glBindTexture( GL_TEXTURE_2D, 0 );
	textureRenderer_->release();
}

void Renderer::RenderSolidBodies( const std::vector<SolidBody>& solidBodies )
{
	std::vector<unsigned> box_indicies;

	
	QMatrix4x4 ortho;
	ortho.ortho( 0.0, 1.0, 0.0, 1.0, -1.0, 1.0 );

	//draw solid bodies
	solidQuadRenderer_->bind();
	{
		solidQuadRenderer_->setUniformValue( "projection", ortho );
		solidQuadRenderer_->setUniformValue( "view", QMatrix4x4() );

		solidBodyTexture_->bind();
		solidQuadRenderer_->setUniformValue( "solidBodyTexture" , 0 ); // texture unit 0 to "solidBodyTexture"

		glBindVertexArray( solidbodyQuadVao_ );

		for (const SolidBody& body : solidBodies)
		{
			QMatrix4x4 model;
			model.translate( body.posX, body.posY, 0.0 );
			//Only scale the transform for the box Circle fragments will get discarded in the shader
			if(body.type == SolidBody::BOX)
				model.scale( body.scaleX, body.scaleY, 1.0 );
			
			solidQuadRenderer_->setUniformValue( "model",  model );
			solidQuadRenderer_->setUniformValue( "radius", body.scaleX );
			solidQuadRenderer_->setUniformValue( "isCircle", body.type == SolidBody::BOX ? 0.0f : 1.0f );
			glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );
		}

		glBindVertexArray( 0 );

		solidBodyTexture_->release();
	}
	solidQuadRenderer_->release();
}
