#ifndef SOMPIXMAP_H
#define SOMPIXMAP_H

#include <Qt/QtGui>


class SOMReader : public QLabel
{
    Q_OBJECT

private:
    QImage *m_pImage;

    unsigned int m_iFieldSize;
    unsigned int m_iWidth;
    unsigned int m_iHeight;

public:
    explicit SOMReader(const unsigned int &iWidth, const unsigned int iHeight, // Height and Width in fields
                       const unsigned int &iFieldSize = 10,                    // Size of a field in pixels
                       QWidget *parent = 0);
    virtual ~SOMReader();

    void Resize(const unsigned int &iWidth, const unsigned int iHeight,
            	const unsigned int &iFieldSize = 10);

    void Save(const QString &sFileName);
    
public slots:
	void Fill(const QColor &color = Qt::white);
	void SetField(const QPoint &pField, const QColor &color);
	void SetField(const QPoint &pField, const std::vector<float> &vColor);
};

#endif // SOMPIXMAP_H
