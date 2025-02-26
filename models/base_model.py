from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Product(Base):
    __tablename__ = "product"

    id = Column(Integer, primary_key=True, index=True)
    brand = Column(String(255))
    content = Column(Text)
    grade = Column(String(255))
    ingredients = Column(Text)
    main_accord = Column(String(255))
    name_en = Column(String(255))
    name_kr = Column(String(255))
    size_option = Column(String(255))
    time_stamp = Column(DateTime)
    category_id = Column(Integer)

class Note(Base):
    __tablename__ = "note"

    id = Column(Integer, primary_key=True, index=True)
    note_type = Column(String(255))
    time_stamp = Column(DateTime)
    product_id = Column(Integer, ForeignKey("product.id"))
    spice_id = Column(Integer)

class Spice(Base):
    __tablename__ = "spice"

    id = Column(Integer, primary_key=True, index=True)
    content_en = Column(Text)
    content_kr = Column(Text)
    name_en = Column(String(255))
    name_kr = Column(String(255))
    time_stamp = Column(DateTime)
    line_id = Column(Integer)

class ProductImage(Base):
    __tablename__ = "product_image"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(255))
    product_id = Column(Integer, ForeignKey("product.id"))

class SimilarText(Base):
    __tablename__ = "similar_text"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("product.id"))
    similar_product_id = Column(Integer, ForeignKey("product.id"))
    similarity_score = Column(Float)

class SimilarImage(Base):
    __tablename__ = "similar_image"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("product.id"))
    similar_product_id = Column(Integer, ForeignKey("product.id"))
    similarity_score = Column(Float)

class Similar(Base):
    __tablename__ = "similar"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("product.id"))
    similar_product_id = Column(Integer, ForeignKey("product.id"))
    similarity_score = Column(Float) 
    
class Review(Base):
    __tablename__ = "review"
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    time_stamp = Column(DateTime, nullable=False)
    member_id = Column(String(255), nullable=False)
    product_id = Column(Integer, ForeignKey("product.id"), nullable=False)
    
class Bookmark(Base):
   __tablename__ = "bookmark"

   id = Column(Integer, primary_key=True, index=True)
   time_stamp = Column(DateTime, nullable=False)
   member_id = Column(Integer, nullable=False)
   product_id = Column(Integer, ForeignKey("product.id"), nullable=False)