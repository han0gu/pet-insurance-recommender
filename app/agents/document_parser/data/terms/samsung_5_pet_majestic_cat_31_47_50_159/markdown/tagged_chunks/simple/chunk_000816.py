from langchain_core.documents import Document

chunk = Document(
    page_content=('(Metacarpalbones) (Shaftofmetacarpal)\n'
 '중수골저 대능형골(Trapezium)\n'
 '(Baseofmetacarpal)\n'
 '소능형골(Trapezoid)\n'
 '유구골(Hamate)\n'
 '두상골(Psiform) 유두골(Capitate)\n'
 '상각골(Titquetrum)\n'
 '수근골\n'
 '요골경상돌기\n'
 '(Carpalbones) 월상골(Lunate) (Stylaidprocess ofradius)\n'
 '척골경상돌기 척골 요골 주상골\n'
 '(Styloidprocess ofulna) (Ulna) (Radius) (Scaphoid)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000816',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
