from langchain_core.documents import Document

chunk = Document(
    page_content=('지골두(Headofphalarx) ①\n'
 '지골 지골체(Shaftofphalarx) 제1말저골(Firstistalphalanx)\n'
 '(Phalanges) 근위지관절(제1지관절)\n'
 '자골저(Baseofphalara)\n'
 '(Pipjoint)\n'
 '중수골두 기저골(근위저금)Fistprainalplane\n'
 '증수지관절 지관절\n'
 '(Headofmetacarpal) 종자골(Sesamoidbones)\n'
 '(Mpjoint)\n'
 '중수골 중수골체\n'
 '제1중수(Res.metacarpal)\n'
 '(Metacarpalbones) (Shaftofmetacarpal)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000815',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
