from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 신체검사, 정형검사, 신경계검사, 안검사, 피부과검사 등 기본검사 2. X-ray, 초음파검사, CT, MRI, 내시경검사 등 '
 '영상검사 3. 혈액검사, 임상병리검사, 조직병리검사, 배양검사 등 실험실검사'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 68},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['head',
                             'joint',
                             'eye',
                             'skin',
                             'digestive',
                             'other']},
 'indexing': {'chunk_id': 'chunk_000349',
              'chunk_char_len': 116,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
