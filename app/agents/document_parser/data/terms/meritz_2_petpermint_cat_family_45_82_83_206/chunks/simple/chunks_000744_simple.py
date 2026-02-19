from langchain_core.documents import Document

chunk = Document(
    page_content=('. 타) 각종 기질성 정신장해와 외상후 뇌전증에 한하여 보상한다. 파) 외상후 스트레스장애, 우울증(반응성) 등의 질환, '
 '정신분열증(조현병), 편집증, 조울증(양극성장 애), 불안장애, 전환장애, 공포장애, 강박장애 등 각종 신경증 및 각종 인격장애는 보상의 '
 '대상이 되지 않는다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 203},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000744',
              'chunk_char_len': 156,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
