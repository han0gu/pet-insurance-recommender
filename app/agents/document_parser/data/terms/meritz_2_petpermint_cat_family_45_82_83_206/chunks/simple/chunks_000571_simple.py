from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항의 전산화단층촬영(CT)이라 함은 제1조(보험금의 지급사유)에서 정한 수의사에 의하여 진단 및 치료가 필요 하다고 '
 '인정된 경우로서 수의사의 관리 하에 전산화단층촬 영(CT)을 사용하는 촬영 의료행위를 말합니다.\n'
 '\uf000 제1항의 내시경처치라 함은 제1조(보험금의 지급사유)에 서 정한 수의사에 의하여 진단 및 치료가 필요하다고 인정 된 경우로서 '
 '수의사의 관리 하에 내시경을 이용하여 비침습 적으로 시행하는 의료행위를 말하며, 식도, 위 또는 장에 시행하는 경우에 한합니다.\n'
 '【자기공명영상(MRI)】'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 164},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000571',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
