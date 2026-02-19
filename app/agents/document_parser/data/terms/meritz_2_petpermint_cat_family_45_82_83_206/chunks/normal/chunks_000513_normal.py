from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 이 특별약관에 있어서 MRI,CT 및 내시경처치라 함은 자 기공명영상(MRI), 전산화단층촬영(CT) 및 내시경처치를 말 '
 '합니다. \uf000 제1항의 자기공명영상(MRI)이라 함은 제1조(보험금의 지 급사유)에서 정한 수의사에 의하여 진단 및 치료가 필요하 '
 '다고 인정된 경우로서 수의사의 관리 하에 자기공명영상 (MRI)을 사용하는 촬영 의료행위를 말합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 152},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000513',
              'chunk_char_len': 197,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
