from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹 음)로 안내하고자 할 때 다음 각 호의 요건을 모두 충족하 는 '
 '경우에「보험업감독규정」제4-36조 제3항에 따른 전자적 상품설명장치를 활용할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 77},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000132',
              'chunk_char_len': 120,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
