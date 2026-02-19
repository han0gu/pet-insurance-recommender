from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는 전자문서가 수신되지 않은 것을 확인한 경우에는 제1항에서 정한 내용을 서면(등기우편 등) 또는 전화(음성 녹음)로 다시 알려 '
 '드립니다. ⑥ 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹음)로 안내하고자 할 때 다음 각 호의 요건을 모두 충족하는 경우에 '
 '「보험업감독규정」 제4-36조 제3항에 따른 전자적 상품설명장치를 활용할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 43},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000133',
              'chunk_char_len': 200,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
