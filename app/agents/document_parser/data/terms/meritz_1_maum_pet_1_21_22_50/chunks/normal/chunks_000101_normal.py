from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹음)로 안내하고자 할 때 다음 각 호의 요건을 모두 충족하는 경우에 '
 '「보험업감독규정」 제4-36조 제3항에 따른 전자 적 상품설명장치를 활용할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 16},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000101',
              'chunk_char_len': 121,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
