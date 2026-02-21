from langchain_core.documents import Document

chunk = Document(
    page_content=('. 지진, 분화, 해일, 홍수 또는 이와 유사한 자연재해로 생긴 손해<br>3. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, '
 '소요, 기타 이들과 유사한 사태<br>4'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000033',
              'chunk_char_len': 97,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
