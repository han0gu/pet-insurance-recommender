from langchain_core.documents import Document

chunk = Document(
    page_content=('×\n'
 '이 계약의 지급보험금\n'
 '다른 계약이 없는 것으로 하여 각각\n'
 '계산한 지급보험금의 합계액\n'
 '② 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 따른 지급보험금 결정에는 영향을 미치지 않습니다.\n'
 '제11조(보험금 받는 방법의 변경)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 8},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000047',
              'chunk_char_len': 141,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
