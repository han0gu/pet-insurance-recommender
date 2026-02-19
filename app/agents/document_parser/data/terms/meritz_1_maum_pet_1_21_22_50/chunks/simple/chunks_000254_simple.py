from langchain_core.documents import Document

chunk = Document(
    page_content=('다시 부가할 수 없습니다. 다만, 제2조(제출서류) 제1항에 따라 제출된 장애인증명서상 장애예상기간(또는 장애기간)이 종료됨에 따라 '
 '전환대상계약이 제1조(특별약관의 적용 범위) 제1항 제2호에서 정한 조건을 만족하지 않게 된 경우에는 이 조항이 적용되지 않습니다.\n'
 '제4조(전환 취소)\n'
 '계약자는 전환대상계약에 대하여 장애인전용보험으로의 전환을 취소할 수 있으며, 이 경우 전환취소 신청서를 회사에 제출하여야 합니다.\n'
 '제5조(준용규정)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 47},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000254',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
