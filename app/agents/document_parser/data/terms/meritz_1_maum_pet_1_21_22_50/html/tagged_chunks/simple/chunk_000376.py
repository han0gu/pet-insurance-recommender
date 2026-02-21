from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장<br>애인전용보험으로 전환을 원할 경우 수익자 지정이 필요합니다.</p><p id='60' data-category='list' "
 "style='font-size:14px'>② 전환대상계약이 해지(解止) 또는 기타 사유로 효력이 없게 된 경우 또는 전환대상계약<br>이 "
 '제1항에서 정한 조건을 만족하지 않게 된 경우 이 특별약관은 그 때부터 효력이 없<br>습니다.<br>③ 제2조 제1항에 따라 제출된 '
 '장애인증명서상 장애예상기간(또는 장애기간)이 종료된 경<br>우에는 제3조 제1항에도 불구하고 이 특별약관은 그때부터 효력이'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000376',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
