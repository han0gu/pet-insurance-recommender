from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 제2조(제출서류) 제1항에 따라 제출된 장애인증명서상<br>장애예상기간(또는 장애기간)이 종료됨에 따라 전환대상계약이 '
 '제1조(특별약관의 적용<br>범위) 제1항 제2호에서 정한 조건을 만족하지 않게 된 경우에는 이 조항이 '
 "적용되지<br>않습니다.</p><h1 id='76' style='font-size:14px'>제4조(전환 취소)</h1><br><p "
 "id='77' data-category='paragraph' style='font-size:14px'>계약자는 전환대상계약에 대하여 "
 '장애인전용보험으로의 전환을 취소할 수'),
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
 'indexing': {'chunk_id': 'chunk_000389',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
