from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만,<br>제2조(제출서류) 제1항에 따라 제출된 장애인증명서상 장애예상기간(또는 장애기간)이<br>종료됨에 따라 제1조(특별약관의 '
 "적용범위) 제1항 제2호에서 정한 조건을 만족하지 않<br>게 된 경우에는 이 조항이 적용되지 않습니다.</p><h1 id='71' "
 "style='font-size:14px'>【예시】</h1><br><p id='72' data-category='paragraph' "
 "style='font-size:14px'>2019년 1월 15일에 전환대상계약에 가입한 계약자가 2019년 6월 1일에 이"),
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
 'indexing': {'chunk_id': 'chunk_000385',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
