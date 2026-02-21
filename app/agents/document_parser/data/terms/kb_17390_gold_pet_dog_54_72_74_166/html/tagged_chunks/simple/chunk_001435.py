from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 제2조(제출서류) 제1항에 따라 제출된 장애인증<br>명서상 장애예상기간(또는 장애기간)이 종료됨에 따라 전환대상계약이 '
 '제1조(적용<br>범위) 제1항 제2호에서 정한 조건을 만족하지 않게 된 경우에는 이 조항이 적용되<br>지 않습니다.</p><br><p '
 "id='102' data-category='paragraph' style='font-size:14px'>제4조(전환 "
 "취소)</p><br><h1 id='103' style='font-size:14px'>계약자는</h1><br><p id='104'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001435',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
