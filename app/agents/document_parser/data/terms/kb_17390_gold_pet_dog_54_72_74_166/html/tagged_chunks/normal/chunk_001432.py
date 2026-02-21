from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 제2조(제출서류)제1항에 따라 제출된 장애인증명서상 장애예상기간(또<br>는 장애기간)이 종료됨에 따라 제1조(적용범위) 제1항 '
 "제2호에서 정한 조건을 만족<br>하지 않게 된 경우에는 이 조항이 적용되지 않습니다.</p><br><table id='100' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>예 시</td><td>특별세액공제 대상 "
 '기간 예시</td></tr><tr><td colspan="2">2022년 1월 15일에 전환대상계약에 가입한 계약자가 2022년 6월'),
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
 'indexing': {'chunk_id': 'chunk_001432',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
