from langchain_core.documents import Document

chunk = Document(
    page_content=('회사가 인정하는 의료기관에서 전문의 자격증을 가진 자가 실시한 진<br>단 결과 피보험자의 남은 생존기간이 6개월 이내라고 판단한 경우에 '
 '회사의 신청<br>서에 정한 바에 따라 사망보험금의 50%를 선지급 사망보험금(이하 "보험금"이라</p><p id=\'204\' '
 "data-category='paragraph' style='font-size:16px'>- 132 -</p><p id='205' "
 "data-category='list' style='font-size:16px'>합니다)으로 피보험자에게 지급합니다.<br>\uf000 이 "
 '특별약관의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001334',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
