from langchain_core.documents import Document

chunk = Document(
    page_content=('합니다.<br>\uf000 설명서, 약관, 계약자 보관용 청약서 및 보험증권의 제공 사실에 관하여 계약자와<br>회사간에 다툼이 있는 '
 '경우에는 회사가 이를 증명하여야 합니다.<br>\uf000 보험설계사 등이 모집과정에서 사용한 회사 제작의 보험안내자료의 내용이 '
 '약관의<br>내용과 다른 경우에는 계약자에게 유리한 내용으로 계약이 성립된 것으로 봅니다.<br>용 어 풀 이 '
 "보험안내자료</p><br><p id='167' data-category='list'></p><br><table id='168'"),
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
 'indexing': {'chunk_id': 'chunk_000317',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
