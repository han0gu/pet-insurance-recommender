from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 때에는 회사는 그로 인하여 늘어난 손해를 보상하지 않습니다.\n'
 '| <table><thead></thead><tbody><tr><td>예 '
 '시</td><td>특</td></tr><tr><td>손해배상청구에 ① 손해배상책임 사고발생 별 피보험자 피해자 보험사 관 ③ '
 '대신하여</td><td>대한 회사의 해결 예시 ② 보험금 지급 청구 약 피보험자를</td></tr></tbody></table> '
 '항변으로써 대항가능 ※ 항변이란 어떤 일을 부당하다고 여겨 따지거나 반대하는 뜻을 밝힌다는 것 |\n'
 '| --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000688',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
