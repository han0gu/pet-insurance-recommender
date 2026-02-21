from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우에 회사의 요구가 있으면 계약자 또는 피보험자는 이에 협력하여야<br>합니다.<br>\uf000 계약자 또는 피보험자가 정당한 '
 '이유없이 제2항, 제3항의 요구에 협조하지 않았<br>을 때에는 회사는 그로 인하여 늘어난 손해를 보상하지 '
 "않습니다.</p><br><table id='207' "
 "style='font-size:20px'><thead></thead><tbody><tr><td><table><thead></thead><tbody><tr><td>예 "
 '시</td><td>특</td></tr><tr><td>손해배상청구에 ① 손해배상책임'),
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
 'indexing': {'chunk_id': 'chunk_001189',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
