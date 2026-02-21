from langchain_core.documents import Document

chunk = Document(
    page_content=("해당된 경우에는 이 특별약관의 보험금을 지급하지 않습니다.</p><h1 id='209' "
 "style='font-size:16px'>제4조(보험금의 지정대리청구인)</h1><br><p id='210' "
 "data-category='paragraph' style='font-size:16px'>\uf000 계약자가 이 특별약관의 보험금을 "
 '청구할 수 없는 특별한 사정이 있을 때에는 계<br>약자가 미리 지정하거나 또는 제5조(지정대리청구인의 변경지정)의 규정에 '
 '따라<br>변경지정한 다음의 자(이하"지정대리청구인"이라 합니다)가 제6조(보험금의'),
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
 'indexing': {'chunk_id': 'chunk_001340',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
