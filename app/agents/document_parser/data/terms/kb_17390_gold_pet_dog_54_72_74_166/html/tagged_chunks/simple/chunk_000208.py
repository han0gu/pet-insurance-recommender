from langchain_core.documents import Document

chunk = Document(
    page_content=(". 계약해당일로 계약해당일: 계약해당일:</td></tr></tbody></table><br><p id='19' "
 "data-category='paragraph' style='font-size:16px'>제24조(계약의 소멸)<br>피보험자의 사망으로 "
 '인하여 이 약관에서 규정하는 보험금 지급사유가 더 이상 발생할<br>공<br>수 없는 경우에는 이 계약은 그 때부터 효력이 없습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000208',
              'chunk_char_len': 212,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
