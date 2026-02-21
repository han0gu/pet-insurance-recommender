from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 부활<br>(효력회복)일을 보험계약일로 하여 제1조(보험금의 지급사유) 제4항 내지 '
 "제5항을<br>적용합니다.<br>제<br>도</p><br><p id='182' data-category='paragraph' "
 "style='font-size:16px'>제17조(보험료</p><h1 id='183' "
 "style='font-size:16px'>제7조(준용규정)</h1><br><h1 id='184' "
 "style='font-size:16px'>\uf000 이 특별약관에서</h1><br><p id='185'"),
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
 'indexing': {'chunk_id': 'chunk_000985',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
