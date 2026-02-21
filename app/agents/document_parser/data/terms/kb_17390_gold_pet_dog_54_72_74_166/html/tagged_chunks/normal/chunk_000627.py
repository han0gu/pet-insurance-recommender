from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만,</p><br><p id='153' data-category='paragraph' style='font-size:14px'>이 "
 '특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금의 지급), 제24조(계<br>약의 소멸) 및 제36조(중도인출)는 '
 "제외합니다.</p><p id='154' data-category='paragraph' style='font-size:14px'>90 KB "
 "금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><h1 id='155'"),
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
 'indexing': {'chunk_id': 'chunk_000627',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
