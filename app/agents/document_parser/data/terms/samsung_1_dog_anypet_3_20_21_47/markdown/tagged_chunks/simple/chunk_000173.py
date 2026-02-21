from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 상해보험\n'
 '- 3. 화재·도난이나 그 밖의 손해를 담보하는 가계에 관한 손해보험\n'
 '- 4. 「수산업협동조합법」 , 「신용협동조합법 또는 「새마을금고법」 에 따른 공제\n'
 '- 5. 「군인공제회법」 , 「한국교직원공제회법」 , 「대한지방행정공제회법」 , 「경찰공제회법」 및 「대한\n'
 '- 소방공제회법」 에 따른 공제\n'
 '- 6. 주택 임차보증금의 반환을 보증하는 것을 목적으로 하는 보험·보증. 다만, 보증대상 임차보증금이 3\n'
 '- 억원을 초과하는 경우는 제외한다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000173',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
