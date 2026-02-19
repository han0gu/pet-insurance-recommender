from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 생명보험 2. 상해보험 3. 화재·도난이나 그 밖의 손해를 담보하는 가계에 관한 손해보험 '
 '4.「수산업협동조합법」,「신용협동조합법」또는「새마을금고법」에 따른 공제 '
 '5.「군인공제회법」,「한국교직원공제회법」,「대한지방행정공제회법」,「경찰공제회 법」및「대한소방공제회법」에 따른 공제 6. 주택 임차보증금의 '
 '반환을 보증하는 것을 목적으로 하는 보험·보증. 다만, 보증대 상 임차보증금이 3억원을 초과하는 경우는 제외한다.\n'
 '【소득세법 시행규칙 제61조의3 (공제대상보험료의 범위)】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 44},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000239',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
