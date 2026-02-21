from langchain_core.documents import Document

chunk = Document(
    page_content=('. 생명보험<br>2. 상해보험<br>3. 화재, 도난이나 그 밖의 손해를 담보하는 가계에 관한 손해보험<br>4. "수산업협동조합법", '
 '"신용협동조합법" 또는 "새마을금고법"에 따른<br>공제<br>5. "군인공제회법", "한국교직원공제회법", "대한지방행정공제회법", '
 '"<br>경찰공제회법" 및 "대한소방공제회법"에 따른 공제<br>6'),
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
 'indexing': {'chunk_id': 'chunk_001413',
              'chunk_char_len': 189,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
