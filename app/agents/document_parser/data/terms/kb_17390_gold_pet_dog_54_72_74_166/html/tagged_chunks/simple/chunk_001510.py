from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>나.</h1><br><table id='205' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>1)</td><td>장해판정기준 ‘코의 "
 '호흡기능을 완전히 잃었을 때’라 함은 일상생활에서 구강호흡의 보조를 받지 않는 상태에서 코로 숨쉬는 것만으로 정상적인 호흡을 할 수 '
 '없다는 것이 비강통기도검사 등 의학적으로 인정된 검사로 확인되는 경우를 말한다.</td></tr><tr><td>2)</td><td>‘코의 '
 '후각기능을 완전히 잃었을 때’라 함은'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_001510',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
