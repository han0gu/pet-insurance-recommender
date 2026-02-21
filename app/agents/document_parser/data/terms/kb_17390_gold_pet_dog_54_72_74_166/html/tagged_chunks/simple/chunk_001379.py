from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자는 갱신일 현재의<br>약관 등에 대해 90일 이내에 그 계약을 취소할 수 있으며, 이 경우 회사는 계약자<br>에게 '
 "갱신일 이후 납입한 보장특약의 보험료를 돌려드립니다.</p><br><table id='26' "
 "style='font-size:20px'><thead><tr><td></td></tr></thead><tbody><tr><td><table><thead></thead><tbody><tr><td></td></tr><tr><td>예 "
 '시 3세인 피보험자 반려동물이 3년만기로 갱신하는 경우 아래 예시에서 최초 계약시'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001379',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
