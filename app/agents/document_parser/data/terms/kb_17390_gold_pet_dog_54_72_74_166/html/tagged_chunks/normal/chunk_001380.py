from langchain_core.documents import Document

chunk = Document(
    page_content=('시 3세인 피보험자 반려동물이 3년만기로 갱신하는 경우 아래 예시에서 최초 계약시 납입할 특약보험료는 가입시점의 3세 보험료인 특 '
 '5,000원입니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001380',
              'chunk_char_len': 82,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
