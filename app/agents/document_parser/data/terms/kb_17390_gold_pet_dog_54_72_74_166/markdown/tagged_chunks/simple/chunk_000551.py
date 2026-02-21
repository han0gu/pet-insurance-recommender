from langchain_core.documents import Document

chunk = Document(
    page_content=('| \uf000 사고로 이 특별약관의 만료된 경우에도 만료일부터 이내의 드립니다. 다만, 사고일 또는 발병일부터 365일 치료인 경우에 '
 '한합니다. \uf000 경우 30일이 지난날의 다음날로 합니다. 단, 원인으로 치료를 받은 경우에는 보험계약일을 보장개시일로 합니다. '
 '계약일은 보험료를 받은 날로 합니다. \uf000 제1항 내지 제4항에도 이 특별약관의 보험계약일로부터 그날을 포함하 여 1년 이내에 '
 '발생한 슬관절탈구, 고관절탈구, 슬관절형성부전, 고관절형성부전 또는 기타 이들과 유사한 사고에 대해서는 보험금을 지급하지 않습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000551',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
