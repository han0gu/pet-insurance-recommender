from langchain_core.documents import Document

chunk = Document(
    page_content=('- 평균 운동가능영역을 기준으로 정상각도 및 측정방법 등을 따른다.\n'
 'KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 147- 147 -나) 관절기능장해를 표시할 경우 장해부위의 장해각도와 정상부위의 '
 '측\n'
 '정치를 동시에 판단하여 장해상태를 명확히 한다. 단, 관절기능장해- 가 신경손상으로 인한 경우에는 운동범위 측정이 아닌 근력 및 근전\n'
 '- 도 검사를 기준으로 평가한다.\n'
 '- 7) ‘관절 하나의 기능을 완전히 잃었을 때’라 함은 아래의 경우 중 하나에\n'
 '- 해당하는 경우를 말한다.\n'
 '- 가) 완전 강직(관절굳음)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000894',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
