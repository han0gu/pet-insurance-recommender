from langchain_core.documents import Document

chunk = Document(
    page_content=('보장제한부 인수 범위 및 사유를 설명하여 드립니다.\n'
 '\uf000 이 특별약관에 대한 회사의 보장개시일은 제4장 반려동물 관련 특별약관 반려동물\n'
 '(강아지) 일반조항 제15조(제1회 보험료 및 회사의 보장개시)에서 정한 보장개시\n'
 '일과 동일합니다.\n'
 '\uf000 보험계약이 해지, 기타 사유에 의하여 효력을 가지지 않게 된 경우에는 이 특별약\n'
 '관도 더 이상 효력을 가지지 않습니다.\n'
 '\uf000 보통약관 및 이 특별약관에서 정한 보장개시일 이전에 발생한 질병에 대하여 계약\n'
 '을 무효로 하는 경우에도 제2조(특별면책(회사가 보험금을 지급하지 않는)조건의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000821',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
