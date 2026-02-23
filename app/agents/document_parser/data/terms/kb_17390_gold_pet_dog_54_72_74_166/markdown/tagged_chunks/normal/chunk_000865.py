from langchain_core.documents import Document

chunk = Document(
    page_content=('- 행동 장해 평가와 비교하여 그 중 높은 지급률 하나만 인정한다.\n'
 '- 12) ‘치아의 결손’이란 치아의 상실 또는 발치된 경우를 말하며, 치아의 일\n'
 '- 부 손상으로 금관치료(크라운 보철수복)를 시행한 경우에는 치아의 일\n'
 '- 부 결손을 인정하여 1/2개 결손으로 적용한다.\n'
 '- 13) 보철치료를 위해 발치한 정상치아, 노화로 인해 자연 발치된 치아, 보철\n'
 'KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 143- 143 -- \n'
 '(복합레진, 인레이, 온레이 등)한 치아, 기존 의치(틀니, 임플란트 등)'),
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
 'indexing': {'chunk_id': 'chunk_000865',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
