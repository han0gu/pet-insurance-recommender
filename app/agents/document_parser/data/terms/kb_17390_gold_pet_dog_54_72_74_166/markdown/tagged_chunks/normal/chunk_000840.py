from langchain_core.documents import Document

chunk = Document(
    page_content=('을 받은 환자인 경우 각막이식술 이전의 시력상태를 기준으로 평가한다.\n'
 '3) ‘한눈이 멀었을 때’라 함은 안구의 적출은 물론 명암을 가리지 못하거\n'
 '나(‘광각무’) 겨우 가릴 수 있는 경우(‘광각유’)를 말한다.\n'
 '4) ‘한눈의 교정시력이 0.02이하로 된 때’라 함은 안전수동(Hand- \n'
 '- Movement)주 \ue034 \ue045, 안전수지(Finger Counting)주 \ue035 \ue045 상태를 포함한다.\n'
 '- 주1) 안전수동 : 물체를 감별할 정도의 시력상태가 아니며 눈앞에서 손\n'
 '- 의 움직임을 식별할 수 있을 정도의 시력상태'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000840',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
