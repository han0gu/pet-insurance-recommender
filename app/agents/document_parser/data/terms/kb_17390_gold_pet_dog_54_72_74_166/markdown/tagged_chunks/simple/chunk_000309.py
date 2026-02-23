from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약관의 보험가입금액을 상해입원일당으로 보험수익자에게 지급합니다.\n'
 '\uf000 제1항의 상해입원일당의 지급일수는 1회 입원당 180일을 한도로 합니다.- 제2조(보험금 지급에 관한 세부규정)\n'
 '- \uf000 제1조(보험금의 지급사유)의 상해입원일당은 같은 상해의 치료를 목적으로 2회\n'
 '- 이상 입원한 경우 이를 1회 입원으로 보아 각 입원일수를 더합니다.\n'
 '- \uf000 피보험자가 보장개시일 이후 입원하여 치료를 받던 중 보험기간이 끝났을 때에도\n'
 '- 퇴원하기 전까지의 계속 중인 입원에 대하여는 제1조(보험금의 지급사유) 제2항'),
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
 'indexing': {'chunk_id': 'chunk_000309',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
