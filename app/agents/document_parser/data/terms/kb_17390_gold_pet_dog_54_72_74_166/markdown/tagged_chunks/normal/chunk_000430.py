from langchain_core.documents import Document

chunk = Document(
    page_content=('최초입원일 보장재개\n'
 '최종입원일\n'
 '퇴원없이\n'
 '…\n'
 '계속 입원\n'
 '보장 보장제외 보장\n'
 '… |\n'
 '(180일)(180일)(180일)- \uf000 피보험자가 질병에 대한 보장개시일 이후 입원하여 치료를 받던 중 보험기간이\n'
 '- 끝났을 때에도 퇴원하기 전까지의 계속중인 입원기간에 대하여는 제1조(보험금의\n'
 '- 지급사유) 제2항에 따라 질병입원일당을 계속 지급합니다.\n'
 '- \uf000 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회\n'
 '- 사는 질병입원일당의 전부 또는 일부를 지급하지 않습니다.'),
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
 'indexing': {'chunk_id': 'chunk_000430',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
