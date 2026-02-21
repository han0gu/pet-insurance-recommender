from langchain_core.documents import Document

chunk = Document(
    page_content=('| 따라 보험금을 | 삭감하여 지급합니다. |\n'
 '| --- | --- |\n'
 '예 시∙ 계약해당일 계산\n'
 '최초계약일과 동일한 월, 일을 말합니다. 해당연도의 계약해당일이 없는 경우\n'
 '에는 해당 월의 마지막 날을 계약해당일로 합니다.\n'
 '계약일: 2022년 10월 1일 => 계약해당일: 10월 1일\n'
 '계약일: 2024년 2월 29일 => 계약해당일: 2월 말일# 제15조(제1회 보험료 및 회사의보장개시)- \uf000 회사는 계약의 '
 '청약을 승낙하고 제1회 보험료 등을 받은 때부터 이 특별약관이 정'),
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
 'indexing': {'chunk_id': 'chunk_000508',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
