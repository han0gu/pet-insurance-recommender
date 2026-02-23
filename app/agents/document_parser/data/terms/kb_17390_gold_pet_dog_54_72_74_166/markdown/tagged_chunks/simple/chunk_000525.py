from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 제1항의 통지를 계약이 해지된 날부터 7일 이내에 하여야 합니다.\n'
 '- \uf000 보험수익자는 통지를 받은 날(제3항에 따라 계약자에게 통지된 경우에는 계약자\n'
 '- 가 통지를 받은 날을 말합니다)부터 15일 이내에 제1항의 절차를 이행할 수 있습\n'
 '니다.| 용 어 풀 이 | 질 |\n'
 '| --- | --- |\n'
 '강제집행∙강제집행이란 사법상 또는 행정법상의 의무를 이행하지 않는 사람에 대하여 국\n'
 '가가 강제 권력으로 그 의무를 이행하는 것을 말합니다.∙담보권실행담보권실행이란 담보권을 설정한 채권자가 채무를 이행하지 않는 채무자에 대 '
 '상'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000525',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
