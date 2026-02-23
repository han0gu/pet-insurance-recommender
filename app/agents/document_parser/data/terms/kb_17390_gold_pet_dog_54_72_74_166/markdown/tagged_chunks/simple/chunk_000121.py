from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 만15세 미만자, 심신상실자(心神喪失者) 또는 심신박약자(心神薄弱者)를 피보\n'
 '- 험자로 하여 사망을 보험금 지급사유로 한 계약의 경우. 다만, 심신박약자가\n'
 '- 계약을 체결하거나 소속 단체의 규약에 따라 단체보험의 피보험자가 될 때에\n'
 '- 의사능력이 있는 경우에는 계약이 유효합니다.\n'
 '- 3. 계약을 체결할 때 계약에서 정한 피보험자의 나이에 미달되었거나 초과되었을\n'
 '- 경우. 다만, 회사가 나이의 착오를 발견하였을 때 이미 계약나이에 도달한 경\n'
 '- 우에는 유효한 계약으로 보나, 제2호의 만15세 미만자에 관한 예외가 인정되는'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000121',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
