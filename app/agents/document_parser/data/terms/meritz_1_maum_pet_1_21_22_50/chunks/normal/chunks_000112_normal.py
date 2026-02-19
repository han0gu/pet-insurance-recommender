from langchain_core.documents import Document

chunk = Document(
    page_content=('【강제집행】\n'
 '사법상 또는 행정법상의 의무를 이행하지 아니하는 사람에 대하여 국가가 강제 권 력으로 그 의무의 이행하는 것을 말합니다.\n'
 '【담보권실행】\n'
 '담보권을 설정한 채권자가 채무를 이행하지 아니하는 채무자에 대하여 해당 담보권 을 실행하는 것을 말합니다.\n'
 '【국세 및 지방세 체납처분 절차】\n'
 '국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하여 체납된 세금에 대하여 가산금징수, 독촉장 발부 및 재산 압류 등의 '
 '집행을 하는 것을 말합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000112',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
