from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 제1항 및 제3항에 따라 계약이 해지된 경우 회사는 제33조(보험료의 환급) 제1항 제1 호에 따른 보험료를 계약자에게 '
 '지급합니다. ⑤ 계약자는 제1항에 따른 제척기간에도 불구하고 민법 등 관계 법령에서 정하는 바에 따 라 법률상의 권리를 행사할 수 '
 '있습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 18},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000115',
              'chunk_char_len': 149,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
